import random
import re
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..utils.llm.chatgpt import AzureChatGPT
from ..utils.llm.glm import GLM4
from ..utils.llm.llama import Llama2, Llama3
from ..utils.kg.query import SPARQLService, RDFLibSPARQL
from .prompt import entity_score_prompt, entity_recognition_prompt
from utils.logging import set_logger


class EntitySearch(object):
    
    def __init__(
        self, 
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3], 
        sparql: Union[SPARQLService, RDFLibSPARQL],
        width: int,
        max_candidate_entities: int,
        domain: Optional[str] = None, 
        entity_regex: Optional[str] = None,
        entity_mapping: Optional[Callable[[str], str]] = None,
        abandon_entity_names: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
        ) -> None:
        self.llm = llm
        self.sparql = sparql
        self.max_candidate_entities = max_candidate_entities
        self.width = width
        self.domain = domain
        self.entity_regex = entity_regex
        self.entity_mapping = entity_mapping if entity_mapping else lambda x: x
        self.abandon_entity_names = abandon_entity_names
        self.logger = logger if logger else set_logger('entity-search-stdout', level=logging.DEBUG)
        
    def __remove_domain(self, entities: List[str]) -> List[Dict[str, str]]:
        if self.domain:
            entities = [re.sub(f'^{self.domain}', '', entity) for entity in entities]
            self.logger.debug(f'detected domain `{self.domain}`, remove domain from entities:\n{entities}')
        return entities
        
    def __entity_search(self, entity: str, relation: str, search_tail: bool) -> List[str]:
        if search_tail:
            entities = self.sparql.get_tail_entity(entity, relation)
        else:
            entities = self.sparql.get_head_entity(relation, entity)
        entities = self.__remove_domain(entities)
        if self.entity_regex: 
            entities = [entity for entity in entities if re.search(self.entity_regex, entity)]
        self.logger.debug(f'obtained entities {entities} based on {entity = }, {relation = }, {search_tail = }')
        return entities

    def __extract_score(self, string: str, entities: List[str], entity_names: List[str]) -> Tuple[List[str], List[str], List[float]]:
        name_to_id = dict(zip(entity_names, entities))
        pattern = r"""{entity:\s+["'](?P<entity>[^"']+)["'],\s+score:\s+(?P<score>[0-9.]+)}"""
        extract_entity_names = list()
        scores  = list()
        for re_match in re.finditer(pattern, string):
            entity = re_match.group("entity")
            score  = re_match.group("score")
            # if it is fabricated entity, drop the entity and score.
            if entity not in entity_names:
                self.logger.warning(f'drop invalid entity `{entity}` in:\n{string}.')
                self.logger.warning(f'valid entity list: {entity_names}.')
                continue
            # if not a valid float format, drop the entity and score.
            try:
                score = float(score)
            except ValueError:
                self.logger.warning(f'drop invalid score `{score} in:\n{string}.')
                continue
            # if the entity has been extracted.
            if entity in extract_entity_names:
                self.logger.warning(f'drop duplicated entity `{entity} in:\n{string}.')
                continue
            # if the entity and score pass all criteria.
            extract_entity_names.append(entity)
            scores.append(score)
        # sometime all entities are fabricated, in that case we give an even score for all candidate entities.
        if len(extract_entity_names) == 0:
            self.logger.warning(f'assign even scores because it cannot match any valid entities in:\n{string}, .')
            return entities, entity_names, [round(1/len(entities),1) for _ in entities]
        else:
            # some entity will be dropped or in different order when calculating their score.
            # so it is needed to align the entity id with the successfully extracted entity names.
            extract_entities = [name_to_id[name] for name in extract_entity_names]
            assert len(extract_entities) == len(extract_entity_names) and len(extract_entities) == len(scores), \
                ValueError(f'observed mismatching in:\n{extract_entities = }\n{extract_entity_names = }\n{scores = }\nfrom:\n{string}')
            # the sum of scores output by LLM may be not equal to 1.0, so we need to re-weight it.
            scores = [round(score/(sum(scores)+1e-10),1) for score in scores]
            self.logger.debug(f'input entity number {len(entities)}, output entity number {len(extract_entities)}')
            self.logger.debug(f'extracted re-weighted {scores = } of {extract_entity_names = }')
            return extract_entities, extract_entity_names, scores

    def __entity_score(
        self,
        question: str,
        entities: List[str],
        relation_score: float,
        relation: str
        ) -> Tuple[List[float], List[str], List[str]]:
        entity_names = [self.entity_mapping(entity_id) for entity_id in entities]
        if self.abandon_entity_names:
            # all abandon entities.
            if all(entity_name in self.abandon_entity_names for entity_name in entity_names):
                self.logger.debug(f'observed all abandon entity: {entity_names}')
                return entities, entity_names, [1/len(entities) * relation_score] * len(entity_names)
            # find abandon entities and remove.
            if len(entity_names) > 1:
                    entity_names = [entity_name for entity_name in entity_names if entity_name not in self.abandon_entity_names]
                    entities     = [entity for entity, entity_name in zip(entities, entity_names) if entity_name not in self.abandon_entity_names]
        prompt = entity_score_prompt(question=question, relation=relation, entities=entity_names)
        self.logger.debug(f'entity score prompt: {prompt}')
        result = self.llm.chat(prompt)
        self.logger.debug(f'llm score feedback: {result}')
        entities, entity_names, scores = self.__extract_score(string=result, entities=entities, entity_names=entity_names)
        scores = [score * relation_score for score in scores]
        self.logger.debug(f'relation weighted entity scores: {scores}')
        return entities, entity_names, scores

    def __entity_prune(
        self, 
        entities: List[str], 
        entity_names: List[str], 
        scores: List[float], 
        relations: List[str], 
        pre_entities: List[str], 
        heads: List[bool]
        ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        pre_entity_names = [self.entity_mapping(entity) for entity in pre_entities]
        keys = ['entity', 'entity_name', 'score', 'relation', 'pre_entity', 'pre_entity_name', 'head']
        candidates = list(map(lambda values: dict(zip(keys, values)), zip(entities, entity_names, scores, relations, pre_entities, pre_entity_names, heads)))
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        candidates = candidates[:self.width]
        self.logger.debug(f'top {self.width} candidate entities: {candidates}')
        triples  = list()
        entities = list()
        for record in candidates:
            entities.append(record['entity'])
            if record['head']:
                triples.append((record['pre_entity_name'], record['relation'], record['entity_name']))
            else:
                triples.append((record['entity_name'], record['relation'], record['pre_entity_name']))
        self.logger.debug(f'obtained {triples = }')
        return triples, entities

    def __call__(self, question, relations):
        recorder = EntitySearchRecorder()
        for relation in relations:
            entities = self.__entity_search(relation['entity'], relation['relation'], search_tail=relation['head'])
            if len(entities) >= self.max_candidate_entities:
                entities = random.sample(entities, self.max_candidate_entities)
            if len(entities) == 0:
                continue
            entities, entity_names, scores = self.__entity_score(question, entities, relation['score'], relation['relation'])
            recorder.update(entities, entity_names, scores, relation['relation'], relation['entity'], relation['head'])
        return self.__entity_prune(**recorder.get_dict())


class EntitySearchRecorder(object):
    
    def __init__(self) -> None:
        self.entities     = list()
        self.entity_names = list()
        self.scores       = list()
        self.relations    = list()
        self.pre_entities = list()
        self.heads        = list() # is pre_entity is head entity
        
    def update(
        self,
        entities: List[str], 
        entity_names: List[str], 
        scores: List[float], 
        relation: str, 
        pre_entity: str, 
        head: bool
        ) -> None:
        self.entities.extend(entities)
        self.entity_names.extend(entity_names)
        self.scores.extend(scores)
        self.relations.extend([relation] * len(entities))
        self.pre_entities.extend([pre_entity] * len(entities))
        self.heads.extend([head] * len(entities))
        
    def get(self) -> Tuple[List[str], List[str], List[float], List[str], List[str], List[bool]]:
        return self.entities, self.entity_names, self.scores, self.relations, self.pre_entities, self.heads
    
    def get_dict(self) -> Dict[str, Union[List[str], List[float], List[bool]]]:
        return dict(
            entities=self.entities,
            entity_names=self.entity_names,
            scores=self.scores, 
            relations=self.relations, 
            pre_entities=self.pre_entities, 
            heads=self.heads
        )
        
class EntityMatcher(object):
    
    def __init__(
        self, 
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3], 
        sparql: Union[SPARQLService, RDFLibSPARQL],
        width: int,
        domain: Optional[str] = None, 
        entity_regex: Optional[str] = None,
        entity_mapping: Optional[Callable[[str], str]] = None,
        abandon_entity_names: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
        ) -> None:
        self.llm = llm
        self.sparql = sparql
        self.width = width
        self.domain = domain
        self.entity_regex = entity_regex
        self.entity_mapping = entity_mapping if entity_mapping else lambda x: x
        self.abandon_entity_names = abandon_entity_names
        self.logger = logger if logger else set_logger('entity-match-stdout', level=logging.DEBUG)
        # preprocess input data.
        self.entity_names, self.name_to_id = self.__list_all_entities()
        self.entity_embeds = torch.cat([self.llm.embedding(e) for e in self.entity_names], dim=0)
    
    def __call__(self, start_entity_name: Union[str, List[str]]) -> List[str]:
        if isinstance(start_entity_name, str):
            return self.__match_point(start_entity_name)
        elif isinstance(start_entity_name, list):
            return self.__match_points(start_entity_name)
        else:
            raise ValueError(f'unsupported start entity type {type(start_entity_name)}')
    
    def __list_all_entities(self) -> Tuple[List[str], Dict[str, str]]:
        entities = list(set(self.sparql.get_all_head_entity() + self.sparql.get_all_tail_entity()))
        entities = self.__remove_domain(entities)
        self.logger.debug(f'entity matcher obtains totally {len(entities)} different entities.')
        entity_names = list()
        name_to_id   = dict()
        for entity_id in entities:
            entity_name = self.entity_mapping(entity_id)
            entity_names.append(entity_name)
            name_to_id[entity_name] = entity_id
        self.logger.debug(f'obtained entity names, the head 100 names are {entity_names[:100]}')
        return entity_names, name_to_id
    
    def __match_point(self, start_entity_name: str) -> List[str]:
        if start_entity_name in self.entity_names:
            self.logger.info(f'matched start entity {start_entity_name} in entities.')
            return [self.name_to_id[start_entity_name]]
        else:
            embed          = self.llm.embedding(start_entity_name)
            similarity     = F.cosine_similarity(embed, self.entity_embeds, dim=1)
            _, index       = torch.topk(similarity, k=self.width)
            start_entities = [self.entity_names[i] for i in index.tolist()]
            self.logger.info(f'cannot matched start entity {start_entity_name}, find similar entities: {start_entities}.')
            start_entities = [self.name_to_id[name] for name in start_entities]
            return start_entities
    
    def __match_points(self, start_entities: List[str]) -> List[str]:
        start_entities_ = list()
        for start_entity_name in start_entities:
            if start_entity_name in self.entity_names:
                self.logger.info(f'matched start entity {start_entity_name} in entities.')
                start_entities_.append(self.name_to_id[start_entity_name])
            else:
                embed      = self.llm.embedding(start_entity_name)
                similarity = F.cosine_similarity(embed, self.entity_embeds, dim=1)
                i          = torch.argmax(similarity).item()
                start_entity_name = self.entity_names[i]
                self.logger.info(f'cannot matched start entity {start_entity_name}, find similar entities: {start_entity_name}.')
                start_entities_.append(self.name_to_id[start_entity_name])
        return start_entities_
    
    def __remove_domain(self, entities: List[str]) -> List[Dict[str, str]]:
        if self.domain:
            entities = [re.sub(f'^{self.domain}', '', entity) for entity in entities]
        return entities
    
    
class QuestionEntityRecognition(object):
    
    def __init__(
        self, 
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3],
        width: int,
        logger: Optional[logging.Logger] = None,
        **kwargs
        ) -> None:
        self.llm = llm
        self.width = width
        self.logger = logger if logger else set_logger('ner-stdout', level=logging.DEBUG)
    
    def __call__(self, question: Dict[str, str]) -> List[str]:
        prompts = entity_recognition_prompt()
        results = self.llm.chat(prompts)
        return self.__extract_entities(results)
        
    
    def __extract_entities(self, string: str) -> List[str]:
        matches = re.findall(r'\{?' + 'entities' + r'\s*[:=]\s*(.+?)(?=\s*}|$)', string)
        if len(matches) == 0:
            return []
        else:
            string = re.sub(r'\s+', ' ', matches[0])
            string = re.sub(r'[\[\]]', '', string)
            entities = string.split(',')
            entities = [re.sub(r"""["']""", '', entity).strip() for entity in entities]
        return entities