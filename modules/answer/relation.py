import random
import re
import logging
from typing import Callable, Dict, List, Optional, Union

from ..utils.llm.chatgpt import AzureChatGPT
from ..utils.llm.glm import GLM4
from ..utils.llm.llama import Llama2, Llama3
from ..utils.kg.query import SPARQLService, RDFLibSPARQL
from .prompt import relation_prune_prompt
from utils.logging import set_logger


class RelationSearch(object):
    
    def __init__(
        self, 
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3], 
        sparql: Union[SPARQLService, RDFLibSPARQL],
        width: int,
        max_candidate_relations: int,
        domain: Optional[str] = None,
        entity_mapping: Optional[Callable[[str], str]] = None,
        abandon_relations: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
        ) -> None:
        self.llm = llm
        self.sparql = sparql
        self.max_candidate_relations = max_candidate_relations
        self.width = width
        self.domain = domain
        self.entity_mapping = entity_mapping if entity_mapping else lambda x: x
        self.abandon_relations = abandon_relations
        self.logger = logger if logger else set_logger('relation-search-stdout', level=logging.DEBUG)
    
    def __remove_domain(self, relations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if self.domain:
            relations = [re.sub(f'^{self.domain}', '', relation) for relation in relations]
            self.logger.debug(f'detected domain `{self.domain}`, remove domain from relations:\n{relations}')
        return relations
        
    def __is_abandon_relation(self, relation: str) -> bool:
        if self.abandon_relations:
            if any([re.search(abandon_relation, relation) for abandon_relation in self.abandon_relations]):
                self.logger.debug(f'found relation `{relation}` listed in abandon relations = {self.abandon_relations}')
                return True
            else:
                return False
        else:
            return False
    
    def __format_relations(
        self, 
        string: str, 
        entity: str,
        total_relations: List[str],
        head_relations: List[str],
        width: int
        ) -> List[Dict[str, Union[float, str, bool]]]:
        pattern = r"{relation:\s+\"(?P<relation>[^\"]+)\",\s+score:\s+(?P<score>[0-9.]+)}"
        extract_relations = list()
        for re_match in re.finditer(pattern, string):
            relation = re_match.group("relation")
            score    = re_match.group("score")
            # if it is a fabricated relations.
            if relation not in total_relations:
                self.logger.warning(f'drop fabricated relation `{relation}` in:\n{string}')
                continue
            # if not a valid float format.
            try:
                score = float(score)
            except ValueError:
                self.logger.warning(f'drop invalid score `{score}` in:\n{string}')
                continue
            # if the relation has been extracted.
            if any([relation == extract_relation['relation'] for extract_relation in extract_relations]):
                self.logger.warning(f'drop duplicated relation `{relation} in:\n{string}.')
                continue
            # if the relation passes all check.
            head = True if relation in head_relations else False
            extract_relations.append(dict(entity=entity, relation=relation, score=score, head=head))
        # if no relations are extracted, we randomly select some valid relations.
        if len(extract_relations) == 0:
            self.logger.warning(f'randomly select relations because no relation found from: {string}')
            # if no enough candidate relations.
            if len(total_relations) <= width:
                for relation in total_relations:
                    score = round(1/len(total_relations),1)
                    head  = True if relation in head_relations else False
                    extract_relations.append(dict(entity=entity, relation=relation, score=score, head=head)) 
            else:
                relations = random.sample(total_relations, width)
                for relation in relations:
                    score = round(1/len(relations),1)
                    head  = True if relation in head_relations else False
                    extract_relations.append(dict(entity=entity, relation=relation, score=score, head=head)) 
        else:
            # the sum of scores output by LLM may be not equal to 1.0, so we need to re-weight it.
            score_sum = sum([relation['score'] for relation in extract_relations])
            for relation in extract_relations:
                relation['score'] = round(relation['score']/(score_sum + 1e-10), 1)
            self.logger.debug(f'obtained score-reweighted `{extract_relations = }` from `{string}`')
        return extract_relations
    
    def __call__(
        self, 
        entity: str, 
        pre_relations: List[str], 
        question: str
        ) -> List[Dict[str, Union[float, str, bool]]]:
        head_relations = self.sparql.get_relation_by_head(entity)
        tail_relations = self.sparql.get_relation_by_tail(entity)
        head_relations = list(set(head_relations))
        tail_relations = list(set(tail_relations))
        head_relations = self.__remove_domain(head_relations)
        tail_relations = self.__remove_domain(tail_relations)
        head_relations = [relation for relation in head_relations if not self.__is_abandon_relation(relation)]
        tail_relations = [relation for relation in tail_relations if not self.__is_abandon_relation(relation)]
        self.logger.debug(f'obtained `{head_relations = }`')
        self.logger.debug(f'obtained `{tail_relations = }`')
        
        # remove relations already considered.
        if len(pre_relations) != 0:
            head_relations = [rel for rel in head_relations if rel not in pre_relations]
            tail_relations = [rel for rel in tail_relations if rel not in pre_relations]
            self.logger.debug(f'remove relations already considered, the remaining {head_relations = }\n{tail_relations = }')
        
        total_relations = list(set(head_relations) | set(tail_relations))
        # sample relations to avoid exceeding maximum token limitation (recommend 200 for BreeBase).
        if len(total_relations) > self.max_candidate_relations:
            total_relations = random.sample(total_relations, self.max_candidate_relations)
            head_relations  = list(set(total_relations) & set(head_relations))
            tail_relations  = list(set(total_relations) & set(tail_relations))
            self.logger.debug(f'sampled {total_relations = }')

        entity_name = self.entity_mapping(entity)
        prompt = relation_prune_prompt(question=question, entity=entity_name, relations=total_relations, width=self.width)
        self.logger.debug(f'relation prune prompt: {prompt}')
        result = self.llm.chat(prompt)
        self.logger.debug(f'llm pruning feedback: {result}')
        
        pruned_relations = self.__format_relations(string=result, entity=entity, total_relations=total_relations, head_relations=head_relations, width=self.width)
        return pruned_relations