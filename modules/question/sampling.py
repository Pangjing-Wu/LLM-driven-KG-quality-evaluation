import copy
import logging
import random
import re
from typing import Callable, Dict, List, Optional, Set, Union

import community as community_louvain
import networkx as nx

from ..utils.llm import AzureChatGPT, GLM4, Llama2, Llama3
from ..utils.kg.query import RDFLibSPARQL
from .prompt import start_entity_prune_prompt
from utils.logging import set_logger


class QuestionBase(object):
    
    def __init__(
        self,
        sparql: RDFLibSPARQL,
        cluster_level: Optional[int] = None,
        entity_mapping: Optional[Callable[[str], str]] = None,
        entity_regex: Optional[str] = None,
        abandon_relations: Optional[List[str]] = None,
        abandon_entity_names: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
        ) -> None:
        self.sparql = copy.deepcopy(sparql)
        self.cluster_level = cluster_level
        self.entity_mapping = lambda x:x if entity_mapping is None else entity_mapping
        self.entity_regex = entity_regex
        self.abandon_relations = abandon_relations
        self.abandon_entity_names = abandon_entity_names
        self.logger = logger if logger else set_logger('question-base-stdout', level=logging.DEBUG)
        self.__clean_kg()
        self.__cluster = self.__clustering()
    
    def __len__(self):
        return len(self.__cluster)
    
    @property
    def question_topic(self) -> Dict[int, Set[str]]:
        return copy.deepcopy(self.__cluster)
    
    def __clean_kg(self):
        self.logger.info(f'KG have {len(self.sparql.kg)} triples.')
        for e in self.sparql.get_all_head_entity():
            if (self.abandon_entity_names and self.entity_mapping(e) in self.abandon_entity_names) \
                or (self.entity_regex and not re.search(self.entity_regex, e)):
                self.sparql.kg.remove((e, None, None))
        for e in self.sparql.get_all_tail_entity():
            if (self.abandon_entity_names and self.entity_mapping(e) in self.abandon_entity_names) \
                or (self.entity_regex and not re.search(self.entity_regex, e)):
                self.sparql.kg.remove((None, None, e))
        for r in self.sparql.get_all_relations():
            if self.abandon_relations and r in self.abandon_relations:
                self.sparql.kg.remove((None, r, None))
        self.logger.info(f'After clean, KG still have {len(self.sparql.kg)} triples.')
    
    def __clustering(self) -> Dict[int, Set[str]]:
        nx_graph = nx.Graph()
        for h, _, t in self.sparql.kg:
            nx_graph.add_edge(h, t)
        if isinstance(self.cluster_level, int):
            dendrogram = community_louvain.generate_dendrogram(nx_graph)
            cluster_level = self.cluster_level if self.cluster_level < len(dendrogram) else len(dendrogram) - 1
            if cluster_level != self.cluster_level:
                self.logger.warning(f'specified level {self.cluster_level} exceed the max value {len(dendrogram) - 1}, use the last level.')
            partition = community_louvain.partition_at_level(dendrogram, cluster_level)
        else:
            partition = community_louvain.best_partition(nx_graph)
        modularity = community_louvain.modularity(partition, nx_graph)
        clusters = dict()
        del nx_graph
        for k, v in partition.items():
            if v in clusters:
                clusters[v].add(k)
            else:
                clusters[v] = set()
                clusters[v].add(k)
        clusters = {key: clusters[key] for key in sorted(clusters)}
        self.logger.info(f'created {len(clusters)} clusters.')
        self.logger.info(f'each cluster contains {[len(c) for c in clusters.values()]}.')
        self.logger.info(f'clustering modularity = {modularity}.')
        return clusters


class NaiveSampler(object):
    
    def __init__(
        self, 
        questions:QuestionBase,
        logger: Optional[logging.Logger] = None
        ) -> None:
        self.logger = logger if logger else set_logger('sampler-stdout', level=logging.DEBUG)
        self.pool   = set(questions.sparql.get_all_head_entity())
        
    def sampling(self, k: int) -> Dict[int, List[str]]:
        if len(self.pool) == 0:
            self.logger.debug('empty start entity pool, no start entity is sampled')
            return dict()
        elif len(self.pool) > k:
            e = random.sample(self.pool, k)
            self.pool.difference_update(e)
            self.logger.debug(f'sampled {len(e)} entities, {len(self.pool)} entities remain.')
            return {0: e}
        elif len(self.pool) <= k:
            e = list(self.pool)
            self.pool = set()
            self.logger.debug(f'sampled the last {len(e)} entities, no entity remains.')
            return {0: e}
    
    
class RandomSampler(object):
    
    def __init__(
        self, 
        questions:QuestionBase,
        logger: Optional[logging.Logger] = None
        ) -> None:
        self.logger = logger if logger else set_logger('sampler-stdout', level=logging.DEBUG)
        self.cluster_pool = questions.question_topic
        
    def sampling(self, k: int) -> Dict[int, List[str]]:
        start_entities = dict()
        for c, e in self.cluster_pool.items():
            if len(e) == 0:
                self.logger.debug(f'empty start entity pool of cluster {c}, no start entity is sampled')
                continue
            elif len(e) > k:
                start_entities[c] = random.sample(e, k)
                e.difference_update(start_entities[c])
                self.logger.debug(f'sampled {len(start_entities[c])} entities, {len(e)} entities remain in cluster {c}.')
            elif len(e) <= k:
                start_entities[c] = list(e)
                self.cluster_pool[c] = set()
                self.logger.debug(f'sampled {start_entities[c]} entities of cluster {c}, no entity remains.')
        return start_entities
    

class IntelligentSampler(object):
    # TODO: support different sampling mode
    def __init__(
        self,
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3],
        questions:QuestionBase,
        mode: str = 'hard',
        entity_mapping: Optional[Callable[[str], str]] = None,
        logger: Optional[logging.Logger] = None
        ) -> None:
        self.llm = llm
        self.mode = mode
        self.cluster_pool = questions.question_topic
        self.entity_mapping = entity_mapping if entity_mapping else lambda x: x
        self.logger = logger if logger else set_logger('sampler-stdout', level=logging.DEBUG)
        
    def sampling(
        self, 
        k: int, 
        history: Optional[Dict[int, List[Dict[str, Union[str,bool]]]]] = None
        ) -> Dict[int, List[str]]:
        if history is None:
            self.logger.debug('no history was provided, conduct random sampling')
            start_entities = self.__random_sampling(k)
            for k, v in start_entities.items():
                self.cluster_pool[k].difference_update(v)
        else:
            start_entities = dict()
            candidate_start_entities = self.__random_sampling(k * 10)
            for c, e in candidate_start_entities.items():
                if len(e) == 0:
                    self.logger.debug(f'no start entity of cluster {c}, continue.')
                    continue
                elif len(e) <= k:
                    start_entities[c] = e
                    self.logger.debug(f'no enough start entities, got {len(e)}/{k}, select all.')
                    continue
                else:
                    entity_names  = [self.entity_mapping(e_) for e_ in e]
                    name_entities = {name: e_ for name, e_ in zip(entity_names, e)} 
                    prompt = start_entity_prune_prompt(history=history[c], candidate_name=entity_names, k=k)
                    self.debug(f'start entity pruning prompt: {prompt}')
                    result = self.llm.chat(prompt)
                    self.debug(f'llm point pruning feedback: {result}')
                    start_entity = self.__extract_start_entities(result)
                    self.logger.debug(f'extracted {len(start_entity)} start entities')
                    if start_entity:
                        start_entity = [s for s in start_entity if s in entity_names]
                        start_entity = [name_entities[name] for name in start_entity]
                        start_entity = start_entity[:k] if len(start_entity) > k else start_entity
                        self.logger.debug(f'found {len(start_entity)} valid start entities')
                        self.cluster_pool[c].difference_update(start_entity)
                        start_entities[c] = start_entity
            return start_entities
                
    def __extract_start_entities(self, string: str) -> List[str]:
        match_bracket = re.search(r'\[(.*?)\]', string)
        if match_bracket:
            bracket_content = match_bracket.group(1) 
            matches = re.findall(r"""['"](.*?)['"]""", bracket_content)
            if len(matches) == 0:
                self.logger.warning(f'cannot match any start entity in {string}')
                return None
            return matches
        else:
            self.logger.warning(f'cannot match bracket in {string}')
            return None
        
    def __random_sampling(self, k: int) -> Dict[int, List[str]]:
        start_entities = dict()
        for c, e in self.cluster_pool.items():
            if len(e) == 0:
                continue
            elif len(e) > k:
                start_entities[c] = random.sample(e, k)
            elif len(e) <= k:
                start_entities[c] = list(e)
        return start_entities