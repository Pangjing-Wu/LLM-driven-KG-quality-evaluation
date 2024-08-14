import logging
from typing import Callable, Dict, List, Optional, Union

from .entity import EntitySearch, EntityMatcher
from .reasoning import Reasoning
from .relation import RelationSearch
from ..utils.kg.query import RDFLibSPARQL
from ..utils.llm.chatgpt import AzureChatGPT
from ..utils.llm.glm import GLM4
from ..utils.llm.llama import Llama2, Llama3


class Answer(object):
    
    def __init__(
        self, 
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3], 
        sparql: RDFLibSPARQL,
        width: int,
        depth: int,
        max_candidate_relations: int,
        max_candidate_entities: int,
        domain: Optional[str] = None,
        entity_mapping: Optional[Callable[[str], str]] = None,
        entity_regex: Optional[str] = None,
        abandon_relations: Optional[List[str]] = None,
        abandon_entity_names: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
        ) -> None:
        # set attributes.
        self.llm = llm
        self.width = width
        self.depth = depth
        self.logger = logger
        # load answer modules.
        self.relation_searcher = RelationSearch(
            llm=llm, 
            sparql=sparql, 
            width=width, 
            max_candidate_relations=max_candidate_relations, 
            domain=domain,
            entity_mapping=entity_mapping,
            abandon_relations=abandon_relations,
            logger=logger
            )
        self.entity_searcher = EntitySearch(
            llm=llm, 
            sparql=sparql,
            width=width,
            max_candidate_entities=max_candidate_entities,
            domain=domain,
            entity_regex=entity_regex,
            entity_mapping=entity_mapping,
            abandon_entity_names=abandon_entity_names,
            logger=logger
            )
        self.entity_matcher = EntityMatcher(
            llm=llm,
            sparql=sparql,
            width=width,
            domain=domain,
            entity_regex=entity_regex,
            entity_mapping=entity_mapping,
            abandon_entity_names=abandon_entity_names,
            logger=logger
        )
        self.reasoner = Reasoning(llm=llm, logger=logger)
        
    def __call__(self, question: Dict[str, str]) -> Union[str, None]:
        question, start_entities = question['question'], question['start_entity']
        self.logger.info(f'{question = }\n{start_entities = }')
        triple_chains     = list()
        pre_relations     = list()
        current_relations = list()
        start_entities    = self.entity_matcher(start_entities)
        for depth in range(self.depth):
            for entity in start_entities:
                relations = self.relation_searcher(entity=entity, pre_relations=pre_relations, question=question)
                current_relations.extend(relations)
            # search entities and format triples.
            triples, entities = self.entity_searcher(question=question, relations=current_relations)
            triple_chains.append(triples)
            # make answer by reasoner.
            ans = self.reasoner(question=question, triple_chains=triple_chains)
            # if got answer at current depth.
            if ans:
                self.logger.info(f'find answer at depth {depth+1}, stop search.')
                return ans
            # if cannot find the answer at current depth.
            else:
                self.logger.info(f'depth {depth+1} still does not find the answer.')
                start_entities = entities
                pre_relations.extend(current_relations)
                # if no any unvisited entities.
                if len(start_entities) == 0:
                    self.logger.info(f'obtain empty entity at depth {depth+1}, stop search.')
                    break
        # if cannot find the answer throughout all depth.
        return None