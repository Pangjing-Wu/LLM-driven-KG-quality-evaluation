import logging
from typing import Callable, List, Optional, Union

from .generation import QuestionGenerator
from .sampling import NaiveSampler, RandomSampler, IntelligentSampler, QuestionBase
from .verification import QuestionComplianceChecker
from ..utils.kg.query import RDFLibSPARQL
from ..utils.llm.chatgpt import AzureChatGPT
from ..utils.llm.glm import GLM4
from ..utils.llm.llama import Llama2, Llama3


class Questioner(object):
    
    def __init__(
        self, 
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3], 
        sparql: RDFLibSPARQL,
        n_start_entities: int,
        depth: int,
        width: int,
        sampling_method: str,
        sampling_mode: Optional[str] = None,
        cluster_level: Optional[int] = None,
        quest_relation: bool = False,
        domain: Optional[str] = None,
        entity_mapping: Optional[Callable[[str], str]] = None,
        entity_regex: Optional[str] = None,
        abandon_relations: Optional[List[str]] = None,
        abandon_entity_names: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
        ) -> None:
        self.n_start_entities = n_start_entities
        self.question_base = QuestionBase(
            sparql=sparql,
            cluster_level=cluster_level,
            entity_mapping=entity_mapping,
            entity_regex=entity_regex,
            abandon_relations=abandon_relations,
            abandon_entity_names=abandon_entity_names,
            logger=logger
        )
        if sampling_method.lower() == 'naive':
            self.sampler = NaiveSampler(self.question_base, logger=logger)
        elif sampling_method.lower() == 'random':
            self.sampler = RandomSampler(self.question_base, logger=logger)
        elif sampling_method.lower() == 'intelligent':
            self.sampler = IntelligentSampler(
                llm=llm, 
                questions=self.question_base, 
                mode=sampling_mode, 
                entity_mapping=entity_mapping, 
                logger=logger
                )
        else:
            raise ValueError(f'Unknown sampling method {sampling_method}.')
        self.generator = QuestionGenerator(
            llm=llm,
            sparql=sparql,
            depth=depth,
            width=width,
            quest_relation=quest_relation,
            domain=domain,
            entity_mapping=entity_mapping,
            logger=logger
        )
        self.checker = QuestionComplianceChecker(llm=llm, logger=logger)
        
    def __call__(self):
        start_entities = self.sampler.sampling(k=self.n_start_entities)
        questions = self.generator(start_entities=start_entities)
        questions = self.checker(questions=questions)
        return questions