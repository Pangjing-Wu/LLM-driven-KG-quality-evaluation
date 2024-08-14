import logging
import random
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

from ..utils.kg.query import RDFLibSPARQL
from ..utils.llm.chatgpt import AzureChatGPT
from ..utils.llm.glm import GLM4
from ..utils.llm.llama import Llama2, Llama3
from .prompt import question_for_tail_prompt, question_for_relation_prompt
from utils.logging import set_logger


class QuestionGenerator(object):
    
    def __init__(
        self, 
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3], 
        sparql: RDFLibSPARQL,
        depth: int,
        width: int,
        quest_relation: bool = False,
        domain: Optional[str] = None,
        entity_mapping: Optional[Callable[[str], str]] = None,
        logger: Optional[logging.Logger] = None,
        ) -> None:
        self.llm = llm
        self.sparql = sparql
        self.depth = depth
        self.width = width
        self.quest_relation = quest_relation
        self.domain = domain
        self.entity_mapping = entity_mapping if entity_mapping else lambda x: x
        self.logger = logger if logger else set_logger('question-generator-stdout', level=logging.DEBUG)
    
    def __call__(self, start_entities: Dict[int, List[str]]) -> List[Dict[str, Union[str,int]]]:
        questions = list()
        for c in start_entities:
            for start_entity in start_entities[c]:
                triple_paths = self.__obtain_neighboring_paths(start_entity)
                for triple_path in triple_paths:
                    # generate question for tail.
                    self.logger.info(f'knowledge triple chain for concept `{c}`: {triple_paths}')
                    question, ref_ans = self.__generate_question_for_tail(triple_path)
                    self.logger.info(f'generated question for entity `{ref_ans}`: {question}')
                    if question:
                        questions.append(dict(question=question, chain=triple_path, start_entity=[self.entity_mapping(start_entity)], ref_ans=ref_ans, cluster=c))
                    # generate question for relation.
                    if self.quest_relation:
                        question, ref_ans = self.__generate_question_for_relation(triple_path)
                        self.logger.info(f'generated question for relation `{ref_ans}`: {question}')
                        if question:
                            questions.append(dict(question=question, chain=triple_path, start_entity=[self.entity_mapping(start_entity)], ref_ans=ref_ans, cluster=c))
        return questions
        
    def __obtain_neighboring_paths(self, entity: str) -> List[List[str]]:
        paths = list()
        head_entity = entity
        head_entity = re.sub(f'^{self.domain}', '', head_entity) if self.domain else head_entity
        relations = self.sparql.get_relation_by_head(head_entity)
        if not relations:
            return paths
        # only perform the widely sampling on the first iteration.
        tail_entities = list()
        k = min(len(relations), self.width)
        
        for relation in random.sample(relations, k=k)[:k]:
            relation = re.sub(f'^{self.domain}', '', relation) if self.domain else relation
            tail_entity = self.sparql.get_tail_entity(head_entity, relation)
            if not tail_entity:
                continue
            else:
                tail_entity = random.choice(tail_entity)
                paths.append([self.entity_mapping(head_entity), relation, self.entity_mapping(tail_entity)])
                tail_entities.append(tail_entity)
        head_entities = tail_entities
        # continuously sample a single path based on the first iteration.
        for i, head_entity in enumerate(head_entities):
            path_cache = [paths[i]]
            for _ in range(1, self.depth):
                head_entity = re.sub(f'^{self.domain}', '', head_entity) if self.domain else head_entity
                relations = self.sparql.get_relation_by_head(head_entity)
                if not relations:
                    break
                relation = random.choice(relations)
                relation = re.sub(f'^{self.domain}', '', relation) if self.domain else relation
                tail_entity = self.sparql.get_tail_entity(head_entity, relation)
                if not tail_entity:
                    break
                tail_entity = random.choice(tail_entity)
                path_cache.append(path_cache[-1] + [relation, self.entity_mapping(tail_entity)])
                head_entity = tail_entity
            path_cache.pop(0)
            paths.extend(path_cache)
        return paths
    
    def __generate_question_for_tail(self, reasoning_path: List[str]) -> Tuple[str, str]:
        ref_ans = reasoning_path[-1]
        prompt  = question_for_tail_prompt(reasoning_path)
        self.logger.debug(f'question for tail entity prompt: {prompt}')
        result = self.llm.chat(prompt)
        self.logger.debug(f'llm question for tail entity feedback: {result}')
        question = self.__extract_question(result)
        return (question, ref_ans)
    
    def __generate_question_for_relation(self, reasoning_path: List[str]) -> Tuple[str, str]:
        ref_ans = reasoning_path[-2]
        prompt  = question_for_relation_prompt(reasoning_path)
        self.logger.debug(f'question for last relation prompt: {prompt}')
        result = self.llm.chat(prompt)
        self.logger.debug(f'llm question for tail entity feedback: {result}')
        question = self.__extract_question(result)
        return (question, ref_ans)
    
    def __extract_question(self, string: str) -> Union[str, None]:
        match = re.search(r'{question: (.+?)}', string)
        return match.group(1) if match else None