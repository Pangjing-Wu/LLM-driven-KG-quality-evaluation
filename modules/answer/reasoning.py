import re
import logging
from typing import List, Optional, Union

from ..utils.llm.chatgpt import AzureChatGPT
from ..utils.llm.glm import GLM4
from ..utils.llm.llama import Llama2, Llama3
from .prompt import answer_prompt, reasoning_prompt
from utils.logging import set_logger
from utils.string import extract_string_from_dict

class Answering(object):
    
    def __init__(
        self, 
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3], 
        logger: Optional[logging.Logger] = None
        ) -> None:
        self.llm = llm
        self.logger = logger if logger else set_logger('answering-stdout', level=logging.DEBUG)
    
    def _extract_answer(self, string: str) -> Union[str, None]:
        ans = extract_string_from_dict(string, 'answer')
        if not ans:
            self.logger.warning(f'cannot match any answers in {string}')
            return None
        return ans if ans != 'None' else None
        
    def __call__(self, question: str, triple_chains: List[List[List[str]]]) -> str:
        prompt = answer_prompt(question, triple_chains)
        self.logger.debug(f'answering prompt: {prompt}')
        result = self.llm.chat(prompt)
        self.logger.debug(f'llm answering feedback: {result}')
        answer = self._extract_answer(result)
        return answer if answer else None


class Reasoning(Answering):
    
    def __init__(
        self, 
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3],
        logger: Optional[logging.Logger] = None
        ) -> None:
        self.llm = llm
        self.logger = logger if logger else set_logger('reasoning-stdout', level=logging.DEBUG)
    
    def __extract_reasoning_flag(self, string: str) -> bool:
        flag = extract_string_from_dict(string, 'sufficient')
        if flag and flag.lower() in ['yes', 'true']:
            return True
        else:
            return False
        
    def __call__(self, question: str, triple_chains: List[List[List[str]]]) -> str:
        prompt = reasoning_prompt(question, triple_chains)
        self.logger.debug(f'reasoning prompt: {prompt}')
        result = self.llm.chat(prompt)
        self.logger.debug(f'llm reasoning feedback: {result}')
        flag = self.__extract_reasoning_flag(result)
        if flag:
            answer = self._extract_answer(result)
            return answer if answer else None
        else:
            return None