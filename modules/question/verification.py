import logging
from typing import Dict, List, Optional, Union

from ..utils.llm.chatgpt import AzureChatGPT
from ..utils.llm.glm import GLM4
from ..utils.llm.llama import Llama2, Llama3
from .prompt import question_direct_answer_check_prompt, question_answerable_check_prompt
from utils.logging import set_logger
from utils.string import extract_string_from_dict, string_match

class QuestionComplianceChecker(object):
    
    def __init__(
        self,
        llm: Union[AzureChatGPT, GLM4, Llama2, Llama3], 
        logger: Optional[logging.Logger] = None
        ) -> None:
        self.llm = llm
        self.logger = logger if logger else set_logger('question-verification-stdout', level=logging.INFO)
    
    def __call__(self, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        passed_questions = list()
        for question in questions:
            directly_check = self.__directly_answer_check(question=question['question'], reference_answer=question['ref_ans'])
            if not directly_check:
                continue
            answerable_check = self.__answerable_check(question=question['question'], reasoning_path=question['chain'], reference_answer=question['ref_ans'])
            if not answerable_check:
                continue
            passed_questions.append(question)
        if questions:
            self.logger.info(f'input {len(questions)} questions, output {len(passed_questions)} questions.')
            self.logger.info(f'question passed rate: {len(passed_questions) / len(questions) * 100 :.2f}%.')
        else:
            self.logger.warning('empty input questions.')
        return passed_questions
    
    def __directly_answer_check(self, question: str, reference_answer: str) -> bool:
        prompt = question_direct_answer_check_prompt(question)
        self.logger.debug(f'question direct answer check prompt: {prompt}')
        result = self.llm.chat(prompt)
        self.logger.debug(f'question direct answer check feedback: {result}')
        answer = self.__extract_answer(result)
        flag = string_match(reference_answer, answer) if answer else False
        if not flag:
            self.logger.debug(f'question: "{question}" PASSED check. {reference_answer = }, {answer = }.')
        else:
            self.logger.debug(f'question: "{question}" FAILED in direct answer check. {reference_answer = }, {answer = }.')
        return not flag
    
    def __answerable_check(self, question: str, reasoning_path: List[str], reference_answer: str) -> bool:
        prompt = question_answerable_check_prompt(question, reasoning_path)
        self.logger.debug(f'question answerable check prompt: {prompt}')
        result = self.llm.chat(prompt)
        self.logger.debug(f'question answerable check feedback: {result}')
        answer = self.__extract_answer(result)
        flag = string_match(reference_answer, answer) if answer else False
        if flag:
            self.logger.debug(f'question: "{question}" PASSED check. {reference_answer = }, {answer = }.')
        else:
            self.logger.debug(f'question: "{question}" FAILED in answerable check. {reference_answer = }, {answer = }.')
        return flag
    
    def __extract_answer(self, string: str) -> Union[str, None]:
        ans = extract_string_from_dict(string, 'answer')
        return ans if ans != 'None' else None