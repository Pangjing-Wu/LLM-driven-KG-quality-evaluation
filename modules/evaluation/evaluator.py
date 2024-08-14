import logging
from scipy.stats import ttest_ind
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from utils.logging import set_logger
from utils.string import string_match


class Evaluator(object):
    
    def __init__(self, p: float = 0.05, logger: Optional[logging.Logger] = None):
        self.p = p
        self.score = dict(side0 = list(), side1=list())
        self.logger = logger if logger else set_logger('evaluator-stdout', level=logging.DEBUG)
    
    def __call__(self, results: List[Dict[str, str]], answer_side: int) -> Tuple[Dict[str, Union[str, bool]], float]:
        """
            results = [{question: , ref_ans: , ans: }]
        """
        correct = 0
        for result in results:
            ref_ans, ans = result['ref_ans'], result['ans']
            if ans:
                flag = string_match(ref_ans, ans)
                correct += 1 if flag else 0
                result['correct'] = flag
            else:
                result['correct'] = False
        score = round(correct/len(results)*100, 2) if results else 0
        self.score[f'side{answer_side}'].append(score)
        return results, score

    def t_test(self) -> bool:
        if len(self.score['side0']) != len(self.score['side1']):
            self.logger.error(f'imbalanced score record {self.score}')
            return False
        t_stat, p_value = ttest_ind(self.score['side0'], self.score['side1'])
        self.logger.info(f'{t_stat = :.5f}, {p_value = :.5f}')
        return (t_stat, p_value, True if p_value < self.p else False)
    
    @property
    def winner(self) -> str:
        if len(self.score['side0']) != len(self.score['side1']):
            self.logger.error(f'imbalanced score record {self.score}')
            return 'error'
        if np.mean(self.score['side0']) < np.mean(self.score['side1']):
            return '1'
        elif np.mean(self.score['side0']) == np.mean(self.score['side1']):
            return 'equal'
        else:
            return '0'