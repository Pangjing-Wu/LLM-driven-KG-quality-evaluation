import abc
from typing import Dict, List, Optional, Union


class ChatLLMBase(abc.ABC):
    
    @abc.abstractmethod
    def chat(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        raise NotImplementedError

    def format_prompt(
        self,
        prompt: str, 
        system_prompt: Optional[str] = None, 
        instruction: Optional[List[Dict[str, str]]] = None
        ) -> List[Dict[str, str]]:
        message = list()
        if system_prompt is not None:
            assert isinstance(system_prompt, str), TypeError
            message.append(dict(role='system', content=system_prompt))
        if instruction is not None:
            assert isinstance(instruction, list) and isinstance(instruction[0], dict)
            for inst in instruction:
                message.append(dict(role=inst['role'], content=inst['content']))
        message.append(dict(role='user', content=prompt))
        return message
    
    
class NamedSingletonType(type):
    _instance = dict()
    
    def __call__(cls,  name: str, *args, **kwargs):
        if name not in cls._instance.keys():
            cls._instance[name] = super().__call__(name, *args, **kwargs)
        return cls._instance[name]