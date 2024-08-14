import os
import time
from typing import Dict, List, Optional, Union

import requests
import torch

from .base import ChatLLMBase


class AzureChatGPT(ChatLLMBase):
    base = 'https://comp.azure-api.net/azure'
    
    def __init__(self,
        model: str,
        key: Optional[str] = None,
        **chat_kwargs
        ) -> None:
        self.model = model
        self.key   = key if key else os.environ['AZURE_GPT_KEY']
        self.chat_kwargs = dict()
        self.chat_kwargs.update(chat_kwargs)
        
    def chat(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str | List[str]]:
        body     = dict(messages=prompt, **self.chat_kwargs)
        response = requests.post(f'{self.base}/openai/deployments/{self.model}/chat/completions', json=body, headers={'api-key': self.key})
        outputs  = response.json()
        for _ in range(10):
            try:
                return outputs["choices"][0]["message"]["content"]
            except KeyError:
                if outputs['statusCode'] == 500:
                    time.sleep(6)
                    continue
                else:
                    raise RuntimeError(f'capture unexpected response while invoking the API, please check the response below:\n{outputs}')
        raise RuntimeError(f'capture unexpected response while invoking the API, please check the response below:\n{outputs}')
        
    def embedding(self, text: str) -> torch.TensorType:
        body     = dict(input=text)
        response = requests.post(f'{self.base}/openai/deployments/ada002/embeddings', json=body, headers={'api-key': self.key})
        outputs  = response.json()
        for _ in range(10):
            try:
                return outputs["data"][0]["embedding"]
            except KeyError:
                if outputs['statusCode'] == 500:
                    time.sleep(6)
                    continue
                else:
                    raise RuntimeError(f'capture unexpected response while invoking the API, please check the response below:\n{outputs}')
        raise RuntimeError(f'capture unexpected response while invoking the API, please check the response below:\n{outputs}')
    
    
class OpenAIChatGPT(ChatLLMBase):
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError