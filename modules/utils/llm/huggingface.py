from typing import Dict, List, Union

import torch
import transformers
from transformers import BitsAndBytesConfig

from .base import ChatLLMBase, NamedSingletonType


class HuggingFaceModel(metaclass=NamedSingletonType):
    """HuggingFace chat pipelines are created in the way of singleton in order to save
    the resource. Models with the same name are associated to the same instance.

    Args:
        object (_type_): _description_
    """

    def __init__(self, name: str, device: str = 'auto', quantization: bool = True) -> None:
        model_kwargs = dict(torch_dtype=torch.bfloat16)
        if quantization:
            model_kwargs['quantization_config'] = self.quantization_config
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=name,
            model_kwargs=model_kwargs,
            device_map=device,
            trust_remote_code=True
        )
        
    @property
    def quantization_config(self):
        return BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    
class HuggingFaceChat(ChatLLMBase):
    """base class of HuggingFace chat model

    Args:
        **model_args (dict): generation model arguments, refer to https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation
    """
    
    def __init__(
        self, 
        name: str, 
        device: str = 'auto', 
        quantization: bool = True,
        **chat_kwargs
        ) -> None:
        self.device      = device
        self.pipeline    = HuggingFaceModel(name, device=device, quantization=quantization).pipeline
        self.terminators = [self.pipeline.tokenizer.eos_token_id]
        self.chat_kwargs = chat_kwargs
        
    def chat(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str | List[str]]:
        prompt = self.pipeline.tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            eos_token_id=self.terminators,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            return_full_text=False,
            **self.chat_kwargs
        )
        return outputs[0]["generated_text"]
    
    def embedding(self, text: str) -> torch.TensorType:
        """
        Returns:
            :embed type: torch.TensorType
            :embed shape: [batch, hidden]
        """
        tokens = self.pipeline.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.pipeline.model(**tokens, output_hidden_states=True)
            embed   = outputs.hidden_states[-1]
            embed   = torch.mean(embed, dim=1).cpu()
            return embed
