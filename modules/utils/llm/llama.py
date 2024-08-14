from typing import Optional

from .huggingface import HuggingFaceChat


class Llama2(HuggingFaceChat):
    
    def __init__(
        self, 
        name: str, 
        device: str = 'auto', 
        quantization: Optional[bool] = False,
        **chat_kwargs
        ) -> None:
        super().__init__(
            name=name, 
            device=device, 
            quantization=quantization,
            **chat_kwargs
        )


class Llama3(HuggingFaceChat):
    
    def __init__(
        self, 
        name: str, 
        device: str = 'auto', 
        quantization: Optional[bool] = False,
        **chat_kwargs
        ) -> None:
        super().__init__(
            name=name, 
            device=device, 
            quantization=quantization,
            **chat_kwargs
        )
        self.terminators.append(self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"))