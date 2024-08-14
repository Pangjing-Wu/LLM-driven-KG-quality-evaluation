from typing import Optional

from transformers import AutoTokenizer

from .huggingface import HuggingFaceChat, HuggingFaceModel


class GLM4(HuggingFaceChat):
    """
    NOTE: we retrain the arguments of `name` and `quantization` for consistence.
    The model name is fixed and GLM-4 does not support quantization for `transformer == 4.38.2`
    """
    def __init__(
        self, 
        name: str = 'THUDM/glm-4-9b-chat', 
        device: str = 'auto', 
        quantization: Optional[bool] = False,
        **chat_kwargs
        ) -> None:
        name = 'THUDM/glm-4-9b-chat'
        quantization  = False
        self.pipeline = HuggingFaceModel(name, device=device, quantization=quantization).pipeline
        setattr(self.pipeline, 'tokenizer', AutoTokenizer.from_pretrained(name, trust_remote_code=True))
        self.terminators = [self.pipeline.tokenizer.eos_token_id]
        self.chat_kwargs = chat_kwargs