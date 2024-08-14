from typing import Union

from modules.utils.llm import AzureChatGPT, GLM4, Llama2, Llama3


def llm_loader(
    name: str, 
    quantization: bool, 
    cuda: Union[int, str], 
    temperature: float, 
    max_new_tokens: int
    ) -> Union[AzureChatGPT, GLM4, Llama2, Llama3]:
    
    cuda = f'cuda:{cuda}' if isinstance(cuda, int) else cuda
    if 'llama-2' in name.lower():
        llm = Llama2(
            name=f'meta-llama/{name}',
            quantization=quantization,
            device=cuda, 
            do_sample=True if temperature > 0 else False, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            )
    elif 'llama-3' in name.lower():
        llm = Llama3(
            name=f'meta-llama/{name}',
            quantization=quantization,
            device=cuda, 
            do_sample=True if temperature > 0 else False, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            )
    elif 'gpt' in name.lower():
        llm = AzureChatGPT(
            name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            )
    elif 'glm' in name.lower():
        llm = GLM4(
            device=cuda, 
            do_sample=True if temperature > 0 else False, 
            max_new_tokens=max_new_tokens,
            temperature=temperature
            )
    else:
        raise ValueError(f'unknown LLM name {name}.')
    return llm