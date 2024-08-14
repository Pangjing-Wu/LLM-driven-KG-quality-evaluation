import argparse
import datetime
import json
import os
import random
import sys
import traceback

import torch
from tqdm import tqdm

sys.path.append('.')
from utils.loader import llm_loader
from utils.logging import set_logger
sys.path.pop()


random.seed(0)
torch.manual_seed(0)


def save_2_jsonl(question, answer, filepath):
    dict = {"question": question, "results": answer}
    with open(filepath, "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def main(args, question_path, answer_path, logger):
    with open(question_path,encoding='utf-8') as f:
        dataset = json.load(f)
    llm = llm_loader(
        name=args.llm, 
        quantization=args.quantization, 
        cuda=args.cuda, 
        temperature=args.temperature, 
        max_new_tokens=args.max_length
        )
    for data in tqdm(dataset):
        question = data[args.question_string]
        logger.info(f'{question = }')
        if args.multiple_ans:
            prompts = [
                dict(content=f"You are an AI assistant that helps answer users' questions", role='system'),
                dict(content=f"""Please directly answer the question and mark the answer with curly braces. Please use "," to split your answers if there is multiple answers:\n{question}""", role='user'),
                ]
        else:
            prompts = [
                dict(content=f"You are an AI assistant that helps answer users' questions", role='system'),
                dict(content=f"""Please directly answer the question and mark the answer with curly braces:\n{question}""", role='user'),
                ]
        logger.info(f'{prompts = }')
        ans = llm.chat(prompts)
        logger.info(f'{ans = }')
        save_2_jsonl(question, ans, answer_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the arguments for the dataset and model settings.')
    parser.add_argument('--dataset', type=str, default='cwq', help='Dataset to use')
    parser.add_argument('--question_string', type=str, default='question', help='key of questions')
    parser.add_argument('--multiple_ans', '-m', action='store_true', help='support multiple answers')
    parser.add_argument('--llm', default='Llama-2-7b-chat-hf', type=str, help='LLM to use')
    parser.add_argument('--quantization', '-q', action='store_true', help='use 4-bit quantization')
    parser.add_argument('--max_length', type=int, default=32, help='Maximum length of the query')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for reasoning')
    parser.add_argument('--cuda', type=int, default=0, help='Cuda device id')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    date = datetime.datetime.now().strftime('%y%m%d-%H%M')
    question_path = f'./data/{args.dataset}.json'
    answer_path   = f'./results/LLM-QA/{args.dataset}-{args.llm}-{date}.jsonl'
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)
    
    level  = 'DEBUG' if args.debug else 'INFO'
    logger = set_logger(f'LLM-QA-{args.llm}-{args.dataset}', logfile=answer_path.replace('.jsonl', '.log'), level=level)
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    try:
        main(args, question_path=question_path, answer_path=answer_path, logger=logger)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e
    