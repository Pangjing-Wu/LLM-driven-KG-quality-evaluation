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
import config.kg
from modules.answer import RelationSearch, EntityMatcher, EntitySearch, QuestionEntityRecognition, Reasoning, Answering
from modules.utils.kg import RDFLibSPARQL
from utils.loader import llm_loader
from utils.logging import set_logger
sys.path.pop()


random.seed(0)
torch.manual_seed(0)


def save_2_jsonl(question, answer, triple_chains, filepath):
    dict = {"question": question, "results": answer, "reasoning_chains": triple_chains}
    with open(filepath, "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def main(args, kg_config, question_path, answer_path, logger):
    with open(question_path,encoding='utf-8') as f:
        dataset = json.load(f)
    
    llm = llm_loader(
        name=args.llm, 
        quantization=args.quantization, 
        cuda=args.cuda, 
        temperature=args.temperature, 
        max_new_tokens=args.max_length
        )
    sparql = RDFLibSPARQL(path=kg_config['path'], domain=kg_config['domain'])
    
    relation_searcher = RelationSearch(
        llm=llm, 
        sparql=sparql, 
        width=args.width, 
        max_candidate_relations=args.max_relation_size,
        logger=logger,
        **kg_config
        )
    entity_searcher = EntitySearch(
        llm=llm, 
        sparql=sparql,
        width=args.num_retain_entity,
        max_candidate_entities=args.max_entity_size,
        logger=logger,
        **kg_config
    )
    entity_recognizer = QuestionEntityRecognition(llm=llm, width=1, logger=logger)
    entity_matcher    = EntityMatcher(llm=llm, sparql=sparql, width=1, logger=logger, **kg_config)
    
    reasoner = Reasoning(llm=llm, logger=logger)
    answer   = Answering(llm=llm, logger=logger)
    
    
    for data in tqdm(dataset):
        question = data[args.question_string]
        logger.info(f'{question = }')
        try:
            topic_entity = data['topic_entity']
            topic_entity = list(topic_entity.values()) if isinstance(topic_entity, dict) else topic_entity
        except KeyError:
            topic_entity = entity_recognizer(question)
            logger.debug(f'no specified topic entity, using question entity recognition and find entity: {topic_entity}')
        topic_entity = entity_matcher(topic_entity)
        logger.debug(f'matched entity: {topic_entity}')
        triple_chains = []
        pre_relations = []
        current_relations = [] 
        answer_flag = False
        try:
            for depth in range(args.depth):
                # search relations.
                logger.debug(f'{topic_entity = }')
                for entity in topic_entity:
                    relations = relation_searcher(entity=entity, pre_relations=pre_relations, question=question)
                    current_relations.extend(relations)
                # search entities and format triples.
                triples, entities = entity_searcher(question=question, relations=current_relations)
                triple_chains.append(triples)
                # make answer by reasoner.
                ans = reasoner(question=question, triple_chains=triple_chains)
                # if got answer at current depth.
                if ans:
                    logger.info(f'find answer at depth {depth+1}, stop search.')
                    save_2_jsonl(question, ans, triple_chains, answer_path)
                    answer_flag = True
                    break
                # if cannot find the answer at current depth.
                else:
                    logger.info(f'depth {depth+1} still does not find the answer.')
                    topic_entity = entities
                    pre_relations.extend(current_relations)
                    if len(topic_entity) == 0:
                        logger.info(f'obtain empty entity at depth {depth+1}, stop search.')
                        break
            # if cannot find the answer throughout all depth.
            if not answer_flag:
                logger.info(f'all depth cannot find the answer, try to make answer with existing knowledge.')
                ans = answer(question=question, triple_chains=triple_chains)
                save_2_jsonl(question, ans, triple_chains, answer_path)
        except Exception as e:
            logger.error(f'catch exception {e}, record blank answers.')
            save_2_jsonl(question, None, None, answer_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the arguments for the dataset and model settings.')
    parser.add_argument('--kg', type=str, required=True, help='name of knowledge graph')
    parser.add_argument('--dataset', type=str, default='cwq', help='Dataset to use')
    parser.add_argument('--question_string', type=str, default='question', help='key of questions')
    parser.add_argument('--llm', default='Meta-Llama-3-8B-Instruct', type=str, help='LLM to use')
    parser.add_argument('--quantization', '-q', action='store_true', help='use 4-bit quantization')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum length of the query')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for reasoning')
    parser.add_argument('--width', type=int, default=3, help='Width parameter for the search')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the reasoning')
    parser.add_argument('--num_retain_entity', type=int, default=5, help='Number of entities to retain')
    parser.add_argument('--max_relation_size', type=int, default=200, help='Maximum size of candidate relations')
    parser.add_argument('--max_entity_size', type=int, default=5, help='Maximum size of candidate entities')
    parser.add_argument('--cuda', type=int, default=0, help='Cuda device id')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    date = datetime.datetime.now().strftime('%y%m%d-%H%M')
    question_path = f'./data/{args.dataset}.json'
    answer_path   = f'./results/ToG/{args.kg}-{args.dataset}-{date}.jsonl'
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)
    
    level  = 'DEBUG' if args.debug else 'INFO'
    logger = set_logger('ToG-enhanced', logfile=answer_path.replace('.jsonl', '.log'), level=level)
    
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    kg_config = getattr(config.kg, f'{args.kg}_configs')
    try:
        main(args, kg_config, question_path=question_path, answer_path=answer_path, logger=logger)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e
    