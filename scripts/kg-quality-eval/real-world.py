import argparse
import datetime
import json
import os
import sys
import traceback


sys.path.append('.')
import config.kg
from modules.answer import Answer
from modules.question import Questioner
from modules.evaluation import Evaluator
from modules.utils.kg import RDFLibSPARQL
from utils.loader import llm_loader
from utils.logging import set_logger
sys.path.pop()


def parse_args():
    parser = argparse.ArgumentParser(description='Process the arguments for the dataset and model settings.')
    parser.add_argument('--kg0', type=str, required=True, help='KG #0 name')
    parser.add_argument('--kg1', type=str, required=True, help='KG #1 name')
    
    parser.add_argument('--p', type=float, default=0.05, help='Significance level for evaluation')
    parser.add_argument('--max_round', type=int, default=10, help='Maximum number of evaluation rounds')
    parser.add_argument('--min_round', type=int, default=5, help='Minimum number of evaluation rounds')
    
    parser.add_argument('--llm', type=str, required=True, help='LLM to use')
    parser.add_argument('--quantization', '-q', action='store_true', help='Use 4-bit quantization')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for reasoning')
    parser.add_argument('--max_length', type=int, default=32, help='Maximum length of the query')
    
    parser.add_argument('--n_start_entities', type=int, default=1, help='Number of start entities for questioning')
    parser.add_argument('--width', type=int, default=3, help='Width of questioning')
    parser.add_argument('--depth', type=int, default=3, help='Depth of questioning')
    parser.add_argument('--sampling_method', type=str, default='random', help='sampling method for start entity')
    parser.add_argument('--sampling_mode', type=str, default='hard', help='sampling mode for ')
    parser.add_argument('--cluster_level', default='best', help='cluster level for sampling, int or `best`')
    
    parser.add_argument('--ans_width', type=int, default=3, help='Width of answering')
    parser.add_argument('--ans_depth', type=int, default=3, help='Depth of answering')
    parser.add_argument('--max_candidate_relations', type=int, default=50, help='Maximum number of candidate relations')
    parser.add_argument('--max_candidate_entities', type=int, default=50, help='Maximum number of candidate entities')
    
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()


def main(args, logger, module_loggers, result_paths):
    kg0_config = getattr(config.kg, f'{args.kg0}_configs')
    kg1_config = getattr(config.kg, f'{args.kg1}_configs')
    
    llm = llm_loader(
        name=args.llm, 
        quantization=args.quantization, 
        cuda=args.cuda, 
        temperature=args.temperature, 
        max_new_tokens=args.max_length
        )
    sparql_0 = RDFLibSPARQL(path=kg0_config['path'], domain=kg0_config['domain'])
    sparql_1 = RDFLibSPARQL(path=kg1_config['path'], domain=kg1_config['domain'])

    questioner_0 = Questioner(
        llm=llm,
        sparql=sparql_0,
        n_start_entities=args.n_start_entities,
        width=args.width,
        depth=args.depth,
        sampling_method=args.sampling_method,
        sampling_mode=args.sampling_mode,
        cluster_level=args.cluster_level,
        logger=module_loggers['questioner_0'],
        **kg0_config
    )
    questioner_1 = Questioner(
        llm=llm,
        sparql=sparql_1,
        n_start_entities=args.n_start_entities,
        width=args.width,
        depth=args.depth,
        sampling_method=args.sampling_method,
        sampling_mode=args.sampling_mode,
        cluster_level=args.cluster_level,
        logger=module_loggers['questioner_1'],
        **kg1_config
    )
    
    answer_0 = Answer(
        llm=llm,
        sparql=sparql_0,
        width=args.ans_width,
        depth=args.ans_depth,
        max_candidate_relations=args.max_candidate_relations,
        max_candidate_entities=args.max_candidate_entities,
        logger=module_loggers['answer_0'],
        **kg0_config
    )
    answer_1 = Answer(
        llm=llm,
        sparql=sparql_1,
        width=args.ans_width,
        depth=args.ans_depth,
        max_candidate_relations=args.max_candidate_relations,
        max_candidate_entities=args.max_candidate_entities,
        logger=module_loggers['answer_1'],
        **kg1_config
    )
    
    evaluator = Evaluator(
        p=args.p,
        logger=module_loggers['evaluator']
    )
    
    significance = False
    for i in range(args.max_round):
        logger.info(f'====== start {i + 1} round evaluation ======')
        # KG 0 ask, KG 1 answer.
        questions = questioner_0()
        if not questions:
            logger.warning(f'no question generated by KG #0 in the {i + 1} round.')
            break
        logger.info(f'KG #0 generates {len(questions)} in the {i + 1} round.')
        for question in questions:
            question['ans'] = answer_1(question)
        questions, acc = evaluator(questions, 1)
        with open(result_paths[1], "a") as f:
            for question in questions:
                json.dump(question, f, indent=4)
        logger.info(f'question answer accuracy of KG #1 = {acc}')
        # KG 1 ask, KG 0 answer.
        questions = questioner_1()
        if not questions:
            logger.warning(f'no question generated by KG #1 in the {i + 1} round.')
            break
        logger.info(f'KG #1 generates {len(questions)} in the {i + 1} round.')
        for question in questions:
            question['ans'] = answer_0(question)
        questions, acc = evaluator(questions, 0)
        with open(result_paths[0], "a") as f:
            for question in questions:
                json.dump(question, f, indent=4)
        logger.info(f'question answer accuracy of KG #0 = {acc}')
        # check significance.
        t_score, p_value, significance = evaluator.t_test()
        logger.info(f'{i + 1} round t-score: {t_score}, p-value: {p_value}, current winner is KG #{evaluator.winner}')
        if i >= args.min_round and significance:
            logger.info(f'achieve stop condition with significance.')
            break
    if not significance:
        logger.info(f'achieve stop condition without significance.')
        
        
if __name__ == "__main__":
    args = parse_args()
    date = datetime.datetime.now().strftime('%y%m%d-%H%M')
    result_dir   = f'./results/KG-quality-eval/{args.kg0}-{args.kg1}/{date}'
    result_paths = [os.path.join(result_dir, 'kg-0-ans.json'), os.path.join(result_dir, 'kg-1-ans.json')]
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    
    level  = 'DEBUG' if args.debug else 'INFO'
    logger = set_logger(f'kg-eval-{date}', logfile=os.path.join(result_dir, 'main.log'), level=level)
    module_loggers = {
        'questioner_0': set_logger(f'kg0-question-module-{date}', logfile=os.path.join(result_dir, 'questioner_0.log'), level=level),
        'questioner_1': set_logger(f'kg1-question-module-{date}', logfile=os.path.join(result_dir, 'questioner_1.log'), level=level),
        'answer_0': set_logger(f'kg0-answer-module-{date}', logfile=os.path.join(result_dir, 'answer_0.log'), level=level),
        'answer_1': set_logger(f'kg1-answer-module-{date}', logfile=os.path.join(result_dir, 'answer_1.log'), level=level),
        'evaluator': set_logger(f'kg-eval-module-{date}', logfile=os.path.join(result_dir, 'evaluator.log'), level=level)
    }
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    try:
        main(args, logger, module_loggers, result_paths)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e