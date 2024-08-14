import argparse
import datetime
import json
import os
import pickle
import sys
import traceback

import numpy as np
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader

sys.path.append('.')
from utils.logging import set_logger
sys.path.pop()


def parse_args():
    parser = argparse.ArgumentParser(description='Process the arguments for the dataset and model settings.')
    parser.add_argument('--kg0', type=str, required=True, help='KG #0 name')
    parser.add_argument('--kg1', type=str, required=True, help='KG #1 name')

    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number')
    
    parser.add_argument('--p', type=float, default=0.05, help='p-value of t-test')
    parser.add_argument('--min_round', type=int, default=10, help='minimum evaluation round')
    parser.add_argument('--max_round', type=int, default=100, help='maximum evaluation round')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()


def main(args, result_path, logger):
    dataset_0 = pickle.load(os.path.join(args.kg0, 'dataset.pkl'))
    dataset_1 = pickle.load(os.path.join(args.kg1, 'dataset.pkl'))
    model_0   = pickle.load(os.path.join(args.kg0, 'model.pkl')).to(f'cuda:{args.cuda}')
    model_1   = pickle.load(os.path.join(args.kg1, 'model.pkl')).to(f'cuda:{args.cuda}')
    # QA process
    question_loader_0 = DataLoader(dataset_0['dataset'], batch_size=100, shuffle=True)
    question_loader_1 = DataLoader(dataset_1['dataset'], batch_size=100, shuffle=True)
    
    scores_0 = list()
    scores_1 = list()
    significance = False
    # ask KG #0 with the question from KG #1
    for eval_round, (question_batch_0, question_batch_1) in enumerate(zip(question_loader_0, question_loader_1)):
        score_0 = model_0.qa_evaluation(question_batch_1, dataset_0['entity_indices'], k=1)
        logger.info(f'KG #0 score: {score_0}')
        scores_0.append(score_0)
        logger.info(f'KG #0 accumulating score mean: {np.mean(scores_0):.3f}, std: {np.std(scores_0):.3f}')
        # ask KG #1 with the question from KG #0
        score_1 = model_1.qa_evaluation(question_batch_0, dataset_1['entity_indices'], k=1)
        logger.info(f'KG #1 score of: {score_1}')
        scores_1.append(score_1)
        logger.info(f'KG #1 accumulating score mean: {np.mean(scores_1):.3f}, std: {np.std(scores_1):.3f}')
        if eval_round > 0:
            t_stat, p_value = ttest_ind(scores_0, scores_1)
            logger.info(f'Round {eval_round}: t-score = {t_stat}, p-value = {p_value}')
        if eval_round >= args.min_round and p_value <= args.p:
            significance = True
            logger.info(f'achieve stop condition with significance.')
            logger.info(f'the winner is KG #{0 if np.mean(scores_0)>np.mean(scores_1) else 1}.')
            break
        if eval_round >= args.max_round:
            logger.info(f'achieve stop condition of maximum evaluation rounds.')
            logger.info(f'the winner is KG #{0 if np.mean(scores_0)>np.mean(scores_1) else 1}.')
            break
    if not significance:
        logger.info(f'achieve stop condition without significance.')

    ret = {
        'KG0_score_mean': f'{np.mean(scores_0):.3f}', 'KG0_score_std': f'{np.std(scores_0):.3f}', 
        'KG1_score_mean': f'{np.mean(scores_1):.3f}', 'KG1_score_std': f'{np.std(scores_1):.3f}',
        'round': eval_round, 'sampling_per_KG': eval_round * 100,
        't_stat': t_stat, 'p-value': p_value, 'significance': significance,
        'KG0_score': scores_0, 'KG1_score': scores_1
        }
    with open(result_path, "w") as f:
        json.dump(ret, f, indent=4)


# NOTE: for synthetic evaluation, please make the synthetic KG and train the FKGE first, then load synthetic KG as args.kg0 or args.kg1.
if __name__ == "__main__":
    args = parse_args()
    date = datetime.datetime.now().strftime('%y%m%d-%H%M')
    result_dir  = f'./results/KG-quality-eval/baselines/fkge-{args.kge}/{args.kg0}-{args.kg1}'
    result_path = os.path.join(result_dir, f'{date}-results.json')
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    
    level  = 'DEBUG' if args.debug else 'INFO'
    logger = set_logger(f'fkge-kg-eval-{date}', logfile=result_path.replace('.json', '.log'), level=level)
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    try:
        main(args, result_path=result_path, logger=logger)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e