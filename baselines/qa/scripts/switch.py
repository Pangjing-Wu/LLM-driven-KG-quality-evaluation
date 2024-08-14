import argparse
import datetime
import json
import os
import sys
import traceback

import numpy as np
import torch
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('.')
import config.kg
from baselines.qa.kge import KGEWrapper, RotatEModel, TransEModel
from baselines.qa.utils import extract_entities_relations, test_sim
from modules.utils.kg import RDFLibSPARQL
from utils.logging import set_logger
from utils.synthetic_kg import make_kg
sys.path.pop()


def parse_args():
    parser = argparse.ArgumentParser(description='Process the arguments for the dataset and model settings.')
    parser.add_argument('--kg0', type=str, required=True, help='KG #0 name')
    parser.add_argument('--kg1', type=str, required=True, help='KG #1 name')
    
    parser.add_argument('--kge', type=str, required=True, help='KG embedding method')
    parser.add_argument('--dim', type=int, default=50, help='hidden dimension size')
    parser.add_argument('--margin', type=float, default=10., help='margin of `nn.MarginRankingLoss`')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--pretrain_epoch', type=int, default=100, help='pretrain epochs')
    parser.add_argument('--switch_epoch', type=int, default=100, help='switch training epochs')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number')
    
    parser.add_argument('--p', type=float, default=0.05, help='p-value of t-test')
    parser.add_argument('--min_round', type=int, default=10, help='minimum evaluation round')
    parser.add_argument('--max_round', type=int, default=100, help='maximum evaluation round')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()



def main(args, result_path, logger):
    kg0_config = getattr(config.kg, f'{args.kg0}_configs')
    kg1_config = getattr(config.kg, f'{args.kg1}_configs')
    sparql_0 = RDFLibSPARQL(path=kg0_config['path'], domain=kg0_config['domain'])
    sparql_1 = RDFLibSPARQL(path=kg1_config['path'], domain=kg1_config['domain'])
    
    entities_0, relations_0, triples_0 = extract_entities_relations(sparql_0.kg)
    entities_1, relations_1, triples_1 = extract_entities_relations(sparql_1.kg)

    all_entities = entities_0.union(entities_1)
    all_relations = relations_0.union(relations_1)
    shared_entities = entities_0.intersection(entities_1)
    shared_relations = relations_0.intersection(relations_1)

    all_entity_index = {entity: idx for idx, entity in enumerate(all_entities)}
    all_relation_index = {relation: idx for idx, relation in enumerate(all_relations)}
    shared_entity_index = [v for k, v in all_entity_index.items() if k in shared_entities]
    shared_relation_index = [v for k, v in all_relation_index.items() if k in shared_relations]
    indexed_triples_0 = [(all_entity_index[h], all_relation_index[r], all_entity_index[t]) for h, r, t in triples_0]
    indexed_triples_1 = [(all_entity_index[h], all_relation_index[r], all_entity_index[t]) for h, r, t in triples_1]
    entities_0_index = [all_entity_index[e] for e in entities_0]
    entities_1_index = [all_entity_index[e] for e in entities_1]
    
    dataset_0 = TensorDataset(torch.tensor(indexed_triples_0, dtype=torch.long))
    dataset_1 = TensorDataset(torch.tensor(indexed_triples_1, dtype=torch.long))
    data_loader_0 = DataLoader(dataset_0, batch_size=args.bs, shuffle=True)    
    data_loader_1 = DataLoader(dataset_1, batch_size=args.bs, shuffle=True)
    test_loader_0 = DataLoader(dataset_0, batch_size=100, shuffle=False)
    test_loader_1 = DataLoader(dataset_1, batch_size=100, shuffle=False)
    
    if args.kge.lower() == 'transe':
        model_0 = TransEModel(len(all_entity_index), len(all_relation_index), args.dim, cuda=args.cuda, lr=args.lr, margin=args.margin)
        model_1 = TransEModel(len(all_entity_index), len(all_relation_index), args.dim, cuda=args.cuda, lr=args.lr, margin=args.margin)
    elif args.kge.lower() == 'rotate':
        model_0 = RotatEModel(len(all_entity_index), len(all_relation_index), args.dim, gamma=12., cuda=args.cuda, lr=args.lr, margin=args.margin)
        model_1 = RotatEModel(len(all_entity_index), len(all_relation_index), args.dim, gamma=12., cuda=args.cuda, lr=args.lr, margin=args.margin)
    else:
        raise ValueError(f'unknown KGE method {args.kge}')
    model_0 = KGEWrapper(model_0)
    model_1 = KGEWrapper(model_1)
    
    logger.info(f'start pretrain KGE model for {args.pretrain_epoch} epochs.')
    for e in range(args.pretrain_epoch):
        loss_0 = model_0.train(data_loader_0)
        loss_1 = model_1.train(data_loader_1)
        logger.debug(f'Pretrain Epoch {e}, model #0 training loss: {loss_0}')
        logger.debug(f'Pretrain Epoch {e}, model #1 training loss: {loss_1}')
        scores_0 = model_0.test(test_loader_0)
        scores_1 = model_1.test(test_loader_1)
        logger.info(f'Pretrain Epoch {e},  model #0 test scores: {scores_0}')
        logger.info(f'Pretrain Epoch {e},  model #1 test scores: {scores_1}')
    
    logger.info(f'start switch training KGE model for {args.switch_epoch} epochs.')
    for e in range(args.switch_epoch):
        if e % 2 == 0:
            logger.info(f'switch: model #0 -> KG #0, model #1 -> KG #1.')
            loss_0 = model_0.train(data_loader_0)
            loss_1 = model_1.train(data_loader_1)
            logger.debug(f'Switch Training Epoch {e}, model #0 training loss: {loss_0}')
            logger.debug(f'Switch Training Epoch {e}, model #1 training loss: {loss_1}')
            scores_0 = model_0.test(test_loader_0)
            scores_1 = model_1.test(test_loader_1)
            logger.info(f'Switch Training Epoch {e},  model #0 test scores: {scores_0}')
            logger.info(f'Switch Training Epoch {e},  model #1 test scores: {scores_1}')
        else:
            logger.info(f'switch: model #0 -> KG #1, model #1 -> KG #0.')
            loss_0 = model_0.train(data_loader_1)
            loss_1 = model_1.train(data_loader_0)
            logger.debug(f'Switch Training Epoch {e}, model #0 training loss: {loss_0}')
            logger.debug(f'Switch Training Epoch {e}, model #1 training loss: {loss_1}')
            scores_0 = model_0.test(test_loader_1)
            scores_1 = model_1.test(test_loader_0)
            logger.info(f'Switch Training Epoch {e},  model #0 test scores: {scores_0}')
            logger.info(f'Switch Training Epoch {e},  model #1 test scores: {scores_1}')
        similarity = test_sim(model_0.model, model_1.model, shared_entity_index, shared_relation_index)
        logger.info(f'Switch Training Epoch {e}, shared entities and relations similarity: {similarity}')
    
    # QA process
    question_loader_0 = DataLoader(dataset_0, batch_size=100, shuffle=True)
    question_loader_1 = DataLoader(dataset_1, batch_size=100, shuffle=True)
    
    scores_0 = list()
    scores_1 = list()
    significance = False
    # ask KG #0 with the question from KG #1
    for eval_round, (question_batch_0, question_batch_1) in enumerate(zip(question_loader_0, question_loader_1)):
        score_0 = model_0.qa_evaluation(question_batch_1, entities_0_index, k=1)
        logger.info(f'KG #0 score: {score_0}')
        scores_0.append(score_0)
        logger.info(f'KG #0 accumulating score mean: {np.mean(scores_0):.3f}, std: {np.std(scores_0):.3f}')
        # ask KG #1 with the question from KG #0
        score_1 = model_1.qa_evaluation(question_batch_0, entities_1_index, k=1)
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


if __name__ == "__main__":
    args = parse_args()
    date = datetime.datetime.now().strftime('%y%m%d-%H%M')
    result_dir  = f'./results/KG-quality-eval/baselines/switch-{args.kge}/{args.kg0}-{args.kg1}'
    result_path = os.path.join(result_dir, f'{date}-results.json')
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    
    level  = 'DEBUG' if args.debug else 'INFO'
    logger = set_logger(f'switch-kg-eval-{date}', logfile=result_path.replace('.json', '.log'), level=level)
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    try:
        main(args, result_path=result_path, logger=logger)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e