import argparse
import datetime
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict

import networkx as nx
import numpy as np

sys.path.append('.')
import config.kg
from modules.utils.kg import RDFLibSPARQL
from utils.logging import set_logger
from utils.synthetic_kg import make_kg
sys.path.pop()


def parse_args():
    parser = argparse.ArgumentParser(description='Process the arguments for the dataset and model settings.')
    parser.add_argument('--kg', type=str, required=True, help='KG name')
    parser.add_argument('--synthetic_mode', type=str, help='Mode for synthetic KG generation {delete/fabricate}')
    parser.add_argument('--portion', type=float, help='portion of synthetic KG portion')
    parser.add_argument('--sampling_ratio', default=0.01, type=float, help='sampling ratio for approximate average path length')
    parser.add_argument('--repeat', type=int, default=1, help='repeat times for evaluation')
    parser.add_argument('--approximate', action='store_true', help='use approximate metric')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser.parse_args()

def number_of_connected_components(graph):
    return nx.number_connected_components(graph)

def diameter(graph):
    if nx.is_connected(graph):
        return nx.diameter(graph)
    else:
        diameters = [nx.diameter(graph.subgraph(c)) for c in nx.connected_components(graph)]
        return max(diameters)

def approximate_diameter(graph):
    """Uses the double sweep algorithm, performing two BFS searches 
    to estimate the graph's diameter efficiently."""
    arbitrary_node = next(iter(graph.nodes))
    distances = nx.single_source_shortest_path_length(graph, arbitrary_node)
    furthest_node = max(distances, key=distances.get)
    distances = nx.single_source_shortest_path_length(graph, furthest_node)
    diam = max(distances.values())
    return diam

def average_path_length(graph):
    if nx.is_connected(graph):
        return nx.average_shortest_path_length(graph)
    else:
        lengths = [nx.average_shortest_path_length(graph.subgraph(c)) for c in nx.connected_components(graph)]
        return sum(lengths) / len(lengths)

def approximate_average_path_length(graph, sample_ratio=0.01):
    """Samples a subset of nodes and calculates shortest paths from 
    these nodes to approximate the graph's average path length."""
    nodes = list(graph.nodes)
    sample_nodes = random.sample(nodes, int(len(nodes) * sample_ratio))
    path_lengths = []
    for node in sample_nodes:
        lengths = nx.single_source_shortest_path_length(graph, node)
        path_lengths.extend(lengths.values())
    avg_path_len = sum(path_lengths) / len(path_lengths)
    return avg_path_len

def entity_relation_diversity(kg):
    entities = set()
    relations = set()
    for h, r, t in kg:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return len(entities), len(relations), len(entities) / len(kg), len(relations) / len(kg)


def main(args, result_path, logger):
    kg_config = getattr(config.kg, f'{args.kg}_configs')
    sparql = RDFLibSPARQL(path=kg_config['path'], domain=kg_config['domain'])
    ret = defaultdict(list)
    
    for i in range(args.repeat):
        logger.info(f'start {i + 1} round evaluation')
        if args.synthetic_mode and args.portion:
            sparql = make_kg(sparql, mode=args.synthetic_mode, portion=args.portion)
        graph = nx.Graph()
        for h, _, t in sparql.kg:
            graph.add_edge(h, t)
        
        start_time = time.time()
        ret['num_components'].append(number_of_connected_components(graph))
        logger.info(f'time taken for num components: {time.time() - start_time:.4f} seconds')
        
        if args.approximate:
            start_time = time.time()
            ret['graph_diameter'].append(approximate_diameter(graph))
            logger.info(f'time taken for graph diameter: {time.time() - start_time:.4f} seconds')
            
            start_time = time.time()
            ret['avg_path_len'].append(approximate_average_path_length(graph))
            logger.info(f'time taken for avg path len: {time.time() - start_time:.4f} seconds')
        else:
            start_time = time.time()
            ret['graph_diameter'].append(diameter(graph))
            logger.info(f'time taken for graph diameter: {time.time() - start_time:.4f} seconds')
            
            start_time = time.time()
            ret['avg_path_len'].append(average_path_length(graph))
            logger.info(f'time taken for avg path len: {time.time() - start_time:.4f} seconds')
        
        start_time = time.time()
        n_entity, n_relation, entity_div, relation_diversity = entity_relation_diversity(sparql.kg)
        ret['entity_number'].append(n_entity)
        ret['relation_number'].append(n_relation)
        ret['entity_div'].append(entity_div)
        ret['relation_div'].append(relation_diversity)
        logger.info(f'time taken for entity & relation div: {time.time() - start_time:.4f} seconds')
        
        if args.repeat > 1 and (not args.synthetic_mode or not args.portion):
            logger.warning(f'stop repeat early because no specified synthetic mode or portion.')
            break
        
    ret = {k: round(np.mean(v), 5) for k, v in ret.items()}
    with open(result_path, 'w') as f:
        json.dump(ret, f, indent=4)
    

if __name__ == "__main__":
    args = parse_args()
    date = datetime.datetime.now().strftime('%y%m%d-%H%M')
    if args.synthetic_mode and args.portion:
        result_dir  = f'./results/KG-quality-eval/baselines/static-stats/{args.kg}-{args.synthetic_mode}/{args.portion}'
    else:
        result_dir  = f'./results/KG-quality-eval/baselines/static-stats/{args.kg}'
    result_path = os.path.join(result_dir, f'{date}-results.json')
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    
    level  = 'DEBUG' if args.debug else 'INFO'
    logger = set_logger(f'static-kg-eval-{date}', logfile=result_path.replace('.json', '.log'), level=level)
    logger.info('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    try:
        main(args, result_path=result_path, logger=logger)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e