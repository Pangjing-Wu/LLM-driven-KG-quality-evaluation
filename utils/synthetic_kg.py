import copy
import hashlib
import random
from collections import Counter

import rdflib

from modules.utils.kg import RDFLibSPARQL


def delete_triples(sparql: RDFLibSPARQL, portion: float) -> RDFLibSPARQL:
    if not (0 <= portion <= 1):
        raise ValueError("Portion must be between 0 and 1.")
    sparql = copy.deepcopy(sparql)
    triples = list(sparql.kg)
    n = int(len(triples) * portion)
    for triple in random.sample(triples, n):
        sparql.kg.remove(triple)
    return sparql


def fabricate_triples(sparql: RDFLibSPARQL, portion: float):
    if not (0 <= portion <= 1):
        raise ValueError("Portion must be between 0 and 1.")
    short_hash = lambda value: hashlib.md5(value.encode()).hexdigest()[:8]
    sparql = copy.deepcopy(sparql)
    entity_counter = Counter()
    for h, _, _ in sparql.kg:
        entity_counter[h] += 1
    sorted_entities = sorted(entity_counter.items(), key=lambda item: item[1])
    target_corruption_count = int(len(sparql.kg) * portion)
    selected_entities = set()
    current_count = 0
    for entity, count in sorted_entities:
        if current_count + count > target_corruption_count:
            break
        selected_entities.add(entity)
        current_count += count
    
    domain = sparql.domain if sparql.domain else ''
    for h, r, t in sparql.kg:
        if h in selected_entities and t in selected_entities:
            sparql.kg.remove((h, r, t))
            h = rdflib.URIRef(f"{domain}UnKnown_#{short_hash(str(h))}")
            t = rdflib.URIRef(f"{domain}UnKnown_#{short_hash(str(t))}")
            sparql.kg.add((h, r, t))
        elif h in selected_entities:
            sparql.kg.remove((h, r, t))
            h = rdflib.URIRef(f"{domain}UnKnown_#{short_hash(str(h))}")
            sparql.kg.add((h, r, t))
        elif t in selected_entities:
            sparql.kg.remove((h, r, t))
            t = rdflib.URIRef(f"{domain}UnKnown_#{short_hash(str(t))}")
            sparql.kg.add((h, r, t))
    return sparql


def make_kg(sparql: RDFLibSPARQL, mode: str, portion: float) -> RDFLibSPARQL:
    if mode == 'delete':
        return delete_triples(sparql, portion)
    elif mode == 'fabricate':
        return fabricate_triples(sparql, portion)
    else:
        raise ValueError(f'unknown synthetic mode `{mode}`, expect `delete` and `fabricate`.')