import torch
import torch.nn.functional as F


def extract_entities_relations(graph):
    entities = set()
    relations = set()
    triples = []
    for h, r, t in graph:
        entities.add(h)
        entities.add(t)
        relations.add(r)
        triples.append((h, r, t))
    return entities, relations, triples


def test_sim(model_0, model_1, shared_entity_index, shared_relation_index):
    with torch.no_grad():
        shared_entity_index = torch.LongTensor([shared_entity_index]).to(model_0.device)
        embed_0 = model_0.entity_embeddings(shared_entity_index)
        embed_1 = model_1.entity_embeddings(shared_entity_index)
        entity_sim = F.cosine_similarity(embed_0, embed_1, dim=-1)
        shared_relation_index = torch.LongTensor([shared_relation_index]).to(model_0.device)
        embed_0 = model_0.relation_embeddings(shared_relation_index)
        embed_1 = model_1.relation_embeddings(shared_relation_index)
        relation_sim = F.cosine_similarity(embed_0, embed_1, dim=1)
    return {
        'entity similarity': f'{entity_sim.mean():.5f}', 
        'relation similarity': f'{relation_sim.mean():.5f}'
    }