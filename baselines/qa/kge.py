import math

import torch
import torch.nn as nn


class TransEModel(nn.Module):
    
    def __init__(self, num_entities, num_relations, embedding_dim, cuda, margin=10., lr=1e-3):
        super().__init__()
        self.device = f'cuda:{cuda}'
        self.entity_embeddings = self.__init_entity_emb(num_entities, embedding_dim)
        self.relation_embeddings = self.__init_relation_emb(num_relations, embedding_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MarginRankingLoss(margin=margin)
        self.to(self.device)
        
    def forward(self, h, r, t):
        h_e = self.entity_embeddings(h)
        r_e = self.relation_embeddings(r)
        t_e = self.entity_embeddings(t)
        return torch.norm(h_e + r_e - t_e, p=1, dim=1)

    def __init_entity_emb(self, num_entities, dim):
        entities_emb = nn.Embedding(
            num_embeddings=num_entities + 1,
            embedding_dim=dim,
            padding_idx=num_entities)
        uniform_range = 6 / math.sqrt(dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb

    def __init_relation_emb(self, num_relations, dim):
        relations_emb = nn.Embedding(
            num_embeddings=num_relations + 1,
            embedding_dim=dim,
            padding_idx=num_relations)
        uniform_range = 6 / math.sqrt(dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb


class RotatEModel(nn.Module):
    
    def __init__(self, num_entities, num_relations, embedding_dim, gamma, cuda, margin=10., lr=1e-3):
        super().__init__()
        self.device = f'cuda:{cuda}'
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.epsilon = 2.0
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim * 2)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, a=-self.epsilon, b=self.epsilon)
        nn.init.uniform_(self.relation_embeddings.weight, a=-self.epsilon, b=self.epsilon)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MarginRankingLoss(margin=margin)
        self.to(self.device)
        
    def forward(self, h, r, t):
        h = self.entity_embeddings(h)
        r = self.relation_embeddings(r)
        t = self.entity_embeddings(t)
        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)
        phase_relation = r / (self.embedding_dim / math.pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return self.gamma - score.sum(dim=-1)
    
    
class KGEWrapper(object):
    
    def __init__(self, model) -> None:
        self.model = model
    
    def train(self, dataloader):
        total_loss = 0
        for batch in dataloader:
            batch = batch[0].to(self.model.device)
            heads, relations, tails = batch[:, 0], batch[:, 1], batch[:, 2]
            head_or_tail = torch.randint(2, heads.size(), device=self.model.device)
            random_entities = torch.randint(self.model.entity_embeddings.weight.size(0), heads.size(), device=self.model.device)
            corrupted_heads = torch.where(head_or_tail == 1, random_entities, heads)
            corrupted_tails = torch.where(head_or_tail == 0, random_entities, tails)
            self.model.optimizer.zero_grad()
            positive_scores = self.model(heads, relations, tails)
            negative_scores = self.model(corrupted_heads, relations, corrupted_tails)
            loss = self.model.criterion(positive_scores, negative_scores, torch.tensor([-1], dtype=torch.long, device=self.model.device))
            loss.backward()
            self.model.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def test(self, dataloader):
        examples_count = 0
        hits_at_1, hits_at_5, hits_at_10, mrr_score = 0, 0, 0 ,0
        num_entities = self.model.entity_embeddings.weight.size(0)
        entity_ids = torch.arange(num_entities, device=self.model.device)
        for batch in dataloader:
            batch = batch[0]
            heads, relations, tails = batch[:, 0], batch[:, 1], batch[:, 2]
            batch_size = heads.size(0)
            heads, relations, tails = heads.to(self.model.device), relations.to(self.model.device), tails.to(self.model.device)
            # test tail and head prediction
            with torch.no_grad():
                tail_prediction = self.model(
                    heads[:, None].expand(-1, num_entities).flatten(),
                    relations[:, None].expand(-1, num_entities).flatten(),
                    entity_ids.expand(batch_size, -1).flatten()
                ).view(batch_size, -1)
                head_prediction = self.model(
                    entity_ids.expand(batch_size, -1).flatten(),
                    relations[:, None].expand(-1, num_entities).flatten(),
                    tails[:, None].expand(-1, num_entities).flatten()
                ).view(batch_size, -1)
            # evaluate hit@1, hit@5, hits@10, and MRR
            prediction = torch.cat((tail_prediction, head_prediction), dim=0)
            ground_truth = torch.cat((tails[:, None], heads[:, None]))
            hits_at_1 += self.__hit_at_k(prediction, ground_truth, k=1)
            hits_at_5 += self.__hit_at_k(prediction, ground_truth, k=5)
            hits_at_10 += self.__hit_at_k(prediction, ground_truth, k=10)
            mrr_score += self.__mrr(prediction, ground_truth)
            examples_count += batch_size
        return {
            'hit@1': f'{hits_at_1 / examples_count * 100:.3f}', 
            'hit@5': f'{hits_at_5 / examples_count * 100:.3f}', 
            'hit@10': f'{hits_at_10 / examples_count * 100:.3f}', 
            'MRR': f'{mrr_score / examples_count * 100:.3f}'
        }
        
    def qa_evaluation(self, question_batch, valid_entities, k) -> float:
        valid_entities = torch.tensor(valid_entities, dtype=torch.long, device=self.model.device)
        num_entities = self.model.entity_embeddings.weight.size(0)
        entity_ids = torch.arange(num_entities, device=self.model.device)
        batch = question_batch[0]
        heads, relations, tails = batch[:, 0], batch[:, 1], batch[:, 2]
        batch_size = heads.size(0)
        heads, relations, tails = heads.to(self.model.device), relations.to(self.model.device), tails.to(self.model.device)
        with torch.no_grad():
            tail_prediction = self.model(
                heads[:, None].expand(-1, num_entities).flatten(),
                relations[:, None].expand(-1, num_entities).flatten(),
                entity_ids.expand(batch_size, -1).flatten()
            ).view(batch_size, -1)
        valid_index = torch.nonzero((tails[:, None] == valid_entities).any(dim=1)).squeeze()
        tail_prediction, tails = tail_prediction[valid_index], tails[valid_index]
        return self.__hit_at_k(tail_prediction, tails[:, None], k=k) / batch_size * 100
    
    def __hit_at_k(self, prediction: torch.Tensor, ground_truth_idx: torch.Tensor, k: int = 10) -> int:
        assert prediction.size(0) == ground_truth_idx.size(0)
        _, indices = prediction.topk(k, dim=1, largest=False)
        hits = (indices == ground_truth_idx).any(dim=-1).sum().item()
        return hits

    def __mrr(self, prediction: torch.Tensor, ground_truth_idx: torch.Tensor) -> float:
        assert prediction.size(0) == ground_truth_idx.size(0)
        _, sorted_indices = prediction.sort(dim=1)
        ranks = (sorted_indices == ground_truth_idx).nonzero(as_tuple=False)[:, 1] + 1
        reciprocal_ranks = 1.0 / ranks.float()
        return reciprocal_ranks.sum().item()