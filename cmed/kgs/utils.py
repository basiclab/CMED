import torch
import torch
import numpy as np
import torch.nn.functional as F
import math


def diversity_regularization(x):
    x = x / F.normalize(x, dim=-1, p=2)
    y = torch.flip(x, [0, 1])

    return torch.cdist(x, y, p=2).mean()



def evaluate(model, batch, t, hit_k=10):
    triplet, _ = batch
    h, r  = triplet[:2]
    pred_ranks = model.predict(h, r)

    hits = [[]]*10

    for (t_, pred_rank) in zip(t, pred_ranks):

        for idx in range(1,11):
            hits[idx-1].append((pred_rank[:idx] == t_).sum())

    for idx in range(10):
        hits[idx] = np.sum(hits[idx])

    return np.array(hits), len(t)

def hit_at_k(predictions: torch.Tensor, ground_truth_idx: torch.Tensor, k: int = 10) -> int:
    """Calculates number of hits@k.
    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :param device: device on which calculations are taking place
    :param k: number of top K results to be considered as hits
    :return: Hits@K score
    """
    assert predictions.size(0) == ground_truth_idx.size(0)

    zero_tensor = torch.tensor([0])
    one_tensor = torch.tensor([1])
    
    _, indices = predictions.topk(k=k, largest=False)
    if indices.is_cuda:
        indices = indices.cpu()
    if ground_truth_idx.is_cuda:
        ground_truth_idx = ground_truth_idx.cpu()

    return torch.where(indices == ground_truth_idx, one_tensor, zero_tensor).sum().item()


def mrr(predictions: torch.Tensor, ground_truth_idx: torch.Tensor) -> float:
    """Calculates mean reciprocal rank (MRR) for given predictions and ground truth values.
    :param predictions: BxN tensor of prediction values where B is batch size and N number of classes. Predictions
    must be sorted in class ids order
    :param ground_truth_idx: Bx1 tensor with index of ground truth class
    :return: Mean reciprocal rank score
    """
    assert predictions.size(0) == ground_truth_idx.size(0)

    indices = predictions.argsort()
    return (1.0 / (indices == ground_truth_idx).nonzero()[:, 1].float().add(1.0)).sum().item()


def chunks(lst, n):
    for idx in range(0, len(lst), n):
        yield lst[idx:idx+n]


def evaluate_types(model, batch, hits_k=10):
    triplet = batch[2]
    h, r, types, _, _  = triplet
    # torch.arange(end=entities_count, device=device)
    batch_size = h.size()[0]
    type_ids = torch.arange(end=model.num_types)
    all_types = type_ids.repeat(batch_size, 1)
    if h.is_cuda:
        type_ids = type_ids.cuda()


    heads = h.reshape(-1, 1).repeat(1, all_types.size()[1])
    relations = r.reshape(-1, 1).repeat(1, all_types.size()[1])
    heads = model.ent_embeddings(heads)
    relations = model.rel_embeddings(relations)
    hidden_dim = heads.shape[-1]
    all_types = model.type_embeddings(type_ids) #/ math.sqrt(hidden_dim)
    predictions =  model.score(heads, all_types, relations)

    if predictions.is_cuda:
        predictions = predictions.cpu()

    ground_truth_type_id = types
    if ground_truth_type_id.is_cuda:
        ground_truth_type_id = ground_truth_type_id.cpu()


    hits_score = [0]*hits_k 
    avg_size = types.shape[-1]
    mrr_score = 0
    for idx in range(avg_size):
        sub_types = ground_truth_type_id[:, [idx]]
        mrr_score += mrr(predictions, sub_types)

        for hit_k in range(hits_k):
            hits_score[hit_k] += hit_at_k(predictions, sub_types, k=hit_k)

    for hit_k in range(hits_k):
        hits_score[hit_k] /= avg_size

    return hits_score, mrr_score/avg_size

def evaluate_(model, batch, hits_k=10):
    triplet = batch[0]
    h, r, t  = triplet
    # torch.arange(end=entities_count, device=device)
    batch_size = h.size()[0]
    
    if model.num_entities < 1e6:
        entity_ids = torch.arange(end=model.num_entities)
        if h.is_cuda:
            entity_ids = entity_ids.cuda()
        all_entities = entity_ids.repeat(batch_size, 1)
        heads = h.reshape(-1, 1).repeat(1, all_entities.size()[1])
        relations = r.reshape(-1, 1).repeat(1, all_entities.size()[1])
        tails = t.reshape(-1, 1).repeat(1, all_entities.size()[1])
        tails_predictions =  model.score(*model(heads, relations, all_entities))

        heads_predictions =  model.score(*model(all_entities, relations, tails))

        predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
        ground_truth_entity_id = torch.cat((t.reshape(-1, 1), h.reshape(-1, 1)))
        if predictions.is_cuda:
            predictions = predictions.cpu()
        if ground_truth_entity_id.is_cuda:
            ground_truth_entity_id = ground_truth_entity_id.cpu()

        hits_score = [0]*hits_k
        for hit_k in range(hits_k):
            hits_score[hit_k] = hit_at_k(predictions, ground_truth_entity_id, k=hit_k)
        mrr_score = mrr(predictions, ground_truth_entity_id)
        # triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        return hits_score, mrr_score, batch_size
    else:
        all_entity_ids = torch.arange(end=model.num_entities)
        ground_truth_entity_ids = None
        tails_predictions = None
        heads_predictions = None

        for entity_ids in chunks(all_entity_ids, 50000):
            if h.is_cuda:
                entity_ids = entity_ids.cuda()
            all_entities = entity_ids.repeat(batch_size, 1)
            heads = h.reshape(-1, 1).repeat(1, all_entities.size()[1])
            relations = r.reshape(-1, 1).repeat(1, all_entities.size()[1])
            tails = t.reshape(-1, 1).repeat(1, all_entities.size()[1])

            tails_prediction =  model.score(*model(heads, relations, all_entities))
            heads_prediction =  model.score(*model(all_entities, relations, tails))

            if heads_prediction.is_cuda:
                heads_prediction = heads_prediction.cpu()

            if heads_predictions is None:
                heads_predictions = heads_prediction
            else:
                heads_predictions = torch.cat([heads_predictions, heads_prediction], dim=1)

            if tails_prediction.is_cuda:
                tails_prediction = tails_prediction.cpu()

            if tails_predictions is None:
                tails_predictions = tails_prediction
            else:
                tails_predictions = torch.cat([tails_predictions, tails_prediction], dim=1)


        predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
        ground_truth_entity_id = torch.cat((t.reshape(-1, 1), h.reshape(-1, 1)))

        if ground_truth_entity_id.is_cuda:
            ground_truth_entity_id = ground_truth_entity_id.cpu()

        hits_score = [0]*hits_k
        for hit_k in range(hits_k):
            hits_score[hit_k] = hit_at_k(predictions, ground_truth_entity_id, k=hit_k)
        mrr_score = mrr(predictions, ground_truth_entity_id)
        # triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
        return hits_score, mrr_score, batch_size

def negsamp_vectorized_bsearch(pos_inds, n_items, n_samp=32, items=None):
    """ Guess and check vectorized
    Assumes that we are allowed to potentially 
    return less than n_samp samples
    """
    if items is not None:
        raw_samps_idx = np.random.randint(0, len(items)-1, size=n_samp)
        raw_samps = items[raw_samps_idx]
    else:
        raw_samps = np.random.randint(0, n_items, size=n_samp)


    if len(pos_inds) > 0:
        ss = np.searchsorted(pos_inds, raw_samps)
        pos_mask = raw_samps == np.take(pos_inds, ss, mode='clip')
        neg_inds = raw_samps[~pos_mask]
        return neg_inds
    return raw_samps


def _calc(h, t, r, norm):
    return torch.norm(h + r - t, p=norm, dim=1).cpu().detach().numpy().tolist()

def evaluate(h, t, r, norm, entity_emb, relation_emb):
    h_e = entity_emb[h].unsqueeze(0).repeat(entity_emb.shape[0], 1)
    r_e = relation_emb[r].unsqueeze(0).repeat(entity_emb.shape[0], 1)
    t_e = entity_emb[t].unsqueeze(0).repeat(entity_emb.shape[0], 1)
    
    scores = _calc(h_e, entity_emb, r_e, norm)
    

    gold_score = scores[t]
    t_rank = 0
    print(gold_score)
    print(np.mean(scores), np.max(scores), np.min(scores))
    for ent_id, score in enumerate(scores):
        # if gold_score is larger than other, lower my rank
        if score <= gold_score:
            t_rank += 1

    h_rank = 0
    scores = _calc(entity_emb, t_e, r_e, norm)
    gold_score = scores[t]
    for ent_id, score in enumerate(scores):
        if score <= gold_score:
            h_rank += 1
    return t_rank, h_rank