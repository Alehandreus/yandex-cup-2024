import os
from tqdm import tqdm
import jsonlines
import numpy as np
import torch


class MetricLogger:
    def __init__(self, alpha = 0.95) -> None:
        self.alpha = alpha
        self.exp = None
        self.metric = 0
        self.metric_list = []

    def update(self, metric) -> None:
        if self.exp is None:
            self.exp = metric
        else:
            self.exp = self.exp * self.alpha + metric * (1 - self.alpha)
        self.metric = metric
        self.metric_list.append(metric)

    def mean(self) -> float:
        return np.mean(self.metric_list)


def pairwise_distances_torch(embeddings, top_size, chunk_size=512, rerank=False):
    s_distances = pairwise_distances_torch_impl(embeddings, top_size, chunk_size=chunk_size)

    if not rerank:
        return s_distances

    weights = [25.0, 8.0, 5.0, 3.0, 2.0, 1.0] # slime 2022
    new_embeddings = embeddings * weights[0]
    for i, w in enumerate(weights[1:]):
        new_embeddings += w * embeddings[s_distances[:, i]]
    new_embeddings /= sum(weights)

    s_distances = pairwise_distances_torch_impl(new_embeddings, top_size, chunk_size=chunk_size)

    return s_distances


def pairwise_distances_torch_impl(embeddings, top_size, chunk_size, device="cuda"):
    res = torch.zeros((len(embeddings), top_size + 1), dtype=torch.int64, device=device)

    for i in tqdm(range(0, len(embeddings), chunk_size)):
        chunk = embeddings[i:i+chunk_size]        
        distances = chunk @ embeddings.T
        s_distances = torch.argsort(distances, dim=1, descending=True)
        res[i:i+chunk_size] = s_distances[:, :top_size+1]

    return res


def calculate_ranking_metrics_torch(embeddings, cliques, rerank=False, device="cuda"):    
    embeddings = embeddings.to(device)
    cliques = cliques.to(device)

    s_distances = pairwise_distances_torch(embeddings, len(embeddings) - 1, chunk_size=512, rerank=rerank).to(device)

    query_cliques = cliques[s_distances[:, 0]]
    search_cliques = cliques[s_distances[:, 1:]]

    query_cliques_square = query_cliques.unsqueeze(1).expand(-1, search_cliques.shape[-1])
    mask = (search_cliques == query_cliques_square)

    ranks = 1.0 / (mask.float().argmax(dim=1).float() + 1.0)

    cumsum = torch.cumsum(mask, dim=1)
    mask2 = mask * cumsum
    mask2 = mask2 / torch.arange(1, mask2.shape[-1] + 1).to(device).float()
    average_precisions = torch.sum(mask2, dim=1) / torch.sum(mask, dim=1)

    dcg_k = 100

    clique_sizes = torch.bincount(cliques)
    dcg = torch.sum(
        mask[:, :dcg_k] / torch.arange(1, dcg_k + 1, device=device).pow(0.5),
        dim=1,
    )
    n_relevant = torch.gather(clique_sizes, 0, query_cliques)
    ideal_preds = torch.arange(1, dcg_k + 1, device=device).expand(len(n_relevant), -1) / n_relevant[:, None]
    ideal_preds[ideal_preds <= 1] = 1
    ideal_preds[ideal_preds > 1] = 0

    idcg = torch.sum(
        ideal_preds[:, :dcg_k] / torch.arange(1, dcg_k + 1, device=device).pow(0.5),
        dim=1,
    )
    ndcg = dcg / idcg

    return {
        "ranks": ranks.cpu().numpy().astype(np.float64),
        "ap": average_precisions.cpu().numpy().astype(np.float64),
        "ndcg": ndcg.cpu().numpy().astype(np.float64),
    }


def save_test_predictions(predictions, output_dir) -> None:
    with open(os.path.join(output_dir, 'submission.txt'), 'w') as foutput:
        for query_item, query_nearest in predictions:
            foutput.write('{}\t{}\n'.format(query_item, '\t'.join(map(str,query_nearest))))


def save_predictions(outputs, output_dir) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key in outputs:
        if "_ids" in key:
            with jsonlines.open(os.path.join(output_dir, f"{key}.jsonl"), "w") as f:
                if len(outputs[key][0]) == 4:
                    for clique, anchor, pos, neg in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor, "positive_id": pos, "negative_id": neg})
                else:
                    for clique, anchor in outputs[key]:
                        f.write({"clique_id": clique, "anchor_id": anchor})
        else:
            np.save(os.path.join(output_dir, f"{key}.npy"), outputs[key])
