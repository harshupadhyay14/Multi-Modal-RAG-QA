from collections import defaultdict

def reciprocal_rank_fusion(results, top_k=5):
    scores = defaultdict(float)
    for res in results:
        for rank, (meta, _) in enumerate(res, 1):
            scores[meta['id']] += 1 / (50 + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
