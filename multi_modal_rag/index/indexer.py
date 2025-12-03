# index/indexer.py
import faiss
import numpy as np

class FaissIndexer:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []

    def add(self, embeddings, metas):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.metadatas.extend(metas)

    def search(self, q_emb, top_k=5):
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb.astype('float32'), top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.metadatas):
                results.append((self.metadatas[idx], score))
        return results
