# embeddings/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-mpnet-base-v2")

def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True)
