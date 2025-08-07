import faiss
import numpy as np
from embedder import embed_text

class VectorStore:
    def __init__(self, dim=768):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []
        self.embeddings = []

    def add_chunks(self, chunks):
        embeds = embed_text(chunks)
        vectors = np.array(embeds).astype('float32')
        self.index.add(vectors)
        self.embeddings.extend(vectors)
        self.chunks.extend(chunks)

    def search(self, query, k=3):
        q_embed = embed_text([query])[0]
        q_vector = np.array([q_embed]).astype('float32')
        distances, indices = self.index.search(q_vector, k)
        return [self.chunks[i] for i in indices[0]]
