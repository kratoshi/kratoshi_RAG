from typing import List, Tuple
import numpy as np
import faiss
import pickle
import os
from src.utils.document_processor import DocumentChunk


class FaissVectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[DocumentChunk] = []

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunks: List[DocumentChunk],
    ):
        #Add embeddings and their corresponding chunks
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have same length")

        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        #Search for most similar chunks
        query_embedding = query_embedding.astype("float32").reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    def save(self, path: str):
        #Persist index and metadata
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str):
        #Load index and metadata
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
