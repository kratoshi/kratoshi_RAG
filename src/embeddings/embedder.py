from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert list of texts into normalized embeddings.
        Returns shape: (N, D)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  
            show_progress_bar=False,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        
        return self.embed_texts([query])[0]
