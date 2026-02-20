from typing import List, Dict

from src.embeddings.embedder import TextEmbedder
from src.retriever.faiss_index import FaissVectorStore


class RAGRetriever:
    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: FaissVectorStore,
        max_context_chars: int = 2000,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.max_context_chars = max_context_chars

    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """Retrieve relevant context for a query."""
        query_vec = self.embedder.embed_query(query)

        results = self.vector_store.search(query_vec, top_k=top_k)

        context_pieces: List[str] = []
        sources: List[Dict] = []
        total_chars = 0

        for chunk, score in results:
            chunk_text = chunk.text.strip()

            if total_chars + len(chunk_text) > self.max_context_chars:
                break

            context_pieces.append(chunk_text)
            total_chars += len(chunk_text)

            sources.append(
                {
                    "source": chunk.metadata.get("source"),
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "score": score,
                }
            )

        context = "\n\n---\n\n".join(context_pieces)

        return {
            "query": query,
            "context": context,
            "sources": sources,
        }