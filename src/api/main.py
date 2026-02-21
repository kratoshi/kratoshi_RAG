from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os

from src.embeddings.embedder import TextEmbedder
from src.utils.document_processor import DocumentProcessor
from src.retriever.faiss_index import FaissVectorStore
from src.retriever.retriever import RAGRetriever
from src.generator.generator import RAGGenerator


# Initialize App
app = FastAPI(title="Kratoshi_RAG")

# Load Embedder, Retriever and Generator
embedder = TextEmbedder()
processor = DocumentProcessor()
generator = RAGGenerator()

vector_store = None
retriever = None

INDEX_PATH = "data/index"

# Request models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# Startup event
@app.on_event("startup")
def startup_event():
    global vector_store, retriever

    vector_store = FaissVectorStore(embedding_dim=384)

    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        vector_store.load(INDEX_PATH)

    retriever = RAGRetriever(embedder, vector_store)


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Document ingestion
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    global vector_store

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported")

    content = (await file.read()).decode("utf-8")

    chunks = processor.chunk_text(content, source=file.filename)
    embeddings = embedder.embed_texts([c.text for c in chunks])

    vector_store.add_embeddings(embeddings, chunks)
    vector_store.save(INDEX_PATH)

    return {
        "message": "Document ingested",
        "chunks_added": len(chunks),
    }


# Query endpoint
@app.post("/query")
def query(request: QueryRequest):
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    retrieved = retriever.retrieve(request.query, top_k=request.top_k)

    generation = generator.generate(
        query=request.query,
        context=retrieved["context"],
    )

    return {
        "query": request.query,
        "answer": generation["answer"],
        "sources": retrieved["sources"],
    }