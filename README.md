# 🧠 RAG Question Answering System (PyTorch + FAISS)

A production-ready Retrieval-Augmented Generation (RAG) pipeline built with PyTorch, FAISS, and FastAPI.

This project demonstrates how to design, build, and deploy a scalable semantic search + LLM generation system — the core pattern behind modern AI assistants and enterprise knowledge bots.

---

## 🚀 Features

* 🔍 Semantic document search using FAISS
* 🧩 Intelligent chunking pipeline
* 🧠 Transformer-based embeddings
* 💬 Retrieval-augmented generation
* ⚡ FastAPI inference service
* 🐳 Fully containerized with Docker
* 🧪 Modular, production-style codebase

---

## 🏗️ Architecture

```
User Query
   ↓
Embed Query
   ↓
FAISS Vector Search
   ↓
Retrieve Top-K Chunks
   ↓
Generator (LLM)
   ↓
Final Answer
```

### Components

| Component    | Responsibility                        |
| ------------ | ------------------------------------- |
| Chunker      | Splits documents into semantic chunks |
| Embedder     | Converts text → dense vectors         |
| Vector Store | Fast similarity search with FAISS     |
| Retriever    | Fetches relevant context              |
| Generator    | Produces final answer                 |
| API          | Exposes inference endpoint            |

---

## 📦 Project Structure

```
kratoshi_RAG/
│
├── data/
├── src/
│   ├── embeddings/
│   ├── utils/
│   ├── retriever/
│   ├── generator/
│   └── api/
│
├── tests/
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Setup

### 1️⃣ Clone repo

```bash
git clone https://github.com/kratoshi/kratoshi_RAG
cd kratoshi_RAG
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Locally

### Run pipeline test

```bash
python main.py
```

---

### Start API server

```bash
uvicorn src.api.main:app --reload
```

API docs:

```
http://127.0.0.1:8000/docs
```

---

## 🐳 Docker

### Build image

```bash
docker build -t kratoshi-rag .
```

---

### Run container

```bash
docker run -p 8000:8000 kratoshi-rag
```

---

## 🔌 API Usage

### Health check

```bash
GET /health
```

---

### Upload Documents

```bash
POST /ingest        # Only supports .txt
```

---

### Query endpoint

```bash
POST /query
```

**Request**

```json
{
  "query": "What is machine learning?"
}
```

**Response**

```json
{
  "answer": "...generated answer..."
}
```

---

## 🧪 Example Workflow

1. Load documents
2. Chunk text
3. Generate embeddings
4. Build FAISS index
5. Query via API
6. Generate answer

---

## 🔮 Future Improvements

* [ ] Streaming responses
* [ ] Hybrid search (BM25 + vectors)
* [ ] Evaluation pipeline
* [ ] Batch indexing pipeline
* [ ] Kubernetes deployment
* [ ] Caching layer
* [ ] Observability (Prometheus/Grafana)


