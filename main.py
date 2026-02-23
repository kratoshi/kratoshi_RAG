# from src.utils.document_processor import DocumentProcessor

# processor = DocumentProcessor(chunk_size=350, chunk_overlap=50)

# text = processor.load_txt("data/sample_text.txt")
# chunks = processor.chunk_text(text, source="sample_text.txt")

# print(len(chunks))
# print(chunks[0])
# print(chunks[1])

# from src.embeddings.embedder import TextEmbedder

# embedder = TextEmbedder()

# vec = embedder.embed_query("What is machine learning?")
# print(vec.shape)

# texts = ["hello world", "machine learning is fun"]
# embs = embedder.embed_texts(texts)
# print(embs.shape)

# from src.retriever.retriever import RAGRetriever
# from src.embeddings.embedder import TextEmbedder
# from src.utils.document_processor import DocumentProcessor
# from src.retriever.faiss_index import FaissVectorStore

# processor = DocumentProcessor()
# embedder = TextEmbedder()

# text = processor.load_txt("data/sample_text.txt")
# chunks = processor.chunk_text(text, source="sample_text.txt")

# embeddings = embedder.embed_texts([c.text for c in chunks])

# store = FaissVectorStore(embedding_dim=embeddings.shape[1])
# store.add_embeddings(embeddings, chunks)

# retriever = RAGRetriever(embedder, store)

# result = retriever.retrieve("What is the role of UNDAC?")

# print(result["context"][:500])
# print(result["sources"])

# from src.generator.generator import RAGGenerator

# gen = RAGGenerator()

# context = "Machine learning is a field of AI that allows systems to learn from data."
# query = "What is AI?"

# result = gen.generate(query, context)

# print(result["answer"])