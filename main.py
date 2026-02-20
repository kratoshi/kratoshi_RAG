# from src.utils.document_processor import DocumentProcessor

# processor = DocumentProcessor(chunk_size=350, chunk_overlap=50)

# text = processor.load_txt("data/sample_text.txt")
# chunks = processor.chunk_text(text, source="sample_text.txt")

# print(len(chunks))
# print(chunks[0])
# print(chunks[1])

from src.embeddings.embedder import TextEmbedder

embedder = TextEmbedder()

# vec = embedder.embed_query("What is machine learning?")
# print(vec.shape)

texts = ["hello world", "machine learning is fun"]
embs = embedder.embed_texts(texts)
print(embs.shape)
