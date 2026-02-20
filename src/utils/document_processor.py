from dataclasses import dataclass
from typing import List, Dict
import os


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_txt(self, file_path: str) -> str:
        #Load a text file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        #Split text into overlapping chunks
        chunks = []
        start = 0
        text_length = len(text)
        chunk_id = 0

        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    metadata={
                        "source": source,
                        "chunk_id": chunk_id,
                        "start_char": start,
                        "end_char": end,
                    },
                )
            )

            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1

        return chunks
