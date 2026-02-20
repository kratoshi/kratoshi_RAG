from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class RAGGenerator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_new_tokens: int = 200,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.max_new_tokens = max_new_tokens

    def build_prompt(self, query: str, context: str) -> str:
        #Structured RAG prompt
        prompt = f"""
        You are a helpful AI assistant.

        Use ONLY the provided context to answer the question.

        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        return prompt.strip()

    @torch.inference_mode()
    def generate(self, query: str, context: str) -> Dict:
        #Generate answer from retrieved context
        prompt = self.build_prompt(query, context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "answer": answer,
            "prompt_length": inputs.input_ids.shape[1],
        }