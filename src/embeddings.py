"""
Dual Embedding Models Module for Pinnacle Plus Capstone.
Implements:
  - Model A (Open-Source Hugging Face): sentence-transformers/all-MiniLM-L6-v2 (or BAAI/bge-m3)
  - Model B (Commercial): OpenAI text-embedding-3-small
Adheres to LangChain Embeddings interface for seamless vector store integration.
"""

import os
import time
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

class OpenSourceHuggingFaceEmbeddings:
    """
    Open-Source Embedding Model using Hugging Face SentenceTransformers.
    Default: sentence-transformers/all-MiniLM-L6-v2 (384 dims, fast, local, CPU-friendly)
    Alternative: BAAI/bge-m3 (1024 dims)
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model_type = "Open-Source (Hugging Face)"
        print(f"Loading Open-Source Embedding Model: {self.model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded {self.model_name} successfully. Dimensions: {self.dimension}")

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()


class CommercialOpenAIEmbeddings:
    """
    Commercial Embedding Model using OpenAI text-embedding-3-small.
    Dimensions: 1536 dims.
    Requires OPENAI_API_KEY in environment or .env file.
    """
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.model_type = "Commercial (OpenAI API)"
        self.dimension = 1536
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            print("WARNING: OPENAI_API_KEY not found in environment or .env.")
            print("To use OpenAI embeddings, please set OPENAI_API_KEY in your .env file.")
            self.client = None
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            print(f"Loaded Commercial OpenAI Embedding Model: {self.model_name} (Dims: {self.dimension})")

    def embed_documents(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        if not self.client:
            raise ValueError("OPENAI_API_KEY is required to generate OpenAI embeddings.")
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        if not self.client:
            raise ValueError("OPENAI_API_KEY is required to generate OpenAI embeddings.")
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding


def get_embedding_model(model_choice: str = "open_source"):
    """
    Factory function to retrieve embedding model instance.
    Choices: 'open_source' (Model A) or 'openai' (Model B).
    """
    if model_choice.lower() in ["open_source", "opensource", "hf", "bge", "minilm"]:
        return OpenSourceHuggingFaceEmbeddings()
    elif model_choice.lower() in ["commercial", "openai", "openai_small"]:
        return CommercialOpenAIEmbeddings()
    else:
        raise ValueError(f"Unknown embedding model choice: {model_choice}. Use 'open_source' or 'openai'.")


def compare_embedding_specs():
    """
    Displays comparative analysis table between Model A and Model B.
    """
    print("\n" + "=" * 70)
    print("           EMBEDDING MODELS COMPARISON EXPERIMENT (STEP 7)")
    print("=" * 70)
    print(f"{'Attribute':<22} | {'Model A (Open-Source)':<23} | {'Model B (Commercial)'}")
    print("-" * 70)
    print(f"{'Model Name':<22} | {'all-MiniLM-L6-v2':<23} | {'text-embedding-3-small'}")
    print(f"{'Provider':<22} | {'Hugging Face':<23} | {'OpenAI'}")
    print(f"{'Embedding Dims':<22} | {'384':<23} | {'1536'}")
    print(f"{'Cost':<22} | {'Free (Local CPU/GPU)':<23} | {'$0.02 / 1M tokens'}")
    print(f"{'Privacy & Offline':<22} | {'100% Local / On-Device':<23} | {'Requires Cloud Internet'}")
    print(f"{'Max Input Tokens':<22} | {'256-512 tokens':<23} | {'8191 tokens'}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    compare_embedding_specs()
    # Test local embedding generation
    model_a = get_embedding_model("open_source")
    test_query = "What is the Transformer self-attention mechanism?"
    vec = model_a.embed_query(test_query)
    print(f"Successfully generated test embedding vector for query.")
    print(f"Dimension: {len(vec)}, Sample snippet: {vec[:5]}...")

