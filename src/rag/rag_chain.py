"""
End-to-End RAG Pipeline Chain for Pinnacle Plus Capstone.
Assembles:
  User Question -> Hybrid Search -> Cross-Encoder Reranker -> Context Builder -> Grounded LLM -> Formatted Answer with Top-3 Citations.
"""

import os
import sys
import time
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankedRetriever
from src.rag.context_builder import format_context, extract_top_sources
from src.rag.llm import LLMClient
from src.rag.response_formatter import format_rag_output

class RAGPipeline:
    """
    Main RAG pipeline coordinating retrieval, context injection, generation, and citation formatting.
    """
    def __init__(
        self,
        retriever=None,
        llm_model: str = "gpt-4o-mini"
    ):
        print("\n--- Initializing Pinnacle Plus RAG Pipeline ---")
        if retriever is None:
            hybrid = HybridRetriever()
            self.retriever = RerankedRetriever(base_retriever=hybrid)
        else:
            self.retriever = retriever

        self.llm = LLMClient(model_name=llm_model)
        print("RAG Pipeline ready.\n")

    def answer_question(self, question: str, top_k: int = 3, initial_candidates: int = 15) -> Dict[str, Any]:
        """
        Executes end-to-end question answering:
        1. Retrieves candidates using two-stage retriever (Hybrid + Reranker).
        2. Builds grounded context and extracts top-3 supporting citations.
        3. Generates grounded answer via LLM.
        4. Returns structured answer package with citations.
        """
        t0 = time.time()

        # 1. Retrieval
        retrieved_docs = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            initial_candidates=initial_candidates
        )

        # 2. Context formulation
        context_str = format_context(retrieved_docs)
        top_sources = extract_top_sources(retrieved_docs, top_k=top_k)

        # 3. LLM Generation
        answer = self.llm.generate(
            question=question,
            context=context_str,
            retrieved_docs=retrieved_docs
        )

        latency_ms = (time.time() - t0) * 1000

        result = {
            "question": question,
            "answer": answer,
            "sources": top_sources,
            "model": getattr(self.llm, "last_provider", self.llm.mode),
            "retrieval_method": self.retriever.method_name,
            "latency_ms": round(latency_ms, 2)
        }

        return result

if __name__ == '__main__':
    pipeline = RAGPipeline()
    q = "What is the key difference between scaled dot-product attention and multi-head attention in Transformer architecture?"
    res = pipeline.answer_question(q, top_k=3)
    print(format_rag_output(res))

