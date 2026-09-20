"""
Cross-Encoder Reranker for Pinnacle Plus Capstone.
Takes candidate documents retrieved from Dense or Hybrid search,
computes joint query-passage cross-attention scores using a pretrained CrossEncoder,
and reranks to produce precise top-K results with preserved metadata.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class CrossEncoderReranker:
    """
    Reranks initial candidate documents using a pretrained cross-encoder model.
    Default model: 'cross-encoder/ms-marco-MiniLM-L-6-v2' (lightweight, high-precision).
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading Cross-Encoder Reranker model: {model_name}...")
        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        self.method_name = "cross_encoder_reranked"
        print("Cross-Encoder loaded successfully.")

    def rerank(self, query: str, candidate_docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Takes candidate documents (from Dense or Hybrid retriever),
        scores them with the cross-encoder, and sorts descending.
        """
        if not candidate_docs:
            return []

        t0 = time.time()
        # Prepare pairs: (query, passage)
        pairs = [(query, doc["content"]) for doc in candidate_docs]

        # Compute cross-encoder relevance logits
        scores = self.model.predict(pairs)

        # Attach reranker scores
        scored_candidates = []
        for doc, score in zip(candidate_docs, scores):
            item = dict(doc)
            item["reranker_score"] = round(float(score), 4)
            scored_candidates.append(item)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)

        reranked_results = []
        latency_ms = (time.time() - t0) * 1000

        for rank, doc in enumerate(scored_candidates[:top_k], 1):
            doc["rank"] = rank
            doc["retrieval_method"] = self.method_name
            doc["rerank_latency_ms"] = round(latency_ms, 2)
            reranked_results.append(doc)

        return reranked_results

class RerankedRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Base Retriever (Dense or Hybrid) fetches top-N candidates (e.g. N=15)
    Stage 2: Cross-Encoder scores and reranks candidates to output final top-K (e.g. K=5)
    """
    def __init__(self, base_retriever, reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.base_retriever = base_retriever
        self.reranker = CrossEncoderReranker(reranker_model_name)
        self.method_name = f"{base_retriever.method_name}+reranker"

    def retrieve(self, query: str, top_k: int = 5, initial_candidates: int = 15) -> List[Dict[str, Any]]:
        t_total_0 = time.time()
        # Stage 1: Initial retrieval
        candidates = self.base_retriever.retrieve(query, top_k=initial_candidates)
        # Stage 2: Reranking
        final_docs = self.reranker.rerank(query, candidates, top_k=top_k)
        total_latency = (time.time() - t_total_0) * 1000

        for d in final_docs:
            d["total_pipeline_latency_ms"] = round(total_latency, 2)
            d["retrieval_method"] = self.method_name

        return final_docs

if __name__ == '__main__':
    from src.retrieval.dense_retriever import DenseRetriever
    base = DenseRetriever()
    two_stage = RerankedRetriever(base)
    q = "What is the Transformer architecture and self-attention mechanism?"
    res = two_stage.retrieve(q, top_k=3, initial_candidates=10)
    print(f"\nReranked retrieved {len(res)} results in {res[0]['total_pipeline_latency_ms']} ms:")
    for r in res:
        print(f"[{r['rank']}] {r['title']} - Page {r['page']} (Reranker Score: {r['reranker_score']})")

