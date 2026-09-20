"""
Hybrid Retriever for Pinnacle Plus Capstone.
Combines Dense Vector Retrieval (semantic embeddings) with BM25 Keyword Search (lexical matching)
using Reciprocal Rank Fusion (RRF).
Preserves paper title, page number, paper_id, and fused ranking scores.
"""

import os
import sys
import re
import time
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.embeddings import get_embedding_model
from src.vectorstore import get_chroma_client, VECTORSTORE_DIR

def simple_tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25: lowercase alphanumeric tokens."""
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

class HybridRetriever:
    """
    Fuses Dense retrieval (ChromaDB) and Sparse retrieval (BM25Okapi)
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self,
        collection_name: str = "papers_opensource",
        persist_directory: str = VECTORSTORE_DIR,
        rrf_k: int = 60
    ):
        self.client = get_chroma_client(persist_directory)
        self.embedding_model = get_embedding_model("open_source")
        self.collection = self.client.get_collection(name=collection_name)
        self.rrf_k = rrf_k
        self.method_name = "hybrid_dense_bm25_rrf"

        print("Initializing BM25 index over vectorstore chunks...")
        t0 = time.time()
        # Fetch all indexed items from Chroma
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.all_ids = all_data["ids"]
        self.all_docs = all_data["documents"]
        self.all_metas = all_data["metadatas"]

        # Build map: id -> (document, metadata)
        self.id_to_data = {
            doc_id: (doc_text, meta)
            for doc_id, doc_text, meta in zip(self.all_ids, self.all_docs, self.all_metas)
        }

        # Tokenize corpus for BM25
        tokenized_corpus = [simple_tokenize(doc) for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built over {len(self.all_docs)} chunks in {time.time() - t0:.2f}s.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_candidates: int = 25,
        sparse_candidates: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. Dense retrieval: queries ChromaDB for top dense candidates.
        2. Sparse retrieval: queries BM25 for top keyword candidates.
        3. Reciprocal Rank Fusion: combines ranks using 1 / (rrf_k + rank).
        """
        t0 = time.time()

        # 1. Dense retrieval
        query_vec = self.embedding_model.embed_query(query)
        dense_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=dense_candidates,
            include=["documents", "metadatas", "distances"]
        )

        dense_ranked_ids = []
        if dense_results and dense_results["ids"] and dense_results["ids"][0]:
            dense_ranked_ids = dense_results["ids"][0]

        # 2. Sparse BM25 retrieval
        tokenized_query = simple_tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Top sparse indices sorted descending
        top_sparse_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:sparse_candidates]
        sparse_ranked_ids = [self.all_ids[idx] for idx in top_sparse_indices]

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}

        for rank, doc_id in enumerate(dense_ranked_ids, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, doc_id in enumerate(sparse_ranked_ids, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort fused items by RRF score
        fused_sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        latency_ms = (time.time() - t0) * 1000

        retrieved = []
        for rank, doc_id in enumerate(fused_sorted_ids, 1):
            doc_text, meta = self.id_to_data[doc_id]
            retrieved.append({
                "rank": rank,
                "title": meta.get("title", "Unknown"),
                "page": int(meta.get("page", 1)),
                "paper_id": meta.get("paper_id", ""),
                "category": meta.get("category", ""),
                "score": round(rrf_scores[doc_id], 5),
                "content": doc_text,
                "retrieval_method": self.method_name,
                "latency_ms": round(latency_ms, 2)
            })

        return retrieved

if __name__ == '__main__':
    retriever = HybridRetriever()
    q = "What is the Transformer architecture and self-attention mechanism?"
    res = retriever.retrieve(q, top_k=3)
    print(f"\nHybrid retrieved {len(res)} results in {res[0]['latency_ms']} ms:")
    for r in res:
        print(f"[{r['rank']}] {r['title']} - Page {r['page']} (RRF: {r['score']})")

