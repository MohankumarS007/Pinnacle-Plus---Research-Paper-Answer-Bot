"""
Maximal Marginal Relevance (MMR) Retriever for Pinnacle Plus Capstone.
Balances query relevance with document diversity to eliminate redundant chunks.
Preserves paper title, page number, paper_id, and computed MMR scores.
"""

import os
import sys
import time
from typing import List, Dict, Any
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.embeddings import get_embedding_model
from src.vectorstore import get_chroma_client, VECTORSTORE_DIR

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class MMRRetriever:
    """
    Maximal Marginal Relevance (MMR) retriever.
    Fetches top candidate pool (fetch_k) via dense retrieval, then greedily selects
    top_k items balancing relevance to query vs novelty relative to selected items.
    """
    def __init__(
        self,
        collection_name: str = "papers_opensource",
        persist_directory: str = VECTORSTORE_DIR,
        lambda_mult: float = 0.7
    ):
        self.client = get_chroma_client(persist_directory)
        self.embedding_model = get_embedding_model("open_source")
        self.collection = self.client.get_collection(name=collection_name)
        self.lambda_mult = lambda_mult
        self.method_name = f"mmr_lambda_{lambda_mult}"

    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves top_k chunks using MMR selection from a fetch_k candidate pool.
        """
        t0 = time.time()
        query_vector = np.array(self.embedding_model.embed_query(query), dtype=np.float32)

        # 1. Fetch initial candidate pool with embeddings
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances", "embeddings"]
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        embeddings = np.array(results["embeddings"][0], dtype=np.float32)
        n_candidates = len(docs)

        # 2. Compute query similarities for all candidates
        query_sims = np.array([cosine_similarity(query_vector, emb) for emb in embeddings])

        # 3. MMR greedy selection
        selected_indices: List[int] = []
        candidate_indices = list(range(n_candidates))

        for _ in range(min(top_k, n_candidates)):
            best_idx = -1
            best_mmr_score = -float('inf')

            for idx in candidate_indices:
                sim_to_query = query_sims[idx]
                if not selected_indices:
                    max_sim_to_selected = 0.0
                else:
                    max_sim_to_selected = max(
                        cosine_similarity(embeddings[idx], embeddings[s_idx])
                        for s_idx in selected_indices
                    )

                mmr_score = self.lambda_mult * sim_to_query - (1.0 - self.lambda_mult) * max_sim_to_selected
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)

        latency_ms = (time.time() - t0) * 1000

        retrieved = []
        for rank, idx in enumerate(selected_indices, 1):
            meta = metas[idx]
            retrieved.append({
                "rank": rank,
                "title": meta.get("title", "Unknown"),
                "page": int(meta.get("page", 1)),
                "paper_id": meta.get("paper_id", ""),
                "category": meta.get("category", ""),
                "score": round(float(query_sims[idx]), 4),
                "content": docs[idx],
                "retrieval_method": self.method_name,
                "latency_ms": round(latency_ms, 2)
            })

        return retrieved

if __name__ == '__main__':
    retriever = MMRRetriever(lambda_mult=0.7)
    q = "What is the Transformer architecture and self-attention mechanism?"
    res = retriever.retrieve(q, top_k=3, fetch_k=15)
    print(f"MMR retrieved {len(res)} results in {res[0]['latency_ms']} ms:")
    for r in res:
        print(f"[{r['rank']}] {r['title']} - Page {r['page']} (Sim: {r['score']})")

