"""
Dense Similarity Retriever for Pinnacle Plus Capstone.
Implements baseline vector similarity search against persistent ChromaDB collection.
Preserves paper title, page number, paper_id, and similarity scores.
"""

import os
import sys
import time
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import chromadb
from src.embeddings import get_embedding_model
from src.vectorstore import get_chroma_client, VECTORSTORE_DIR

class DenseRetriever:
    """
    Performs standard dense/cosine similarity search using vector embeddings.
    """
    def __init__(self, collection_name: str = "papers_opensource", persist_directory: str = VECTORSTORE_DIR):
        self.client = get_chroma_client(persist_directory)
        self.embedding_model = get_embedding_model("open_source")
        self.collection = self.client.get_collection(name=collection_name)
        self.method_name = "dense_cosine"

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most similar chunks for a given query.
        """
        t0 = time.time()
        query_vector = self.embedding_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        latency_ms = (time.time() - t0) * 1000

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for rank, (doc_text, meta, dist) in enumerate(zip(docs, metas, distances), 1):
                similarity = 1.0 - float(dist)
                retrieved.append({
                    "rank": rank,
                    "title": meta.get("title", "Unknown"),
                    "page": int(meta.get("page", 1)),
                    "paper_id": meta.get("paper_id", ""),
                    "category": meta.get("category", ""),
                    "score": round(similarity, 4),
                    "content": doc_text,
                    "retrieval_method": self.method_name,
                    "latency_ms": round(latency_ms, 2)
                })

        return retrieved

if __name__ == '__main__':
    retriever = DenseRetriever()
    q = "What is the Transformer architecture and self-attention mechanism?"
    res = retriever.retrieve(q, top_k=3)
    print(f"Retrieved {len(res)} results in {res[0]['latency_ms']} ms:")
    for r in res:
        print(f"[{r['rank']}] {r['title']} - Page {r['page']} (Score: {r['score']})")

