"""
Retrieval Evaluation & Benchmarking Engine for Pinnacle Plus Capstone.
Runs the 10 canonical test questions through:
  1. Dense Similarity Search (Cosine)
  2. Maximum Marginal Relevance (MMR)
  3. Hybrid Search (Dense + BM25 with RRF)
  4. Two-Stage Reranking (Hybrid + Cross-Encoder)
Computes Hit@1, Hit@3, Hit@5, MRR, Latency, and saves structured experiment records.
"""

import os
import sys
import json
import csv
import time
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.mmr_retriever import MMRRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankedRetriever

EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments', 'retrieval')
QUERIES_PATH = os.path.join(BASE_DIR, 'data', 'evaluation', 'retrieval_queries.json')

def load_queries(path: str = QUERIES_PATH) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_retriever_on_queries(retriever, queries: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    """
    Runs test queries through a retriever and calculates retrieval metrics.
    """
    results = []
    hit_1_count = 0
    hit_3_count = 0
    hit_5_count = 0
    rr_sum = 0.0
    latencies = []

    for q in queries:
        qid = q['query_id']
        question = q['question']
        expected_ids = q.get('expected_papers', [])

        t0 = time.time()
        retrieved_docs = retriever.retrieve(question, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [doc['paper_id'] for doc in retrieved_docs]

        # Metric calculation
        first_match_rank = None
        for rank, pid in enumerate(retrieved_ids, 1):
            if pid in expected_ids:
                if first_match_rank is None:
                    first_match_rank = rank

        is_hit_1 = (first_match_rank == 1)
        is_hit_3 = (first_match_rank is not None and first_match_rank <= 3)
        is_hit_5 = (first_match_rank is not None and first_match_rank <= 5)
        rr = (1.0 / first_match_rank) if first_match_rank is not None else 0.0

        if is_hit_1:
            hit_1_count += 1
        if is_hit_3:
            hit_3_count += 1
        if is_hit_5:
            hit_5_count += 1
        rr_sum += rr

        results.append({
            "query_id": qid,
            "question": question,
            "category": q.get('category', ''),
            "expected_papers": expected_ids,
            "first_match_rank": first_match_rank,
            "is_hit_1": is_hit_1,
            "is_hit_3": is_hit_3,
            "is_hit_5": is_hit_5,
            "reciprocal_rank": round(rr, 4),
            "latency_ms": round(latency_ms, 2),
            "top_results": [
                {
                    "rank": d['rank'],
                    "paper_id": d['paper_id'],
                    "title": d['title'],
                    "page": d['page'],
                    "score": d.get('reranker_score', d.get('score', 0.0)),
                    "snippet": d['content'][:200].replace('\n', ' ')
                }
                for d in retrieved_docs
            ]
        })

    n = len(queries)
    metrics = {
        "retriever_method": retriever.method_name,
        "num_queries": n,
        "hit_at_1": round(hit_1_count / n, 4),
        "hit_at_3": round(hit_3_count / n, 4),
        "hit_at_5": round(hit_5_count / n, 4),
        "mrr": round(rr_sum / n, 4),
        "avg_latency_ms": round(sum(latencies) / n, 2),
        "query_results": results
    }
    return metrics

def run_all_experiments():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    queries = load_queries()
    print(f"Loaded {len(queries)} test queries from {QUERIES_PATH}")

    # 1. Initialize retrievers
    print("\n--- Initializing Retrievers ---")
    dense = DenseRetriever()
    mmr = MMRRetriever(lambda_mult=0.7)
    hybrid = HybridRetriever(rrf_k=60)
    reranked = RerankedRetriever(base_retriever=hybrid)

    retrievers = [
        ("dense", dense),
        ("mmr", mmr),
        ("hybrid", hybrid),
        ("reranked", reranked)
    ]

    all_metrics = []

    print("\n=== EXECUTING RETRIEVAL BENCHMARK ON 10 CANONICAL QUERIES ===")
    for key, r_instance in retrievers:
        print(f"\nEvaluating: {r_instance.method_name}...")
        metrics = evaluate_retriever_on_queries(r_instance, queries, top_k=5)
        all_metrics.append(metrics)

        # Save individual experiment result JSON
        json_path = os.path.join(EXPERIMENTS_DIR, f"{key}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Hit@1: {metrics['hit_at_1']:.1%} | Hit@3: {metrics['hit_at_3']:.1%} | Hit@5: {metrics['hit_at_5']:.1%} | MRR: {metrics['mrr']:.4f} | Latency: {metrics['avg_latency_ms']} ms")
        print(f"  Saved detailed results to {json_path}")

    # 2. Generate Comparison CSV
    comparison_csv_path = os.path.join(EXPERIMENTS_DIR, "comparison.csv")
    csv_rows = []
    for q_idx in range(len(queries)):
        qid = queries[q_idx]['query_id']
        question = queries[q_idx]['question']
        for m in all_metrics:
            q_res = m['query_results'][q_idx]
            top_hit = q_res['top_results'][0] if q_res['top_results'] else {}
            csv_rows.append({
                "query_id": qid,
                "question": question,
                "method": m['retriever_method'],
                "top_paper_id": top_hit.get('paper_id', 'N/A'),
                "top_paper_title": top_hit.get('title', 'N/A'),
                "top_page": top_hit.get('page', 'N/A'),
                "hit_at_1": 1 if q_res['is_hit_1'] else 0,
                "hit_at_3": 1 if q_res['is_hit_3'] else 0,
                "reciprocal_rank": q_res['reciprocal_rank'],
                "latency_ms": q_res['latency_ms']
            })

    with open(comparison_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_id", "question", "method", "top_paper_id", "top_paper_title",
            "top_page", "hit_at_1", "hit_at_3", "reciprocal_rank", "latency_ms"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nComparison CSV successfully generated at {comparison_csv_path}")

    # 3. Print Final Benchmark Summary Table
    print("\n" + "=" * 90)
    print("                    PHASE 2 RETRIEVAL BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Retrieval Method':<30} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Avg Latency (ms)':<15}")
    print("-" * 90)
    for m in all_metrics:
        print(f"{m['retriever_method']:<30} | {m['hit_at_1']:<8.1%} | {m['hit_at_3']:<8.1%} | {m['hit_at_5']:<8.1%} | {m['mrr']:<8.4f} | {m['avg_latency_ms']:<15.1f}")
    print("=" * 90)

    return all_metrics

if __name__ == '__main__':
    run_all_experiments()

