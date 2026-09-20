"""
Context Builder Module for Pinnacle Plus Capstone.
Formats retrieved passages with exact Paper Title and Page Number tags,
preparing clean structured context for LLM generation.
"""

from typing import List, Dict, Any

def format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Transforms retrieved chunks into a standardized context block for LLM prompts.
    """
    if not retrieved_docs:
        return "No relevant passages found in the knowledge base."

    context_blocks = []
    for rank, doc in enumerate(retrieved_docs, 1):
        title = doc.get("title", "Unknown Paper")
        page = doc.get("page", "N/A")
        paper_id = doc.get("paper_id", "")
        passage = doc.get("content", "").strip()

        block = (
            f"[SOURCE {rank}]\n"
            f"Paper: {title} (arXiv: {paper_id})\n"
            f"Page: {page}\n"
            f"Passage:\n{passage}\n"
        )
        context_blocks.append(block)

    return "\n" + "-" * 50 + "\n" + "\n".join(context_blocks) + "\n" + "-" * 50

def extract_top_sources(retrieved_docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Extracts the top-K supporting sources with Paper Title, Page Number, and excerpt.
    """
    top_sources = []
    for rank, doc in enumerate(retrieved_docs[:top_k], 1):
        top_sources.append({
            "rank": rank,
            "title": doc.get("title", "Unknown"),
            "page": doc.get("page", "N/A"),
            "paper_id": doc.get("paper_id", ""),
            "category": doc.get("category", ""),
            "score": doc.get("reranker_score", doc.get("score", 0.0)),
            "passage": doc.get("content", "").strip()
        })
    return top_sources

