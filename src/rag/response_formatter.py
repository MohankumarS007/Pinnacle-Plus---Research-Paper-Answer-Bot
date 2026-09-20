"""
Response Formatter for Pinnacle Plus Capstone.
Formats the final answer along with the Top-3 supporting source citations
including Paper Title, Page Number, and supporting passage excerpts.
"""

from typing import Dict, Any, List

def format_rag_output(rag_result: Dict[str, Any]) -> str:
    """
    Renders user-facing formatted text showing Question, Answer, and Top-3 Supporting Sources.
    """
    question = rag_result.get("question", "")
    answer = rag_result.get("answer", "")
    sources = rag_result.get("sources", [])
    model_name = rag_result.get("model", "N/A")
    latency = rag_result.get("latency_ms", 0.0)

    lines = []
    lines.append("=" * 80)
    lines.append(f"QUESTION: {question}")
    lines.append("=" * 80)
    lines.append(f"\nANSWER ({model_name}, {latency:.1f}ms):")
    lines.append(answer)
    lines.append("\n" + "-" * 80)
    lines.append("TOP-3 SUPPORTING SOURCES (Citations):")
    lines.append("-" * 80)

    if not sources:
        lines.append("  No supporting sources available.")
    else:
        for idx, src in enumerate(sources, 1):
            title = src.get("title", "Unknown")
            page = src.get("page", "N/A")
            paper_id = src.get("paper_id", "")
            score = src.get("score", 0.0)
            passage = src.get("passage", "").replace("\n", " ")[:200]

            lines.append(f"\n[{idx}] {title}")
            lines.append(f"    Page Number : Page {page}")
            lines.append(f"    Paper ID    : arXiv:{paper_id} (Score: {score})")
            lines.append(f"    Passage     : \"{passage}...\"")

    lines.append("=" * 80)
    return "\n".join(lines)

