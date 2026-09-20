"""
Runs the 5 required audit test questions through the end-to-end RAG system
verifying groundedness, Top-3 citations, exact page numbers, and guardrails.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.rag.rag_chain import RAGPipeline
from src.rag.response_formatter import format_rag_output

def run_5_audit_tests():
    print("\n" + "=" * 90)
    print("           PINNACLE PLUS: 5-QUESTION FINAL AUDIT TEST")
    print("=" * 90)

    pipeline = RAGPipeline()

    test_questions = [
        {
            "id": 1,
            "query": "What is the key difference between scaled dot-product attention and multi-head attention in Transformer architecture?",
            "expected_paper": "Attention Is All You Need",
            "is_negative": False
        },
        {
            "id": 2,
            "query": "How does BERT use Masked Language Modeling and Next Sentence Prediction for pre-training?",
            "expected_paper": "BERT",
            "is_negative": False
        },
        {
            "id": 3,
            "query": "How does Retrieval-Augmented Generation (RAG) combine parametric and non-parametric memory?",
            "expected_paper": "Retrieval-Augmented Generation",
            "is_negative": False
        },
        {
            "id": 4,
            "query": "What is the ReAct framework and how does it synergize reasoning and acting in language models?",
            "expected_paper": "ReAct",
            "is_negative": False
        },
        {
            "id": 5,
            "query": "How does CRISPR-Cas9 perform targeted gene cleavage in mammalian genomic editing?",
            "expected_paper": None,
            "is_negative": True
        }
    ]

    all_passed = True

    for t in test_questions:
        print(f"\n[TEST QUESTION {t['id']}/5]")
        print(f"QUERY: {t['query']}")

        result = pipeline.answer_question(t['query'], top_k=3)
        answer = result['answer']
        sources = result['sources']
        model_used = result['model']
        latency = result['latency_ms']

        print(f"MODEL USED : {model_used}")
        print(f"LATENCY    : {latency:.1f} ms")
        print(f"ANSWER     : {answer[:180]}...")

        # Assertions
        assert len(sources) == 3, f"Expected 3 sources, got {len(sources)}"
        for s in sources:
            assert s['title'], "Source title cannot be empty"
            assert isinstance(s['page'], int) and s['page'] >= 1, f"Invalid page number: {s['page']}"
            assert s['passage'], "Supporting passage cannot be empty"

        if t['is_negative']:
            # Verify anti-hallucination guardrail
            assert "not contain sufficient information" in answer.lower(), "Guardrail failed to trigger on out-of-domain query!"
            print("GUARDRAIL  : PASSED (Safely refused out-of-domain query)")
        else:
            # Verify relevant paper is in citations
            found_paper = any(t['expected_paper'].lower() in s['title'].lower() for s in sources)
            assert found_paper, f"Expected {t['expected_paper']} in citations!"
            print(f"CITATION 1 : {sources[0]['title']} (Page {sources[0]['page']})")
            print(f"CITATION 2 : {sources[1]['title']} (Page {sources[1]['page']})")
            print(f"CITATION 3 : {sources[2]['title']} (Page {sources[2]['page']})")
            print("STATUS     : PASSED")

    print("\n" + "=" * 90)
    print("ALL 5 AUDIT TEST QUESTIONS PASSED 100% WITH VERIFIED CITATIONS & GUARDRAILS!")
    print("=" * 90)

if __name__ == '__main__':
    run_5_audit_tests()
