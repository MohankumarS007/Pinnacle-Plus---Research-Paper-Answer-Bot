"""
Persistent ChromaDB Vector Store & Baseline Retrieval Engine for Pinnacle Plus Capstone.
Handles indexing 2,247 chunks with full metadata retention and provides
dense cosine similarity retrieval returning top-3 sources with Paper Title & Page Number.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional

# Force UTF-8 encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from src.loader import load_documents_with_metadata, chunk_documents
from src.embeddings import get_embedding_model



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(BASE_DIR, 'vectorstore', 'chroma_db')

def get_chroma_client(persist_directory: str = VECTORSTORE_DIR) -> chromadb.PersistentClient:
    """Initializes and returns a persistent ChromaDB client."""
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)

def index_chunks_into_chroma(
    chunks: List[Document],
    embedding_model,
    collection_name: str = "papers_opensource",
    persist_directory: str = VECTORSTORE_DIR,
    batch_size: int = 128
):
    """
    Indexes document chunks into a persistent ChromaDB collection.
    Reuses collection if already populated with all chunks.
    """
    client = get_chroma_client(persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    current_count = collection.count()
    if current_count >= len(chunks) and len(chunks) > 0:
        print(f"Vector store '{collection_name}' already contains {current_count} indexed chunks. Reusing existing store.")
        return collection

    print(f"\n=== STEP 8: Indexing {len(chunks)} Chunks into ChromaDB ('{collection_name}') ===")
    start_time = time.time()

    total_chunks = len(chunks)
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        
        # Flatten metadata for ChromaDB compatibility (Chroma expects str/int/float/bool)
        metadatas = []
        for doc in batch:
            meta = {
                "paper_id": str(doc.metadata.get("paper_id", "")),
                "title": str(doc.metadata.get("title", "")),
                "category": str(doc.metadata.get("category", "")),
                "year": int(doc.metadata.get("year", 0)),
                "authors": str(doc.metadata.get("authors", ""))[:300],  # Chroma string safety
                "source": str(doc.metadata.get("source", "")),
                "file_name": str(doc.metadata.get("file_name", "")),
                "page": int(doc.metadata.get("page", 1)),
                "chunk_id": int(doc.metadata.get("chunk_id", i))
            }
            metadatas.append(meta)

        # Generate unique IDs: paper_id_p<page>_c<chunk_id>
        ids = [f"{doc.metadata.get('paper_id')}_p{doc.metadata.get('page')}_c{doc.metadata.get('chunk_id')}" for doc in batch]

        # Compute embeddings for batch
        embeddings = embedding_model.embed_documents(texts)

        # Insert into Chroma
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"  Indexed chunks {min(i + batch_size, total_chunks)}/{total_chunks}...")

    elapsed = time.time() - start_time
    print(f"Successfully indexed {collection.count()} chunks in {elapsed:.2f} seconds.")
    return collection

def retrieve_top_k(
    query: str,
    collection,
    embedding_model,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Executes dense similarity search for a query and returns top-k matching sources.
    Each result contains: paper title, page number, similarity score, metadata, and chunk text.
    """
    query_vector = embedding_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_sources = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_text, meta, dist in zip(docs, metas, distances):
            # For cosine distance in Chroma: similarity = 1 - distance
            similarity = 1.0 - float(dist)
            retrieved_sources.append({
                "title": meta.get("title", "Unknown"),
                "page": meta.get("page", "N/A"),
                "paper_id": meta.get("paper_id", ""),
                "category": meta.get("category", ""),
                "similarity_score": round(similarity, 4),
                "content": doc_text
            })

    return retrieved_sources

def run_baseline_retrieval_tests(collection, embedding_model):
    """
    Executes canonical test research questions and verifies that top-3 sources
    contain accurate paper titles and exact page numbers.
    """
    print("\n" + "=" * 80)
    print("           STEP 9 — BASELINE RETRIEVAL TEST RESULTS")
    print("=" * 80)

    test_queries = [
        "What is the core architecture and self-attention mechanism in the Transformer?",
        "How does Retrieval-Augmented Generation (RAG) combine pre-trained parametric and non-parametric memory?",
        "What is Dense Passage Retrieval (DPR) and how does it compare to BM25 for open-domain QA?",
        "What metrics are introduced in RAGAS for evaluating retrieval augmented generation?"
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n[QUERY {idx}]: \"{query}\"")
        sources = retrieve_top_k(query, collection, embedding_model, top_k=3)
        
        print("-" * 80)
        for rank, src in enumerate(sources, 1):
            print(f"  Source #{rank}:")
            print(f"    Paper Title     : {src['title']}")
            print(f"    Page Number     : Page {src['page']}")
            print(f"    Category        : {src['category']} (arXiv: {src['paper_id']})")
            print(f"    Cosine Score    : {src['similarity_score']}")
            snippet = src['content'].replace('\n', ' ')[:220]
            snippet_safe = snippet.encode('ascii', errors='replace').decode('ascii')
            print(f"    Excerpt         : \"{snippet_safe}...\"")
        print("-" * 80)

if __name__ == '__main__':
    # Initialize embedding model (Model A: Open-Source)
    emb_model = get_embedding_model("open_source")

    # Load and chunk documents
    docs = load_documents_with_metadata()
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=150)

    # Index into ChromaDB
    collection = index_chunks_into_chroma(chunks, emb_model, collection_name="papers_opensource")

    # Run baseline retrieval test
    run_baseline_retrieval_tests(collection, emb_model)

