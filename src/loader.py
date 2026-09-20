"""
PDF Document Loader and Chunking Engine for Pinnacle Plus Capstone.
Loads 15 curated research papers, extracts text while strictly preserving
page-level metadata, and chunks text using RecursiveCharacterTextSplitter.
"""

import os
import sys
import csv
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PAPERS_DIR = os.path.join(DATA_DIR, 'papers')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')

def load_documents_with_metadata(metadata_path: str = METADATA_PATH, papers_dir: str = PAPERS_DIR) -> List[Document]:
    """
    Loads all PDFs listed in metadata.csv and extracts text page by page.
    Preserves all paper metadata and exact 1-indexed page number on each Document.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    documents: List[Document] = []
    print(f"=== STEP 5: Loading {len(reader)} PDFs from {papers_dir} ===")

    for idx, row in enumerate(reader, 1):
        pdf_filename = row['file_name']
        pdf_path = os.path.join(papers_dir, pdf_filename)
        
        if not os.path.exists(pdf_path):
            print(f"  [Warning] Missing PDF file: {pdf_filename}")
            continue

        try:
            reader_obj = PdfReader(pdf_path)
            num_pages = len(reader_obj.pages)
            extracted_pages = 0

            for page_idx, page in enumerate(reader_obj.pages):
                page_text = page.extract_text() or ""
                # Normalize line breaks and clean whitespace
                page_text = page_text.strip()
                if not page_text:
                    continue  # Skip blank pages (e.g. blank cover/trailing pages)

                # Construct robust metadata dictionary
                metadata: Dict[str, Any] = {
                    "paper_id": row['paper_id'],
                    "title": row['title'],
                    "category": row['category'],
                    "year": int(row['year']),
                    "authors": row['authors'],
                    "source": row['source'],
                    "file_name": pdf_filename,
                    "page": page_idx + 1  # 1-indexed page number
                }

                doc = Document(page_content=page_text, metadata=metadata)
                documents.append(doc)
                extracted_pages += 1

            print(f"  [{idx}/{len(reader)}] Loaded '{row['title']}' — {extracted_pages}/{num_pages} pages extracted")

        except Exception as e:
            print(f"  [Error loading {pdf_filename}]: {e}")

    print(f"\nTotal page documents extracted: {len(documents)}")
    return documents

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits page-level documents into semantic chunks using RecursiveCharacterTextSplitter.
    Ensures every generated chunk retains complete metadata including paper title and page number.
    """
    print(f"\n=== STEP 6: Chunking Documents (chunk_size={chunk_size}, overlap={chunk_overlap}) ===")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False
    )

    chunks = splitter.split_documents(documents)
    
    # Add chunk_id to metadata for deterministic referencing
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Total chunks generated: {len(chunks)}")
    avg_len = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
    print(f"Average chunk character length: {avg_len:.1f}")

    return chunks

def inspect_sample_chunk(chunks: List[Document], index: int = 0):
    if not chunks or index >= len(chunks):
        print("No chunks available to inspect.")
        return
    
    sample = chunks[index]
    print("\n--- SAMPLE CHUNK INSPECTION ---")
    print(f"Paper Title : {sample.metadata.get('title')}")
    print(f"Category    : {sample.metadata.get('category')}")
    print(f"Page Number : {sample.metadata.get('page')}")
    print(f"arXiv ID    : {sample.metadata.get('paper_id')}")
    print(f"File Name   : {sample.metadata.get('file_name')}")
    print(f"Chunk Length: {len(sample.page_content)} characters")
    print(f"Content Preview:\n{sample.page_content[:300]}...")
    print("--------------------------------")

if __name__ == '__main__':
    docs = load_documents_with_metadata()
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=150)
    inspect_sample_chunk(chunks, index=10)

