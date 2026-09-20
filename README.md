# Pinnacle Plus — Research Paper Answer Bot

## 1. Problem Statement
Large Language Models (LLMs) frequently exhibit factual hallucination, lack provenance for specialized academic assertions, and cannot provide verified source citations for mathematical formulas or architectural mechanisms. When researchers consult lengthy academic publications (often 20 to 80 pages each), locating specific equations, hyperparameters, and experimental claims requires laborious manual search. Naive LLM question answering fails because parametric weights cannot cite exact page numbers or guarantee that assertions reflect the author's verbatim findings.

## 2. Objective
The objective of this project is to construct a production-grade, reproducible **Retrieval-Augmented Generation (RAG)** system over 15 landmark Generative AI research papers from arXiv. The system must:
1. Guarantee that **100% of generated responses are grounded strictly in retrieved peer-reviewed context**.
2. Return verifiable **Top-3 Supporting Sources**, each detailing the **Paper Title**, **Exact 1-Indexed Page Number**, and **Verbatim Passage**.
3. Enforce an active **Anti-Hallucination Guardrail** that deterministically refuses to answer out-of-domain or ungrounded queries.
4. Provide an intuitive, interactive **Streamlit web application** for real-time scientific inquiry.

## 3. Solution Overview
The system implements a two-stage retrieval pipeline coupled with a strictly instructed generative model:
- **Corpus Ingestion:** 15 seminal arXiv papers are parsed page-by-page, strictly maintaining 1-indexed page metadata.
- **Semantic Chunking:** Text is split using `RecursiveCharacterTextSplitter` into 2,247 chunks (1,000 characters, 150 overlap).
- **Dual Embeddings:** Open-source `sentence-transformers/all-MiniLM-L6-v2` (384d) is benchmarked against commercial `text-embedding-3-small` (1536d), selecting the open-source model for fast, local execution.
- **Vector Store:** ChromaDB persists chunk vectors using Cosine distance.
- **Two-Stage Retrieval:** Stage 1 combines Dense Vector Search and BM25Okapi keyword search via Reciprocal Rank Fusion (RRF, $k=60$) to retrieve the top-15 candidate passages; Stage 2 applies a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to rerank candidates down to the Top-3.
- **Grounded Generation:** **Google Gemini (`gemini-2.5-flash`)** synthesizes the answer strictly from retrieved passages, citing paper titles and page numbers.
- **Deployment:** A Streamlit interface (`app.py`) enables real-time query execution, paper catalog browsing, and citation verification.

## 4. Key Features
- **100% Grounded Synthesis:** Eliminates hallucinations by constraining LLM generation exclusively to retrieved scientific passages.
- **Verified Page Numbers:** Page numbers are preserved directly from PDF page indexes, not inferred or guessed.
- **Hybrid Search + Cross-Encoder:** Combines semantic recall (Dense) and lexical precision (BM25) with cross-attention reranking.
- **Anti-Hallucination Fallback:** Safely declines out-of-domain queries with a standard refusal message.
- **High Speed & Responsiveness:** End-to-end response times under 7 seconds powered by `gemini-2.5-flash`.
- **Reproducible Academic Notebook:** Complete 12-section Jupyter Notebook executed with zero errors.

## 5. Dataset
The knowledge base comprises **15 seminal Generative AI research papers** downloaded directly from arXiv.org spanning 2017 to 2023, cataloged in `data/metadata.csv`:

| # | Paper ID | Title | Category | Year | Pages | Chunks |
|---|---|---|---|---|---|---|
| 1 | `1706.03762` | Attention Is All You Need | Transformer Architecture | 2017 | 15 | 49 |
| 2 | `1810.04805` | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Language Representation | 2018 | 16 | 83 |
| 3 | `2005.14165` | Language Models are Few-Shot Learners (GPT-3) | Large Language Models | 2020 | 75 | 313 |
| 4 | `1908.10084` | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Embeddings & Search | 2019 | 11 | 55 |
| 5 | `2004.04906` | Dense Passage Retrieval for Open-Domain Question Answering | Information Retrieval | 2020 | 13 | 72 |
| 6 | `2005.11401` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | RAG Foundation | 2020 | 19 | 88 |
| 7 | `2107.13586` | Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing | Prompting Methods | 2021 | 46 | 448 |
| 8 | `2201.11903` | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Reasoning & CoT | 2022 | 43 | 173 |
| 9 | `2302.04761` | Toolformer: Language Models Can Teach Themselves to Use Tools | Tool Use & API Calling | 2023 | 17 | 91 |
| 10 | `2210.03629` | ReAct: Synergizing Reasoning and Acting in Language Models | AI Agents | 2022 | 33 | 140 |
| 11 | `2307.03172` | Lost in the Middle: How Language Models Use Long Contexts | Long-Context Behavior | 2023 | 18 | 80 |
| 12 | `2310.11511` | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Advanced RAG | 2023 | 30 | 133 |
| 13 | `2312.10997` | Retrieval-Augmented Generation for Large Language Models: A Survey | Modular RAG Survey | 2023 | 21 | 136 |
| 14 | `2309.15217` | RAGAS: Automated Evaluation of Retrieval Augmented Generation | RAG Evaluation | 2023 | 8 | 40 |
| 15 | `2311.05232` | A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions | Hallucination & Reliability | 2023 | 58 | 346 |
| **Total** | — | **15 Landmark Publications** | — | — | **423** | **2,247** |

### Paper Selection Concept
The corpus tracks the chronological evolution of Generative AI: starting from the fundamental attention mechanism (Vaswani et al.), progressing through pre-training (BERT, GPT-3), dense representations (SBERT, DPR), the introduction of RAG (Lewis et al.), reasoning paradigms (Chain-of-Thought, ReAct, Toolformer), context window dynamics (Lost in the Middle), self-reflective architectures (Self-RAG), and evaluation & hallucination mitigation (RAGAS, Survey on Hallucination).

## 6. System Architecture

```
                                  USER INTERFACE
                          [ Streamlit Web Application (app.py) ]
                                        │
                                        ▼
                             USER RESEARCH QUESTION
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: HYBRID CANDIDATE RETRIEVAL                                          │
│                                                                              │
│    Dense Vector Retrieval                         Sparse Lexical Retrieval   │
│    (ChromaDB + all-MiniLM-L6-v2)                  (BM25Okapi Token Index)    │
│    Cosine Similarity, 384 dims                    Exact Keyword Matching     │
│                 │                                            │               │
│                 └────────────────────┬───────────────────────┘               │
│                                      ▼                                       │
│                         Reciprocal Rank Fusion (RRF)                         │
│                           Score = Σ 1 / (60 + Rank)                          │
│                                      ▼                                       │
│                             Top-15 Candidates                                │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: CROSS-ENCODER RERANKING                                             │
│ Model: cross-encoder/ms-marco-MiniLM-L-6-v2                                  │
│ Full cross-attention query-passage interaction scoring                       │
│                                      ▼                                       │
│                             Top-3 Selected Chunks                            │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: CONTEXT ASSEMBLY & STRICT GROUNDED GENERATION                       │
│ Context Formatter attaches Paper Title + 1-Indexed PDF Page Number           │
│                                                                              │
│ Dual-Layer Anti-Hallucination Guardrails:                                    │
│  - Heuristic relevance floor & out-of-domain rejection                       │
│  - Strict system prompt: "Base answers ONLY on provided context"             │
│                                                                              │
│ LLM Engine: Google Gemini (gemini-2.5-flash) [Multi-tier Failover]           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
                             STRUCTURED OUTPUT
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 1. Grounded Answer / Refusal Notice                                    │
   │ 2. Top-3 Verifiable Citations (Paper Title, Exact Page, Passage Quote) │
   └────────────────────────────────────────────────────────────────────────┘
```

## 7. PDF Processing & Chunking
- **Page Extraction:** Implemented in `src/loader.py` using `pypdf`. Iterates page-by-page and stores `metadata['page'] = i + 1` (1-indexed) alongside `title`, `year`, `authors`, and `category`.
- **Chunking Algorithm:** LangChain `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " "]`.
- **Configuration:**
  - `chunk_size = 1000` characters (~150–200 words) — preserves complete mathematical statements and paragraph context.
  - `chunk_overlap = 150` characters (~15%) — prevents loss of semantic transitions at chunk boundaries.
- **Result:** 423 pages yielded exactly **2,247 semantic chunks** with an average length of 837.2 characters.

## 8. Embedding Experiments
Two distinct embedding architectures were implemented and compared in `src/embeddings.py`:

| Attribute | Model A: Open-Source (Selected) | Model B: Commercial |
|---|---|---|
| **Model Name** | `sentence-transformers/all-MiniLM-L6-v2` | `OpenAI text-embedding-3-small` |
| **Provider** | Hugging Face / SBERT | OpenAI |
| **Dimensions** | 384 dimensions | 1,536 dimensions |
| **Inference Mode** | Local CPU/GPU (Zero cost, offline) | Remote API ($0.02 / 1M tokens) |
| **Latency** | < 2 ms per chunk | 80–150 ms per chunk (network dependent) |
| **Selection** | **Primary Engine** (Fast, resilient, reproducible) | Benchmark alternative |

## 9. Vector Database
- **Engine:** ChromaDB (Persistent Disk Backend at `vectorstore/chroma_db/`).
- **Collection:** `papers_opensource`.
- **Distance Metric:** Cosine Distance:
  $$\text{Cosine Distance} = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
- **Chunk Identifier Scheme:** Deterministic UUIDs (`{paper_id}_p{page}_c{chunk_idx}`).
- **Capacity:** 2,247 indexed vectors. Loads instantly upon startup without re-indexing.

## 10. Retrieval Experiments
Four retrieval paradigms were implemented and evaluated across 10 canonical gold-standard research queries (`src/retrieval/`):

1. **Dense Cosine Retrieval (`dense_retriever.py`):** Pure vector dot-product in 384d space.
2. **Maximal Marginal Relevance (`mmr_retriever.py`):** Penalizes informational redundancy ($\lambda = 0.7$):
   $$\text{MMR} = \operatorname{argmax}_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$
3. **Hybrid Search (`hybrid_retriever.py`):** BM25Okapi sparse lexical search merged with Dense Cosine search via Reciprocal Rank Fusion ($k = 60$):
   $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
4. **Two-Stage Cross-Encoder Reranker (`reranker.py`):** Evaluates top-15 hybrid candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2` with full cross-attention.

### Quantitative Benchmark Results:
| Retrieval Strategy | Hit@1 | Hit@3 | Hit@5 | MRR | Avg Latency |
|---|---|---|---|---|---|
| **Dense Cosine** | 90.0% | 100.0% | 100.0% | 0.9500 | 36.2 ms |
| **MMR ($\lambda=0.7$)** | 90.0% | 100.0% | 100.0% | 0.9333 | 38.4 ms |
| **Hybrid Search (BM25 + Dense)** | 90.0% | 100.0% | 100.0% | 0.9500 | 44.1 ms |
| **Two-Stage Reranker** | 90.0% | 100.0% | 100.0% | 0.9500 | 694.5 ms |

## 11. Final Retrieval Strategy
The production retrieval strategy is a **Two-Stage Pipeline**:
- **Stage 1 (Recall):** Hybrid Search (Dense + BM25Okapi via RRF) retrieves the top-15 candidate chunks in ~44ms, combining semantic understanding with exact lexical matching for acronyms and mathematical notation.
- **Stage 2 (Precision):** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) performs deep cross-attention reranking over the 15 candidates to extract the Top-3 supporting passages with the highest sentence-level relevance.

## 12. RAG Pipeline
The complete pipeline (`src/rag/rag_chain.py`) executes:
1. `User Question` received.
2. `Two-Stage Retrieval` fetches the Top-3 passages.
3. `Context Formatter` prepares the prompt injecting Paper Title and 1-Indexed Page Number.
4. `LLMClient` executes generation under strict system prompt constraints.
5. `Response Formatter` structures the grounded answer alongside Top-3 citation cards.

## 13. LLM
- **Primary Generator:** **Google Gemini (`gemini-2.5-flash`)** via `GEMINI_API_KEY`.
- **Generation Speed:** Sub-7 seconds end-to-end.
- **Multi-Tier Resilient Failover:**
  - *Tier 1:* Google Gemini (`gemini-2.5-flash`)
  - *Tier 2:* OpenAI (`gpt-4o-mini`)
  - *Tier 3:* Hugging Face Serverless (`Qwen/Qwen2.5-72B-Instruct`) via `HF_TOKEN`
  - *Tier 4:* Local Grounded Synthesizer (fully offline deterministic fallback)

## 14. Source Citation
Every generated answer returns the **Top-3 Supporting Sources** containing:
- **Paper Title:** Canonical published paper title (e.g. *"Attention Is All You Need"*).
- **Exact Page Number:** Verified 1-indexed PDF page where the passage appears (e.g. `Page 4`).
- **Supporting Passage:** Direct verbatim quotation from the original PDF text.
- **Relevance Score:** Cross-encoder interaction score or RRF rank score.

## 15. Evaluation
The end-to-end system was evaluated against 10 canonical test queries (`data/evaluation/rag_evaluation_queries.json`):
- **Retrieval Relevance:** 100% (10/10 true papers retrieved in Top-3).
- **Groundedness Score:** 100% (All claims backed by retrieved context).
- **Citation & Page Accuracy:** 100% (Exact title and 1-indexed page matches).
- **Anti-Hallucination Fallback:** 100% (Out-of-domain CRISPR query safely refused).

## 16. Streamlit Application
A full-featured web application is implemented in `app.py`:
- **Knowledge Base Stats:** Live counters (15 papers, 423 pages, 2,247 chunks).
- **Paper Catalog Browser:** Expandable list of all 15 papers with direct arXiv links.
- **Sample Query Dropdown:** 10 pre-loaded research questions for one-click testing.
- **Custom Question Input:** Free-form text input box for user queries.
- **Grounded Answer Display:** Color-coded answer box with execution latency and model attribution.
- **Visual Citation Cards:** Formatted cards displaying Rank, Title, Page Badge (`📄 Page X`), Relevance Score, and quoted passage.

## 17. Project Structure
```
pinnacle capstone project/
├── app.py                          # Streamlit Web Application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Git exclusion rules
├── README.md                       # Comprehensive documentation
├── data/
│   ├── papers/                     # 15 research paper PDFs from arXiv
│   ├── metadata.csv                # Paper catalog (Title, Year, Pages, arXiv ID)
│   └── evaluation/
│       ├── retrieval_queries.json  # 10 retrieval benchmark queries
│       └── rag_evaluation_queries.json # 10 end-to-end RAG evaluation queries
├── docs/
│   ├── presentation_slides.md      # 14 academic presentation slides
│   ├── demo_script.md              # 5-minute live demonstration script
│   └── viva_qa.md                  # Comprehensive viva defense Q&A guide
├── experiments/
│   ├── retrieval/                  # Retrieval benchmark results (JSON & CSV)
│   └── rag/                        # RAG evaluation benchmark results (CSV)
├── notebooks/
│   └── pinnacle_plus_capstone.ipynb # 12-section audited reproducible notebook
├── scripts/
│   ├── build_notebook.py           # Generates the 12-section notebook
│   ├── execute_notebook.py         # Executes and validates the notebook
│   ├── verify_phase2.py            # Retrieval verification script
│   └── verify_all_phases.py        # Complete end-to-end test suite
├── src/
│   ├── __init__.py
│   ├── dataset_builder.py          # Paper downloader & metadata validator
│   ├── loader.py                   # PDF extractor & text chunker
│   ├── embeddings.py               # Dual embedding interfaces
│   ├── vectorstore.py              # Persistent ChromaDB manager
│   ├── retrieval/
│   │   ├── dense_retriever.py      # Cosine vector retriever
│   │   ├── mmr_retriever.py        # Maximal Marginal Relevance retriever
│   │   ├── hybrid_retriever.py     # BM25Okapi + Dense RRF retriever
│   │   ├── reranker.py             # Cross-Encoder reranker
│   │   └── evaluation.py           # Retrieval benchmark runner
│   └── rag/
│       ├── prompt.py               # Grounded system and user prompts
│       ├── context_builder.py      # Context formulation with page attribution
│       ├── llm.py                  # Multi-tier LLM client (Gemini / OpenAI / HF / Local)
│       ├── response_formatter.py   # Citation formatter
│       ├── rag_chain.py            # End-to-end RAG pipeline coordinator
│       └── evaluation.py           # End-to-end benchmark evaluator
└── vectorstore/
    └── chroma_db/                  # Persistent ChromaDB index (2,247 chunks)
```

## 18. Installation
```bash
# 1. Clone repository and navigate to root
git clone <repo-url>
cd "pinnacle capstone project"

# 2. Install required Python packages
pip install -r requirements.txt
```

## 19. Environment Variables
Create a `.env` file from the provided template:
```bash
cp .env.example .env
```
Populate `.env` with your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```
> **Note:** A free Gemini API key can be obtained instantly from [Google AI Studio](https://aistudio.google.com/app/apikey). If no API key is provided, the system automatically falls back to the **Local Grounded Engine**.

## 20. How to Run

### Run the Streamlit Application:
```bash
python -m streamlit run app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

### Run the Reproducible Jupyter Notebook:
```bash
python scripts/execute_notebook.py
```
Or open in JupyterLab:
```bash
jupyter lab notebooks/pinnacle_plus_capstone.ipynb
```

### Run the Automated System Verification:
```bash
python scripts/verify_all_phases.py
```

## 21. Example Questions
1. *"What is the key difference between scaled dot-product attention and multi-head attention in Transformer architecture?"*
   - **Target Paper:** *Attention Is All You Need* (Page 4)
2. *"How does BERT use Masked Language Modeling and Next Sentence Prediction for pre-training?"*
   - **Target Paper:** *BERT* (Pages 3–4)
3. *"How does Retrieval-Augmented Generation (RAG) combine parametric and non-parametric memory?"*
   - **Target Paper:** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Page 2)
4. *"How does Toolformer decide when and where to insert external API calls into generated text?"*
   - **Target Paper:** *Toolformer* (Pages 2–3)
5. *"How does CRISPR-Cas9 perform targeted gene cleavage in mammalian genomic editing?"*
   - **Expected Behavior:** Triggers anti-hallucination guardrail: *"The provided research papers do not contain sufficient information to answer this question."*

## 22. Limitations
- **Multi-Modal Content:** Tables, architectural diagrams, and mathematical plots embedded in PDFs as raster images are not parsed by pure text extraction (`pypdf`).
- **Cross-Encoder Latency:** Deep cross-attention reranking requires ~650ms on CPU, adding slight latency over pure vector search.
- **Cross-Document Multi-Hop Reasoning:** Complex queries requiring synthesis across 4+ disparate papers are bounded by the top-3 citation context window.

## 23. Future Improvements
- **Multi-Modal Document Parsing:** Integrate vision-language models (e.g. ColPali or Nougat) to embed architectural figures and performance tables.
- **ColBERT Late-Interaction:** Implement token-level late interaction for sub-millisecond reranking.
- **Graph RAG:** Build knowledge graphs linking citations and author cross-references across papers for complex multi-hop synthesis.
