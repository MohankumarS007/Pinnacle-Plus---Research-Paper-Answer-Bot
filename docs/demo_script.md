# Pinnacle Plus — Research Paper Answer Bot
## Official Live Demonstration Script & Mentor Walkthrough Guide
**Audience:** Industrial Mentor, Academic Jury, Technical Reviewers  
**Target Duration:** 5–7 minutes  
**Format:** Live Streamlit Application Demo (`app.py`) + Code/Notebook Inspection  

---

### Phase 0: Pre-Demo Preparation Checklist (T-Minus 5 Minutes)
1. Open PowerShell terminal in the workspace directory:
   ```bash
   cd "c:\Users\shanm\Documents\pinnacle capstone project"
   ```
2. Verify application is running:
   ```bash
   python -m streamlit run app.py --server.port 8501
   ```
3. Open browser at `http://localhost:8501`.
4. Verify the sidebar loads knowledge base stats:
   - Curated Papers: **15**
   - Extracted Pages: **423**
   - Indexed Chunks: **2,247**
   - Primary Generator: **Google Gemini (`gemini-2.5-flash`)**

---

### Step 1: Introduction & Problem Context (1:00 min)
**Spoken Script:**
> *"Welcome to the demonstration of our Capstone Project: **Pinnacle Plus — Research Paper Answer Bot**.*  
>  
> *In scientific research, Large Language Models often suffer from hallucinations, lack temporal grounding, and cannot point to exact pages for complex technical equations or algorithmic definitions.*  
>  
> *To solve this, we engineered an end-to-end, production-grade Retrieval-Augmented Generation (RAG) system grounded exclusively over 15 landmark Generative AI research papers from arXiv—spanning Transformers, BERT, GPT-3, SBERT, DPR, RAG, Prompting Survey, Chain-of-Thought, Toolformer, ReAct, Lost in the Middle, Self-RAG, RAG Survey, RAGAS, and Hallucination in LLMs.*  
>  
> *Our system guarantees two core properties:*  
> *1. Every answer provides the **Top-3 Supporting Sources** with verified **Paper Title**, **1-Indexed Page Number**, and exact **Passage Quote**.*  
> *2. An active **Anti-Hallucination Guardrail** that detects ungrounded queries and declines to answer rather than fabricating facts."*

---

### Step 2: Architecture & Knowledge Base Walkthrough (1:00 min)
**Action:** Point to the Streamlit left sidebar.  
**Spoken Script:**
> *"Looking at the sidebar on the left, you can see our fully ingested knowledge base:*  
> *- 15 peer-reviewed papers downloaded from arXiv.org.*  
> *- 423 total pages parsed with PyPDF, retaining exact 1-indexed page tracking.*  
> *- 2,247 semantic chunks indexed using LangChain's `RecursiveCharacterTextSplitter` (1,000 characters with 150 character overlap).*  
> *- All chunks are embedded into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` and indexed in a persistent ChromaDB vector store.*  
> *- Our LLM generation engine is powered by **Google Gemini (`gemini-2.5-flash`)**, providing sub-7-second generation with strict factual grounding.*  
>  
> *Furthermore, clicking 'Browse 15 Indexed Papers' reveals the complete catalog with direct links to the official arXiv preprints."*

---

### Step 3: Factual Research Query & Retrieval Demonstration (1:30 min)
**Action:** In the Streamlit UI, select the sample question:
> *"What is the key difference between scaled dot-product attention and multi-head attention in Transformer architecture?"*
Click **Ask Research Bot**.

**Spoken Script:**
> *"Let us submit a foundational question regarding the Transformer architecture from Vaswani et al. (2017).*  
>  
> *Watch the pipeline execute:*  
> *1. **Stage 1 (Hybrid Retrieval):** The query is dispatched simultaneously to ChromaDB (Dense Cosine) and BM25Okapi (Lexical Keyword Matching). The results are fused using Reciprocal Rank Fusion ($k=60$) to obtain the top-15 candidates.*  
> *2. **Stage 2 (Cross-Attention Reranking):** The top-15 candidates are evaluated by `cross-encoder/ms-marco-MiniLM-L-6-v2`, computing deep cross-attention interaction logits.*  
> *3. **Stage 3 (Grounded Synthesis):** The top-3 passages are fed into Google Gemini under strict grounding constraints."*

**Action:** Point to the generated Grounded Answer box.  
**Spoken Script:**
> *"As you can see, Gemini generates a precise answer explaining that Scaled Dot-Product Attention computes a single attention function scaled by $\frac{1}{\sqrt{d_k}}$, whereas Multi-Head Attention linearly projects queries, keys, and values $h$ times in parallel, allowing the model to attend to information at different representation subspaces simultaneously [citing Page 4]."*

---

### Step 4: Source Verification & Exact Page Number Verification (1:00 min)
**Action:** Scroll down to the **Top-3 Supporting Citations** cards.  
**Spoken Script:**
> *"Notice the rigorous citation structure required by the Pinnacle Plus rubric:*  
> *- **Citation 1:** Confirms the paper 'Attention Is All You Need' at **Page 4** with a high relevance score.*  
> *- **Citation 2:** Also points to 'Attention Is All You Need' at **Page 4**, highlighting the mathematical equations.*  
> *- Below each citation is the verbatim blockquote extracted directly from the paper's PDF page.*  
>  
> *Any researcher or peer reviewer can open the original PDF to Page 4 and verify that the exact text, equations, and assertions match without deviation."*

---

### Step 5: Anti-Hallucination Guardrail Demonstration (1:30 min)
**Action:** Select the sample query:
> *"How does CRISPR-Cas9 perform targeted gene cleavage in mammalian genomic editing?"*
Click **Ask Research Bot**.

**Spoken Script:**
> *"Now, let us rigorously test the system's anti-hallucination guardrail.*  
>  
> *CRISPR-Cas9 is a biological biotechnology topic, but it does NOT exist anywhere in our 15 Generative AI papers.*  
>  
> *A standard ungrounded LLM will willingly answer based on pre-training weights, violating the RAG grounding premise.*  
>  
> *Notice what our system does: The retrieval relevance and keyword overlap check flags that the context lacks supporting evidence. The system immediately outputs the fallback response:*  
>  
> **'The provided research papers do not contain sufficient information to answer this question.'**  
>  
> *This confirms that the bot is 100% safeguarded against ungrounded hallucinations."*

---

### Step 6: Retrieval Optimization & Benchmark Presentation (1:00 min)
**Action:** Switch briefly to the Jupyter Notebook (`pinnacle_plus_capstone.ipynb`) or Presentation Slide 8.  
**Spoken Script:**
> *"We conducted a rigorous comparative benchmark across 4 distinct retrieval paradigms over 10 canonical queries:*  
> *- **Dense Cosine Retrieval:** Achieved 90% Hit@1, 100% Hit@3, and 0.9500 MRR in 36ms.*  
> *- **Maximal Marginal Relevance (MMR, $\lambda=0.7$):** Balanced redundancy and diversity with 100% Hit@3.*  
> *- **Hybrid Search (BM25 + Dense via RRF):** Handled both semantic concepts and exact acronyms, hitting 100% Hit@3 in 44ms.*  
> *- **Two-Stage Reranker:** Achieved maximum precision through cross-attention.*  
>  
> *For our production pipeline, we selected Hybrid Search with Cross-Encoder Reranking, delivering the optimal balance of recall, precision, and verifiable attribution."*

---

### Step 7: Conclusion & Q&A Transition (0:30 min)
**Spoken Script:**
> *"In summary, the Pinnacle Plus Research Paper Answer Bot fulfills all capstone requirements:*  
> *- 15 Curated Papers and 2,247 chunks indexed in persistent ChromaDB.*  
> *- Four retrieval strategies evaluated and benchmarked.*  
> *- 100% grounded answers with exact Top-3 Paper Titles and 1-indexed Page Numbers.*  
> *- Google Gemini integration with sub-7s latency.*  
> *- A functional Streamlit UI and fully executed Jupyter Notebook.*  
>  
> *Thank you. I am now pleased to take your questions."*
