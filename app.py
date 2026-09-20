"""
Pinnacle Plus — Research Paper Answer Bot
Streamlit Web Application (Capstone Stretch Goal)
Provides an interactive, publication-grade user interface for scientific QA
with grounded responses, verifiable citations, and exact page numbers.
"""

import os
import sys
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment configuration (.env)
load_dotenv()

# Force UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.rag.rag_chain import RAGPipeline
from src.vectorstore import get_chroma_client, VECTORSTORE_DIR

# Set page configuration
st.set_page_config(
    page_title="Pinnacle Plus — Research Paper Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-page {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .badge-score {
        background-color: #DCFCE7;
        color: #166534;
    }
    .badge-id {
        background-color: #F3E8FF;
        color: #6B21A8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading RAG Pipeline & Reranking Models...")
def load_pipeline():
    return RAGPipeline()

@st.cache_data
def load_metadata():
    meta_path = os.path.join(BASE_DIR, 'data', 'metadata.csv')
    if os.path.exists(meta_path):
        return pd.read_csv(meta_path)
    return pd.DataFrame()

# Header
st.markdown('<div class="main-header">📚 Pinnacle Plus — Research Paper Answer Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Grounded AI Question Answering with Verifiable Citations & Exact Page Numbers across 15 Seminal GenAI Papers.</div>', unsafe_allow_html=True)

# Load pipeline and metadata
try:
    pipeline = load_pipeline()
    metadata_df = load_metadata()
except Exception as e:
    st.error(f"Error loading system components: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📊 Knowledge Base")
    st.metric(label="Curated Papers", value="15")
    st.metric(label="Extracted Pages", value="423")
    st.metric(label="Indexed Chunks", value="2,247")

    st.markdown("---")
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Number of Citations (Top-K)", min_value=1, max_value=5, value=3)
    initial_candidates = st.slider("Initial Candidates (Stage 1)", min_value=5, max_value=30, value=15)

    st.markdown("---")
    st.subheader("🤖 Engine Details")
    st.markdown(f"**Retriever:** Hybrid (Dense + BM25) + Cross-Encoder")
    st.markdown(f"**Embedding:** `all-MiniLM-L6-v2` (384 dims)")
    st.markdown(f"**Generator:** `{getattr(pipeline.llm, 'last_provider', pipeline.llm.mode)}`")

    st.markdown("---")
    with st.expander("📖 Browse 15 Indexed Papers"):
        if not metadata_df.empty:
            for _, row in metadata_df.iterrows():
                st.markdown(f"- **{row['title']}** ({row['year']})  \n  *Category:* {row['category']}  \n  [arXiv:{row['paper_id']}](https://arxiv.org/abs/{row['paper_id']})")

# Sample questions selector
SAMPLE_QUESTIONS = [
    "-- Select a sample question or type your own below --",
    "What is the key difference between scaled dot-product attention and multi-head attention in Transformer architecture?",
    "How does BERT use Masked Language Modeling and Next Sentence Prediction for pre-training?",
    "How does GPT-3 perform few-shot learning through in-context prompting without updating model weights?",
    "Why does Sentence-BERT use a Siamese network architecture with pooling instead of standard cross-encoders?",
    "How does Dense Passage Retrieval (DPR) use dual encoders to outperform traditional BM25 in open-domain QA?",
    "How does Retrieval-Augmented Generation (RAG) combine parametric and non-parametric memory?",
    "Can breaking down complex word math problems into step-by-step thinking traces improve LLM accuracy?",
    "How does Toolformer decide when and where to insert external API calls into generated text?",
    "What is the ReAct loop and how does it combine reasoning traces with action execution?",
    "How does CRISPR-Cas9 perform targeted gene cleavage in mammalian genomic editing? [Anti-Hallucination Test]"
]

selected_sample = st.selectbox("💡 Quick-Test with Curated Research Questions:", SAMPLE_QUESTIONS)

default_query = ""
if selected_sample and not selected_sample.startswith("--"):
    # Strip tag if anti-hallucination test
    default_query = selected_sample.replace(" [Anti-Hallucination Test]", "")

# User input
user_query = st.text_input("Enter your research question:", value=default_query, placeholder="e.g. What is the Transformer self-attention mechanism?")

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("🔍 Ask Research Bot", type="primary", use_container_width=True)

if ask_button:
    if not user_query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        with st.spinner("Searching 2,247 chunks across 15 papers, executing cross-attention reranking, and generating grounded response..."):
            t0 = time.time()
            try:
                result = pipeline.answer_question(
                    question=user_query,
                    top_k=top_k,
                    initial_candidates=initial_candidates
                )
                latency = time.time() - t0

                answer = result.get("answer", "")
                sources = result.get("sources", [])

                st.markdown("### 📝 Grounded Answer")
                # Check for anti-hallucination guardrail trigger
                if "not contain sufficient information" in answer.lower():
                    st.info(f"🛡️ **Anti-Hallucination Guardrail Active:**\n\n{answer}")
                else:
                    st.success(answer)

                st.caption(f"⏱️ End-to-end response generated in {latency:.2f}s | Engine: {result.get('model')} | Retriever: {result.get('retrieval_method')}")

                st.markdown("---")
                st.markdown(f"### 📑 Top-{len(sources)} Supporting Citations (Proven Evidence)")

                if not sources:
                    st.write("No supporting passages found.")
                else:
                    for s in sources:
                        rank = s.get("rank", 1)
                        title = s.get("title", "Unknown")
                        page = s.get("page", "N/A")
                        pid = s.get("paper_id", "")
                        category = s.get("category", "")
                        score = s.get("score", 0.0)
                        passage = s.get("passage", "")

                        with st.container():
                            st.markdown(f"""
                            <div class="citation-card">
                                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">
                                    [{rank}] {title}
                                </div>
                                <div style="margin-bottom: 8px;">
                                    <span class="badge badge-page">📄 Page {page}</span>
                                    <span class="badge badge-id">arXiv:{pid}</span>
                                    <span class="badge badge-score">Relevance Score: {score:.4f}</span>
                                    <span style="font-size: 0.85rem; color: #64748B;">Category: {category}</span>
                                </div>
                                <div style="font-size: 0.92rem; color: #334155; line-height: 1.5; font-style: italic; background: #FFFFFF; padding: 8px; border-radius: 4px; border-left: 3px solid #3B82F6;">
                                    "{passage}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred while processing the question: {e}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>Pinnacle Plus Capstone Project | GenAI & LLM Research Paper Answer Bot | Academic Submission</div>", unsafe_allow_html=True)

