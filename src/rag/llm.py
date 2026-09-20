"""
LLM Client Interface for Pinnacle Plus Capstone.
Multi-tier resilient generation engine:
  1. Google Gemini (e.g. gemini-1.5-flash / gemini-2.0-flash) via GEMINI_API_KEY (Recommended & Free).
  2. OpenAI API (e.g. gpt-4o-mini) via OPENAI_API_KEY.
  3. Hugging Face Serverless (Qwen/Qwen2.5-72B-Instruct) via HF_TOKEN.
  4. Local Grounded Synthesizer when offline, strictly enforcing context grounding.
"""

import os
import re
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from src.rag.prompt import SYSTEM_PROMPT, build_prompt

load_dotenv()

class LLMClient:
    """
    Manages generation requests across Google Gemini, OpenAI, Hugging Face, or local grounded fallback.
    """
    def __init__(self, gemini_model: str = "gemini-2.5-flash", openai_model: str = "gpt-4o-mini", model_name: Optional[str] = None):
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.hf_token = os.environ.get("HF_TOKEN")

        if model_name:
            if "gemini" in model_name.lower():
                gemini_model = model_name
            elif "gpt" in model_name.lower():
                openai_model = model_name

        self.gemini_model_name = gemini_model
        self.openai_model_name = openai_model

        self.gemini_client = None
        self.openai_client = None
        self.hf_client = None

        # 1. Initialize Google Gemini client if GEMINI_API_KEY is configured
        if self.gemini_key and not self.gemini_key.startswith("your_"):
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                print(f"LLMClient: Google Gemini client initialized ({self.gemini_model_name}).")
            except Exception as e:
                print(f"Warning: Failed to initialize Google Gemini client: {e}")
                self.gemini_client = None

        # 2. Initialize OpenAI client if OPENAI_API_KEY is configured
        if self.openai_key and not self.openai_key.startswith("your_"):
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_key)
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self.openai_client = None

        # 3. Initialize Hugging Face client if HF_TOKEN is configured
        if self.hf_token and not self.hf_token.startswith("your_"):
            try:
                from huggingface_hub import InferenceClient
                self.hf_client = InferenceClient(api_key=self.hf_token)
            except Exception as e:
                print(f"Warning: Failed to initialize HF InferenceClient: {e}")
                self.hf_client = None

        # Determine default active mode based on priority
        if self.gemini_client:
            self.mode = "gemini"
            self.last_provider = f"Google Gemini ({self.gemini_model_name})"
            print(f"LLMClient initialized: Primary provider = Google Gemini ({self.gemini_model_name}).")
        elif self.openai_client:
            self.mode = "openai"
            self.last_provider = f"OpenAI ({self.openai_model_name})"
            print(f"LLMClient initialized: Primary provider = OpenAI ({self.openai_model_name}).")
        elif self.hf_client:
            self.mode = "huggingface"
            self.last_provider = "HuggingFace (Qwen/Qwen2.5-72B-Instruct)"
            print("LLMClient initialized: Primary provider = HuggingFace (Qwen/Qwen2.5-72B-Instruct).")
        else:
            self.mode = "local_grounded"
            self.last_provider = "Local Grounded Engine"
            print("LLMClient initialized: Provider = Local Grounded Mode.")

    def generate(self, question: str, context: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generates grounded response using retrieved context with automated multi-tier failover.
        """
        # Tier 1: Try Google Gemini if configured
        if self.gemini_client:
            try:
                ans = self._generate_gemini(question, context)
                if ans:
                    self.last_provider = f"Google Gemini ({self.gemini_model_name})"
                    return ans
            except Exception as e:
                print(f"Gemini generation error: {e}. Attempting failover to secondary provider...")

        # Tier 2: Try OpenAI if configured
        if self.openai_client:
            try:
                ans = self._generate_openai(question, context)
                if ans:
                    self.last_provider = f"OpenAI ({self.openai_model_name})"
                    return ans
            except Exception as e:
                print(f"OpenAI generation error: {e}. Attempting failover to secondary provider...")

        # Tier 3: Try Hugging Face Inference if configured
        if self.hf_client:
            try:
                ans = self._generate_huggingface(question, context)
                if ans:
                    self.last_provider = "HuggingFace (Qwen/Qwen2.5-72B-Instruct)"
                    return ans
            except Exception as e:
                print(f"HuggingFace inference error: {e}. Falling back to local grounded mode...")

        # Tier 4: Local Grounded Synthesis
        self.last_provider = "Local Grounded Engine"
        return self._generate_local_grounded(question, context, retrieved_docs)

    def _generate_gemini(self, question: str, context: str) -> str:
        import logging
        logging.getLogger("google.genai").setLevel(logging.ERROR)
        from google.genai import types
        user_prompt = build_prompt(question, context)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=800,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model_name,
                contents=user_prompt,
                config=config
            )
            return response.text.strip()
        except Exception:
            # Fallback to gemini-flash-latest if gemini-2.5-flash encounters an issue
            fallback_model = "gemini-flash-latest" if self.gemini_model_name == "gemini-2.5-flash" else "gemini-2.5-flash"
            response = self.gemini_client.models.generate_content(
                model=fallback_model,
                contents=user_prompt,
                config=config
            )
            self.gemini_model_name = fallback_model
            return response.text.strip()

    def _generate_openai(self, question: str, context: str) -> str:
        user_prompt = build_prompt(question, context)
        response = self.openai_client.chat.completions.create(
            model=self.openai_model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()

    def _generate_huggingface(self, question: str, context: str) -> str:
        user_prompt = build_prompt(question, context)
        response = self.hf_client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()

    def _generate_local_grounded(self, question: str, context: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Deterministic local grounded synthesis for offline execution:
        - Extracts high-overlap sentences from top retrieved sources.
        - Strictly enforces hallucination fallback if relevant overlap is below threshold.
        """
        if not retrieved_docs:
            return "The provided research papers do not contain sufficient information to answer this question."

        # Check relevance: extract query keywords
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())) - {
            "what", "how", "why", "when", "where", "which", "does", "explain", "describe", "paper", "architecture"
        }

        # Check top document relevance score or keyword presence
        top_doc = retrieved_docs[0]
        top_content = top_doc.get("content", "")
        top_title = top_doc.get("title", "Unknown Paper")
        top_page = top_doc.get("page", 1)
        doc_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', top_content.lower()))

        overlap = len(query_words & doc_words)
        overlap_ratio = overlap / max(len(query_words), 1)
        top_score = top_doc.get("reranker_score", top_doc.get("score", 0.0))

        # Strict anti-hallucination guardrail
        if overlap < 2 or overlap_ratio < 0.25 or top_score < 0.5:
            return "The provided research papers do not contain sufficient information to answer this question."

        # Extract the most informative sentences from top passages
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', top_content) if len(s.strip()) > 30]
        scored_sentences = []
        for s in sentences:
            s_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()))
            score = len(query_words & s_words)
            scored_sentences.append((score, s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for score, s in scored_sentences[:3] if score > 0]

        if not top_sentences:
            top_sentences = sentences[:2]

        extracted_text = " ".join(top_sentences)
        answer = (
            f"Based on the research findings in [{top_title}], {extracted_text} "
            f"[Source 1: {top_title}, Page {top_page}]."
        )
        return answer
