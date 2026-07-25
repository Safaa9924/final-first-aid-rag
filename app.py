"""
First Aid RAG Assistant
========================
Hybrid retrieval (TF-IDF + BM25 + Semantic embeddings) + Cross-Encoder
reranking + Groq LLM generation, over a pre-chunked First Aid Reference
Guide (St. John Ambulance Canada).

Deployed on Streamlit Community Cloud. LLM calls go to Groq's free API
(instead of local Ollama, which cannot run on Streamlit Cloud).
"""

import os
import re
import ast
import requests
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from langdetect import detect
from deep_translator import GoogleTranslator

# --------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------
st.set_page_config(
    page_title="First Aid RAG Assistant",
    page_icon="🩹",
    layout="wide",
)

DATA_PATH = os.path.join("data", "first_aid_semantic_chunks_final.csv")

TOP_K = 40
TOP_N_RERANK = 8
TFIDF_WEIGHT = 0.1
BM25_WEIGHT = 0.1
SEMANTIC_WEIGHT = 0.8
MAX_CONTEXT_CHUNKS = 6
WORD_BUDGET = 1200
MAX_CHUNK_WORDS = 180

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ======================================================================
# Loading & index building (cached so this only runs once per instance)
# ======================================================================

@st.cache_data(show_spinner=False)
def load_chunks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["chunk_text"] = df["chunk_text"].astype(str)
    return df


@st.cache_resource(show_spinner="Building TF-IDF index...")
def build_tfidf_index(texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
        dtype="float32",
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def simple_tokenize(text: str):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class MiniBM25:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = tokenized_docs
        self.N = len(tokenized_docs)
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = np.mean(self.doc_lens) if self.doc_lens else 0.0
        self.term_freqs = [Counter(doc) for doc in tokenized_docs]

        df = Counter()
        for doc in tokenized_docs:
            for term in set(doc):
                df[term] += 1
        self.idf = {
            term: np.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens):
        scores = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            tf = self.term_freqs[i]
            dl = self.doc_lens[i]
            for term in query_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (freq * (self.k1 + 1)) / (denom or 1)
        return scores


@st.cache_resource(show_spinner="Building BM25 index...")
def build_bm25_index(texts):
    tokenized = [simple_tokenize(t) for t in texts]
    return MiniBM25(tokenized)


@st.cache_resource(show_spinner="Loading embedding model (first run only, ~30s)...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Building semantic embedding index...")
def build_embedding_index(texts):
    model = load_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


@st.cache_resource(show_spinner="Loading cross-encoder reranker (first run only)...")
def load_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_NAME)


def min_max_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# ======================================================================
# Retrieval
# ======================================================================

def retrieve_hybrid(query, tfidf_vectorizer, tfidf_matrix, bm25, embedding_model,
                     embedding_matrix, chunks_df, k=TOP_K):
    # TF-IDF
    q_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # BM25
    bm25_scores = bm25.get_scores(simple_tokenize(query))

    # Semantic
    q_emb = embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sem_scores = cosine_similarity(q_emb, embedding_matrix).flatten()

    combined = (
        TFIDF_WEIGHT * min_max_normalize(tfidf_scores)
        + BM25_WEIGHT * min_max_normalize(bm25_scores)
        + SEMANTIC_WEIGHT * min_max_normalize(sem_scores)
    )

    ranking = np.argsort(combined)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["hybrid_score"] = combined[ranking]
    return results.reset_index(drop=True)


def rerank_candidates(query, candidates_df, top_n=TOP_N_RERANK):
    reranker = load_cross_encoder()
    pairs = [(query, text) for text in candidates_df["chunk_text"].tolist()]
    scores = reranker.predict(pairs)
    df = candidates_df.copy()
    df["rerank_score"] = scores
    df = df.sort_values("rerank_score", ascending=False).head(top_n).reset_index(drop=True)
    return df


def build_context_package(reranked_df, max_chunks=MAX_CONTEXT_CHUNKS,
                           word_budget=WORD_BUDGET, max_chunk_words=MAX_CHUNK_WORDS):
    selected_rows = []
    total_words = 0
    for _, row in reranked_df.iterrows():
        if len(selected_rows) >= max_chunks:
            break
        text = row["chunk_text"]
        words = text.split()
        if len(words) > max_chunk_words:
            text = " ".join(words[:max_chunk_words])
            words = text.split()
        if total_words + len(words) > word_budget and selected_rows:
            continue
        total_words += len(words)
        selected_rows.append({**row.to_dict(), "chunk_text": text})

    selected_df = pd.DataFrame(selected_rows)
    context_text = "\n\n---\n\n".join(selected_df["chunk_text"].tolist())
    return selected_df, context_text


# ======================================================================
# Language handling
# ======================================================================

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return "ar" if lang == "ar" else "en"
    except Exception:
        return "en"


def translate(text: str, target: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text


# ======================================================================
# LLM generation (Groq)
# ======================================================================

def build_prompt(question: str, context: str) -> str:
    return f"""You are an expert Evidence-Based First Aid Assistant.
Answer the user question strictly in ENGLISH using ONLY the provided context. Never add outside knowledge.
If the answer is not in the context, respond exactly: "I couldn't find this information in the retrieved first aid reference."

RULES:
1. Be concise and direct. Use at most 5 short bullet points.
2. Do not repeat advice.
3. Do not mention "the context" or "the document" in your answer — answer as direct first-aid guidance.
4. Only use facts present in the context below.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def generate_answer(prompt: str, api_key: str, model: str = GROQ_MODEL,
                     temperature: float = 0.1, max_tokens: int = 800) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ======================================================================
# UI
# ======================================================================

def main():
    st.title("🩹 First Aid RAG Assistant")
    st.caption(
        "Hybrid RAG (TF-IDF + BM25 + Semantic) + Cross-Encoder Reranking + Groq LLM "
        "— grounded in the *First Aid Reference Guide* (St. John Ambulance Canada)."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else "",
            help="Get a free key at https://console.groq.com/keys",
        )
        show_sources = st.checkbox("Show retrieved sources", value=True)
        st.markdown("---")
        st.markdown(
            "**Note:** Ask in English or Arabic — the app auto-detects the "
            "language and translates as needed."
        )

    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at `{DATA_PATH}`. Make sure it is committed to the repo.")
        st.stop()

    chunks_df = load_chunks(DATA_PATH)
    texts = chunks_df["chunk_text"].tolist()

    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(texts)
    bm25 = build_bm25_index(texts)
    embedding_model = load_embedding_model()
    embedding_matrix = build_embedding_index(texts)

    st.success(f"✅ Knowledge base ready — {len(chunks_df)} chunks indexed.", icon="✅")

    question = st.text_input(
        "Ask a first-aid question / اسأل سؤال إسعافات أولية:",
        placeholder="e.g. What should I do for a conscious adult who is choking?",
    )
    ask = st.button("Ask / اسأل", type="primary")

    if ask and question.strip():
        if not api_key:
            st.warning("Please enter a Groq API key in the sidebar (it's free).")
            st.stop()

        with st.spinner("Retrieving relevant information..."):
            lang = detect_language(question)
            retrieval_query = translate(question, "en") if lang == "ar" else question

            candidates = retrieve_hybrid(
                retrieval_query, tfidf_vectorizer, tfidf_matrix, bm25,
                embedding_model, embedding_matrix, chunks_df, k=TOP_K,
            )
            reranked = rerank_candidates(retrieval_query, candidates, top_n=TOP_N_RERANK)
            selected_df, context_text = build_context_package(reranked)

        with st.spinner("Generating answer..."):
            prompt = build_prompt(retrieval_query, context_text)
            try:
                answer_en = generate_answer(prompt, api_key=api_key)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

            final_answer = translate(answer_en, "ar") if lang == "ar" else answer_en

        st.markdown("### 📋 Answer")
        st.markdown(final_answer)

        if show_sources and not selected_df.empty:
            st.markdown("### 📚 Sources used")
            for _, row in selected_df.iterrows():
                section = row.get("section", "N/A")
                score = row.get("rerank_score", 0)
                with st.expander(f"{row['chunk_id']} — {section} (score: {score:.2f})"):
                    st.write(row["chunk_text"])

    elif ask:
        st.warning("Please type a question first.")


if __name__ == "__main__":
    main()
