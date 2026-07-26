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
# Page config — لازم يتكتب مرة واحدة بس وأول أمر Streamlit في الملف
# --------------------------------------------------------------------
SITE_NAME = "نبضة"
SITE_TAGLINE = "أول خطوة نحو النجاة"

st.set_page_config(
    page_title=f"{SITE_NAME} | مساعدك في الإسعافات الأولية",
    page_icon="💓",
    layout="centered",
    initial_sidebar_state="expanded",
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

# ---------------------------------------------------------------
# API Key: بتتقرا من secrets فقط. مفيش أي input أو ذكر ليها
# في الواجهة نهائيًا. لو مش موجودة، الموقع بيوريك رسالة صيانة
# عادية من غير أي تفاصيل تقنية.
# ---------------------------------------------------------------
def _get_api_key() -> str:
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


# ---------------------------------------------------------------
# تنسيق (CSS) عشان الموقع يبقى شكله منظمة إسعافات أولية حقيقية
# ---------------------------------------------------------------
CUSTOM_CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: linear-gradient(160deg, #eafaf6 0%, #f4fbf9 25%, #ffffff 55%);
    }

    .hero {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 55%, #ff6b5b 100%);
        padding: 2.2rem 1.8rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(192,57,43,0.25);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
    }
    .hero p {
        margin-top: 0.5rem;
        font-size: 1rem;
        opacity: 0.95;
    }
    .cross {
        font-size: 2.4rem;
        line-height: 1;
        margin-bottom: 0.3rem;
    }

    .info-chip {
        display: inline-block;
        background: #ffffff;
        border: 1px solid #f1c6c0;
        color: #c0392b;
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.85rem;
        margin: 0.2rem;
        font-weight: 600;
    }

    .category-card {
        background: white;
        border-radius: 14px;
        padding: 0.9rem;
        text-align: center;
        border: 1px solid #d9f0ea;
        box-shadow: 0 2px 10px rgba(15,118,110,0.06);
        height: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .category-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(15,118,110,0.12);
    }
    .category-card .emoji {
        font-size: 1.8rem;
    }
    .category-card .label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #444;
        margin-top: 0.2rem;
    }

    .disclaimer {
        background: #fff8e6;
        border: 1px solid #ffe2a3;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #7a5b00;
        margin-top: 1rem;
    }

    section[data-testid="stChatMessage"] {
        border-radius: 14px;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px !important;
        border: 1px solid #eee !important;
    }
</style>
"""

FIRST_AID_TOPICS = [
    ("🩸", "نزيف"),
    ("🔥", "حروق"),
    ("🫁", "اختناق"),
    ("❤️", "إنعاش قلبي (CPR)"),
    ("🦴", "كسور"),
    ("🐝", "لسعات وحساسية"),
]


def render_hero():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero">
            <div class="cross">💓</div>
            <h1>{SITE_NAME}</h1>
            <p style="font-weight:600; opacity:0.9; margin-top:-0.3rem;">{SITE_TAGLINE}</p>
            <p>مساعدك الفوري والموثوق في حالات الطوارئ — إجابات مبنية على دليل الإسعافات
            الأولية المعتمد من St. John Ambulance Canada</p>
            <div>
                <span class="info-chip">🌐 عربي / English</span>
                <span class="info-chip">⚡ إجابة فورية</span>
                <span class="info-chip">📚 مصادر موثقة</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topics():
    st.markdown("#### 🗂️ استكشف الحالات الشائعة")
    cols = st.columns(len(FIRST_AID_TOPICS))
    clicked_topic = None
    for col, (emoji, label) in zip(cols, FIRST_AID_TOPICS):
        with col:
            st.markdown(
                f"""
                <div class="category-card">
                    <div class="emoji">{emoji}</div>
                    <div class="label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("اسأل", key=f"topic_{label}", use_container_width=True):
                clicked_topic = label
    return clicked_topic


def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 💓 عن {SITE_NAME}")
        st.write(
            f"«{SITE_NAME}» منصة إرشادية تقدّم معلومات إسعافات أولية سريعة وموثوقة "
            "بالعربي والإنجليزي، مبنية على مصادر طبية معتمدة."
        )
        st.markdown("---")
        st.markdown("### 🚑 تذكير مهم")
        st.markdown(
            "<div class='disclaimer'>في حالة الطوارئ الحقيقية، اتصل فورًا "
            "بخدمات الإسعاف المحلية (123). المعلومات هنا للإرشاد الأولي فقط "
            "ولا تغني عن الرعاية الطبية المتخصصة.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        show_sources = st.toggle("📚 عرض المصادر مع كل إجابة", value=False)
        if st.button("🗑️ مسح المحادثة", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        return show_sources


def main():
    render_hero()
    show_sources = render_sidebar()

    api_key = _get_api_key()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not os.path.exists(DATA_PATH):
        st.error("عذرًا، قاعدة المعرفة غير متاحة حاليًا. برجاء المحاولة لاحقًا.")
        st.stop()

    with st.spinner("جارِ تجهيز قاعدة المعرفة..."):
        chunks_df = load_chunks(DATA_PATH)
        texts = chunks_df["chunk_text"].tolist()
        tfidf_vectorizer, tfidf_matrix = build_tfidf_index(texts)
        bm25 = build_bm25_index(texts)
        embedding_model = load_embedding_model()
        embedding_matrix = build_embedding_index(texts)

    clicked_topic = render_topics()
    st.markdown("---")
    st.markdown("#### 💬 اسأل المساعد")

    # عرض المحادثة السابقة على هيئة شات
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🚑" if msg["role"] == "assistant" else "🧑"):
            st.markdown(msg["content"])
            if msg.get("sources") and show_sources:
                with st.expander("📚 المصادر المستخدمة"):
                    for s in msg["sources"]:
                        st.markdown(f"**{s['section']}**")
                        st.write(s["text"])

    question = st.chat_input("اكتب سؤالك عن الإسعافات الأولية هنا...")
    if clicked_topic and not question:
        question = f"إزاي أتصرف في حالة {clicked_topic}؟"

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        if not api_key:
            with st.chat_message("assistant", avatar="🚑"):
                st.warning("الخدمة غير متاحة حاليًا، برجاء المحاولة بعد قليل 🙏")
            st.stop()

        with st.chat_message("assistant", avatar="🚑"):
            with st.spinner("جارِ البحث عن أفضل إجابة..."):
                lang = detect_language(question)
                retrieval_query = translate(question, "en") if lang == "ar" else question
                candidates = retrieve_hybrid(
                    retrieval_query, tfidf_vectorizer, tfidf_matrix, bm25,
                    embedding_model, embedding_matrix, chunks_df, k=TOP_K,
                )
                reranked = rerank_candidates(retrieval_query, candidates, top_n=TOP_N_RERANK)
                selected_df, context_text = build_context_package(reranked)

                prompt = build_prompt(retrieval_query, context_text)
                try:
                    answer_en = generate_answer(prompt, api_key=api_key)
                except Exception:
                    st.error("حصل خطأ أثناء تجهيز الإجابة، حاول تاني من فضلك.")
                    st.stop()

                final_answer = translate(answer_en, "ar") if lang == "ar" else answer_en

            st.markdown(final_answer)

            sources_payload = []
            if not selected_df.empty:
                for _, row in selected_df.iterrows():
                    sources_payload.append({
                        "section": row.get("section", "N/A"),
                        "text": row["chunk_text"],
                    })
                if show_sources:
                    with st.expander("📚 المصادر المستخدمة"):
                        for s in sources_payload:
                            st.markdown(f"**{s['section']}**")
                            st.write(s["text"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_answer,
            "sources": sources_payload,
        })


if __name__ == "__main__":
    main()
