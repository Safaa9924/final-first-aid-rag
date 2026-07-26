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
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@700;800;900&family=Tajawal:wght@400;500;700&display=swap');

    :root {
        --ink: #0E2A24;
        --bg: #F5FAF8;
        --surface: #FFFFFF;
        --teal: #0B7A66;
        --teal-deep: #06463C;
        --coral: #FF5A4E;
        --amber: #F2A93B;
        --line: #DCEEE8;
    }

    html, body, [class^="css"], [class*=" css"] {
        font-family: 'Tajawal', sans-serif;
        color: var(--ink);
    }

    #MainMenu, footer, header {visibility: hidden;}

    .stApp { background: var(--bg); }

    .block-container { padding-top: 1.4rem; max-width: 760px; }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        background: var(--teal-deep);
        background-image:
            radial-gradient(circle at 12% 15%, rgba(11,122,102,0.65), transparent 42%),
            radial-gradient(circle at 88% 85%, rgba(255,90,78,0.22), transparent 45%);
        border-radius: 24px;
        padding: 2.6rem 1.6rem 2.1rem;
        text-align: center;
        margin-bottom: 1.8rem;
        overflow: hidden;
        box-shadow: 0 16px 40px rgba(6,70,60,0.28);
    }
    .hero-eyebrow {
        font-family: 'Tajawal', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.16em;
        color: var(--amber);
        font-weight: 700;
        text-transform: uppercase;
    }
    .hero h1 {
        font-family: 'Cairo', sans-serif;
        font-weight: 900;
        font-size: 2.8rem;
        color: #F5FAF8;
        margin: 0.25rem 0 0.15rem;
        letter-spacing: -0.01em;
    }
    .hero .tagline {
        font-family: 'Tajawal', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: #CDEDE4;
        margin-bottom: 0.9rem;
    }
    .pulse-wrap { width: 100%; max-width: 380px; margin: 0 auto 0.9rem; }
    .pulse-line {
        stroke: var(--coral);
        stroke-width: 2.5;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-dasharray: 900;
        stroke-dashoffset: 900;
        animation: draw 2.8s ease-in-out infinite;
    }
    @keyframes draw {
        0%   { stroke-dashoffset: 900; opacity: 0.35; }
        55%  { stroke-dashoffset: 0;   opacity: 1; }
        100% { stroke-dashoffset: -900; opacity: 0.35; }
    }
    .hero p.sub {
        font-size: 0.92rem;
        color: #A9D6CB;
        max-width: 460px;
        margin: 0 auto 1.1rem;
        line-height: 1.7;
    }
    .chip-row { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; }
    .info-chip {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        color: #EAFBF6;
        border-radius: 999px;
        padding: 0.35rem 0.95rem;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* ---------- Section labels ---------- */
    .section-label {
        font-family: 'Cairo', sans-serif;
        font-weight: 800;
        font-size: 1.15rem;
        color: var(--teal-deep);
        margin: 0.2rem 0 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ---------- Topic cards ---------- */
    .category-card {
        background: var(--surface);
        border-radius: 16px;
        padding: 1.1rem 0.5rem 0.9rem;
        text-align: center;
        border: 1px solid var(--line);
        box-shadow: 0 2px 10px rgba(6,70,60,0.05);
        height: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .category-card:hover {
        transform: translateY(-3px);
        border-color: var(--teal);
        box-shadow: 0 10px 22px rgba(6,70,60,0.14);
    }
    .category-card .emoji {
        font-size: 1.5rem;
        width: 44px; height: 44px;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 0.5rem;
        background: var(--bg);
        border: 1px solid var(--line);
        border-radius: 50%;
    }
    .category-card .label {
        font-family: 'Tajawal', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--ink);
    }

    /* ---------- Disclaimer ---------- */
    .disclaimer {
        background: #FFF6E8;
        border: 1px solid #F3D998;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        font-size: 0.83rem;
        color: #7A5200;
        margin-top: 0.9rem;
        line-height: 1.7;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        border-radius: 10px !important;
        font-family: 'Tajawal', sans-serif !important;
        font-weight: 700 !important;
        border: 1px solid var(--line) !important;
        color: var(--teal-deep) !important;
    }
    .stButton>button:hover {
        border-color: var(--teal) !important;
        color: var(--teal) !important;
        background: #F0FAF7 !important;
    }
    .stButton>button[kind="primary"] {
        background: var(--coral) !important;
        border-color: var(--coral) !important;
        color: white !important;
    }

    /* ---------- Chat ---------- */
    section[data-testid="stChatMessage"] {
        border-radius: 16px;
        border: 1px solid var(--line);
        background: var(--surface);
        box-shadow: 0 1px 6px rgba(6,70,60,0.04);
    }
    div[data-testid="stChatInput"] textarea {
        font-family: 'Tajawal', sans-serif !important;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px !important;
        border: 1px solid var(--line) !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--teal-deep);
        background-image: radial-gradient(circle at 20% 0%, rgba(11,122,102,0.55), transparent 55%);
    }
    section[data-testid="stSidebar"] * { color: #EAFBF6; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

    .side-brand {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 1.4rem;
    }
    .side-brand .dot {
        width: 34px; height: 34px;
        border-radius: 50%;
        background: var(--coral);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .side-brand .name {
        font-family: 'Cairo', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        color: #F5FAF8;
    }
    .side-brand .role {
        font-family: 'Tajawal', sans-serif;
        font-size: 0.72rem;
        color: #9FD9CB;
        font-weight: 500;
    }

    .side-heading {
        font-family: 'Cairo', sans-serif;
        font-weight: 800;
        font-size: 0.95rem;
        color: #F5FAF8 !important;
        margin: 1.1rem 0 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .side-text {
        font-family: 'Tajawal', sans-serif;
        font-size: 0.85rem;
        color: #BFE7DC !important;
        line-height: 1.7;
    }
    .side-divider {
        height: 1px;
        background: rgba(255,255,255,0.14);
        border: none;
        margin: 1.1rem 0;
    }

    section[data-testid="stSidebar"] .disclaimer {
        background: rgba(242,169,59,0.14);
        border: 1px solid rgba(242,169,59,0.4);
    }
    section[data-testid="stSidebar"] .disclaimer * { color: #FBE3AE !important; }

    section[data-testid="stSidebar"] .stButton>button {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        color: #EAFBF6 !important;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background: var(--coral) !important;
        border-color: var(--coral) !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] label { color: #EAFBF6 !important; }
</style>
"""

PULSE_SVG = (
    '<div class="pulse-wrap"><svg viewBox="0 0 380 60" xmlns="http://www.w3.org/2000/svg">'
    '<path class="pulse-line" d="M0,32 L70,32 L88,10 L104,52 L120,20 L138,32 L190,32 '
    'L208,10 L224,52 L240,20 L258,32 L380,32" /></svg></div>'
)

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
    hero_html = (
        '<div class="hero">'
        '<div class="hero-eyebrow">مساعد الطوارئ الرقمي</div>'
        f'<h1>{SITE_NAME}</h1>'
        f'<div class="tagline">{SITE_TAGLINE}</div>'
        f'{PULSE_SVG}'
        '<p class="sub">إجابات فورية وموثوقة وقت الطوارئ، مبنية على مرجع معتمد '
        'من أساسيات الإسعافات الأولية العالمية.</p>'
        '<div class="chip-row">'
        '<span class="info-chip">🌐 واجهة عربية بالكامل</span>'
        '<span class="info-chip">⚡ إجابة فورية</span>'
        '<span class="info-chip">📚 مصادر موثقة</span>'
        '</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def render_topics():
    st.markdown('<div class="section-label">🗂️ استكشف الحالات الشائعة</div>', unsafe_allow_html=True)
    cols = st.columns(len(FIRST_AID_TOPICS))
    clicked_topic = None
    for col, (emoji, label) in zip(cols, FIRST_AID_TOPICS):
        with col:
            card_html = (
                '<div class="category-card">'
                f'<div class="emoji">{emoji}</div>'
                f'<div class="label">{label}</div>'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)
            if st.button("اسأل", key=f"topic_{label}", use_container_width=True):
                clicked_topic = label
    return clicked_topic


def render_sidebar():
    with st.sidebar:
        brand_html = (
            '<div class="side-brand">'
            '<div class="dot">💓</div>'
            '<div>'
            f'<div class="name">{SITE_NAME}</div>'
            '<div class="role">مساعد الإسعافات الأولية</div>'
            '</div>'
            '</div>'
        )
        st.markdown(brand_html, unsafe_allow_html=True)
        st.markdown(f'<div class="side-heading">💓 عن {SITE_NAME}</div>', unsafe_allow_html=True)
        about_html = (
            f'<p class="side-text">«{SITE_NAME}» منصة إرشادية تقدّم معلومات إسعافات أولية '
            'سريعة وموثوقة، مبنية على مصادر طبية معتمدة.</p>'
        )
        st.markdown(about_html, unsafe_allow_html=True)
        st.markdown('<hr class="side-divider">', unsafe_allow_html=True)
        st.markdown('<div class="side-heading">🚑 تذكير مهم</div>', unsafe_allow_html=True)
        disclaimer_html = (
            "<div class='disclaimer'>في حالة الطوارئ الحقيقية، اتصل فورًا "
            "بخدمات الإسعاف المحلية (123). المعلومات هنا للإرشاد الأولي فقط "
            "ولا تغني عن الرعاية الطبية المتخصصة.</div>"
        )
        st.markdown(disclaimer_html, unsafe_allow_html=True)
        st.markdown('<hr class="side-divider">', unsafe_allow_html=True)
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
    st.markdown('<div class="section-label">💬 اسأل المساعد</div>', unsafe_allow_html=True)

    # عرض المحادثة السابقة على هيئة شات
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🚑" if msg["role"] == "assistant" else "🧑"):
            if msg["role"] == "assistant":
                if "content_en" in msg:
                    st.markdown("**🇬🇧 English**")
                    st.markdown(msg["content_en"])
                    st.markdown("---")
                    st.markdown("**🇪🇬 بالعربي**")
                    st.markdown(msg["content_ar"])
                else:
                    # محادثات قديمة اتخزنت قبل التعديل — تعرض بشكلها الأصلي
                    st.markdown(msg.get("content", ""))
            else:
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

                answer_ar = translate(answer_en, "ar")

            st.markdown("**🇬🇧 English**")
            st.markdown(answer_en)
            st.markdown("---")
            st.markdown("**🇪🇬 بالعربي**")
            st.markdown(answer_ar)

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
            "content_en": answer_en,
            "content_ar": answer_ar,
            "sources": sources_payload,
        })


if __name__ == "__main__":
    main()
