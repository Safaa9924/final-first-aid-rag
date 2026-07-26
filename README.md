# 🚑 First Aid RAG Assistant

A hybrid Retrieval-Augmented Generation (RAG) system built on the
**St. John Ambulance Canada — First Aid Reference Guide, 4th Edition**.

The document is parsed, cleaned, chunked, indexed three ways (TF‑IDF, BM25,
sentence embeddings), stored in Chroma, retrieved with a weighted hybrid
search + cross-encoder reranker, and finally answered by an LLM through
**OpenRouter** inside an attractive first-aid themed Streamlit chat app.

```
PDF ──▶ 01_documents ──▶ 02_preprocessing ──▶ 03_chunking
                                                   │
                                                   ▼
                                      04_vector_representation
                                        (TF‑IDF · BM25 · Embeddings)
                                                   │
                                                   ▼
                                      05_create_chroma_store
                                                   │
                        user question ─────────────┼─────────────
                                                   ▼
                                      06_retrieve_context
                    (language detect → translate → expand →
                     hybrid retrieval → cross-encoder rerank →
                     context packaging)
                                                   │
                                                   ▼
                                        07_prompting
                          (prompt build → OpenRouter → grounding)
                                                   │
                                                   ▼
                                     streamlit_app.py 🚑
```

## 📁 Files

| File | Purpose |
|---|---|
| `01_documents.py` | Loads the source PDF with Docling, preserving structure. |
| `02_preprocessing.py` | Cleans text, removes publisher front matter, normalizes structure. |
| `03_chunking.py` | Adaptive semantic chunking + metadata, saves chunk table (CSV). |
| `04_vector_representation.py` | Builds TF‑IDF, BM25, and sentence-embedding indexes. |
| `05_create_chroma_store.py` | Loads chunks + embeddings into a persistent Chroma collection. |
| `06_retrieve_context.py` | Query language detection, translation, expansion, hybrid retrieval, reranking, context packaging. |
| `07_prompting.py` | Builds the grounded first-aid prompt and calls **OpenRouter** for generation; produces the confidence/grounding report. |
| `streamlit_app.py` | The web app — first-aid themed chat UI tying every stage together. |
| `requirements.txt` | Python dependencies. |

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Set the source PDF path (defaults to the filename in the current folder):

```bash
export FIRST_AID_PDF_PATH="/path/to/First aid reference guide_V4.1_Public.pdf"
```

### 🔑 OpenRouter API key — kept out of the code

The app reads your key from one of these, **never from the source files**:

- A Streamlit secrets file at `.streamlit/secrets.toml`:
  ```toml
  OPENROUTER_API_KEY = "sk-or-..."
  ```
- Or an environment variable:
  ```bash
  export OPENROUTER_API_KEY="sk-or-..."
  ```

Optional: override the default model with `OPENROUTER_MODEL`
(also selectable from the app's sidebar).

## ▶️ Build the index (run once, or whenever the PDF changes)

```bash
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_vector_representation.py
python 05_create_chroma_store.py
```

## 🌐 Launch the app

```bash
streamlit run streamlit_app.py
```

## 🩹 Notes

- The app supports **Arabic and English** questions — Arabic questions are
  translated to English for retrieval, and the final answer is translated
  back to Arabic.
- Answers are strictly grounded in the retrieved guide text; if nothing
  relevant is found, the assistant says so instead of guessing.
- This tool is for **educational reference only** — it is not a substitute
  for professional medical care or emergency services.

```
first-aid-rag/
├── app.py                  # تطبيق Streamlit الكامل (retrieval + rerank + LLM)
├── requirements.txt
├── data/
│   └── first_aid_semantic_chunks_final.csv
├── .streamlit/
│   └── secrets.toml.example
└── .gitignore
```
