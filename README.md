# 🩹 First Aid RAG Assistant

مساعد إسعافات أولية بنظام **RAG** (Retrieval-Augmented Generation) مبني على:
- **TF-IDF + BM25 + Semantic Embeddings** (بحث هجين Hybrid Retrieval)
- **Cross-Encoder Reranking** (`ms-marco-MiniLM-L12-v2`)
- **Groq LLM API** (بديل سحابي لـ Ollama المحلي)
- **دعم اللغة العربية** عن طريق كشف اللغة والترجمة التلقائية

مبني فوق قاعدة معرفة جاهزة (`data/first_aid_semantic_chunks_final.csv`) مستخرجة من
*First Aid Reference Guide, 4th Edition* — St. John Ambulance Canada.

---

## التشغيل محليًا (اختياري)

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# افتح secrets.toml وحط مفتاح Groq بتاعك
streamlit run app.py
```

---

## النشر (Deployment) — الخطوات كاملة في README الرئيسي للمحادثة
راجع الشرح خطوة بخطوة اللي اتبعت في المحادثة، أو ملخص سريع:

1. اعمل مفتاح مجاني من https://console.groq.com/keys
2. ارفع الفولدر ده على GitHub (repo عام أو خاص)
3. روح https://share.streamlit.io → New app → اختَر الـ repo → `app.py`
4. من Settings → Secrets ضيف:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
5. Deploy — هتاخد لينك دائم زي:
   `https://your-app-name.streamlit.app`

---

## هيكل المشروع

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
