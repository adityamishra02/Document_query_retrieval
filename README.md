# 📄 Fast GPT-5 RAG API

A high-performance Retrieval-Augmented Generation (RAG) API powered by **FastAPI**, **PyMuPDF**, **OpenAI GPT-5**, and **BM25** for accurate, context-aware answers from PDF documents.  

This project downloads a PDF, extracts text, chunks it, runs hybrid search (vector + BM25), and generates precise answers to user questions — all in one API call.

---

## 🚀 Features
- **Hybrid Search**: Combines semantic embeddings and BM25 keyword scoring.
- **Fast PDF Processing**: Uses `PyMuPDF` for efficient text extraction.
- **Accurate Context Retrieval**: Retrieves only the most relevant document chunks.
- **Optimized for Speed**: Tuned to avoid Heroku request timeouts.
- **Bearer Token Security**: Static API token authentication for controlled access.

---

## 📦 Requirements
```bash
pip install fastapi uvicorn httpx pymupdf numpy scikit-learn rank-bm25 langchain-text-splitters python-dotenv openai pydantic
```

---

## ⚙️ Environment Variables
Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_api_key
STATIC_API_TOKEN=your_static_token
```

---

## ▶️ Running Locally
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 🔑 Authentication
All requests require a **Bearer Token** in the `Authorization` header:
```
Authorization: Bearer YOUR_STATIC_API_TOKEN
```

---

## 📡 API Endpoint

### **POST** `/api/v1/hackrx/run`

**Request Body:**
```json
{
  "documents": "https://example.com/sample.pdf",
  "questions": [
    "What is the main topic of this document?",
    "Does it contain information about AI?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The main topic is the use of AI in modern applications.",
    "Yes, it describes several AI use cases."
  ]
}
```

---

## 🧪 Example with `curl`
```bash
curl -X POST "https://hackrx01-ce36358d4306.herokuapp.com/api/v1/hackrx/run" -H "Content-Type: application/json" -H "Authorization: Bearer a727d9d8a26f71359d7f45ba30f104d07d4174c8b5818e962ad7c0e1f6fffd48" -d '{
  "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
  "questions": [
    "What text appears in the PDF?",
    "Does it mention accessibility?"
  ]
}'
```

---

## 🛠 Tech Stack
- **FastAPI** – High-performance Python web framework.
- **PyMuPDF (fitz)** – PDF text extraction.
- **OpenAI GPT-5** – LLM for generating answers.
- **BM25Okapi** – Keyword-based ranking.
- **LangChain** – Text chunking.

---

## 📜 License
MIT License — free to use and modify.
