# Fast GPT-5 RAG API

A **FastAPI-based Retrieval-Augmented Generation (RAG) service** that:
- Downloads and parses PDFs
- Splits content into chunks
- Uses hybrid search (BM25 + embeddings)
- Generates answers to questions using GPT-5

---

## Features
- **PDF Text Extraction** via [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Text Chunking** with `RecursiveCharacterTextSplitter`
- **Hybrid Search** combining BM25 and cosine similarity
- **Answer Generation** using OpenAI GPT-5
- **Bearer Token Authentication**

---

## Requirements

### Python Version
- Python 3.9+

### Dependencies
Install with:
```bash
pip install -r requirements.txt
