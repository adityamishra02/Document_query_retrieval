import os
import asyncio
import httpx
import fitz  # PyMuPDF
import tempfile
import numpy as np
from contextlib import asynccontextmanager
from typing import List

from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STATIC_API_TOKEN = os.getenv("STATIC_API_TOKEN")

if not OPENAI_API_KEY or not STATIC_API_TOKEN:
    raise RuntimeError("OPENAI_API_KEY or STATIC_API_TOKEN is missing")

# --- Global Clients ---
http_client = None
openai_client = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, openai_client
    http_client = httpx.AsyncClient(timeout=120.0)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    yield
    await http_client.aclose()

app = FastAPI(title="Fast GPT-5 RAG", lifespan=lifespan)
api_router = APIRouter(prefix="/api/v1")

# --- Schemas ---
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != STATIC_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# --- PDF Extraction ---
def extract_text_from_pdf(file_path: str) -> str:
    with fitz.open(file_path) as doc:
        return "\n".join([page.get_text() for page in doc])

async def download_and_extract_pdf_text(url: str) -> str:
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            return await asyncio.to_thread(extract_text_from_pdf, tmp_file.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF download or parsing failed: {e}")

# --- Text Chunking ---
def create_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_text(text)

# --- Embeddings ---
async def embed_content_async(texts: List[str]) -> np.ndarray:
    resp = await openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([r.embedding for r in resp.data])

# --- Hybrid Search ---
async def hybrid_search(query: str, chunks: List[str], embeddings: np.ndarray, bm25: BM25Okapi, top_k: int = 10) -> List[str]:
    query_embedding = await embed_content_async([query])
    vector_scores = cosine_similarity(query_embedding, embeddings)[0]
    bm25_scores = bm25.get_scores(query.lower().split())

    bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores
    final_scores = 0.7 * vector_scores + 0.3 * bm25_norm

    top_indices = np.argsort(final_scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

# --- Answer Generation ---
async def generate_answer_async(question: str, context: str) -> str:
    prompt = f"Answer the question using only the given context. Be precise.\nContext:\n{context}\nQuestion: {question}"
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# --- Main Endpoint ---
@api_router.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_rag_pipeline(request: RunRequest):
    text = await download_and_extract_pdf_text(request.documents)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Document is empty or unreadable")

    chunks = await asyncio.to_thread(create_chunks, text)
    embeddings = await embed_content_async(chunks)
    bm25_index = BM25Okapi([c.lower().split() for c in chunks])

    async def process_question(q: str) -> str:
        relevant_chunks = await hybrid_search(q, chunks, embeddings, bm25_index, top_k=10)
        context = "\n---\n".join(relevant_chunks)
        if len(context) > 9000:
            context = context[:9000]
        return await generate_answer_async(q, context)

    answers = await asyncio.gather(*(process_question(q) for q in request.questions))
    return RunResponse(answers=answers)

# --- Register router ---
app.include_router(api_router)
