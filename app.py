import os
import asyncio
import httpx
import fitz  
import tempfile
import numpy as np
from contextlib import asynccontextmanager
from typing import List
from collections import OrderedDict

from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STATIC_API_TOKEN = os.getenv("STATIC_API_TOKEN")

if not OPENAI_API_KEY or not STATIC_API_TOKEN:
    raise RuntimeError("OPENAI_API_KEY or STATIC_API_TOKEN is missing")

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

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != STATIC_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

def extract_text_from_pdf(file_path: str) -> str:
    with fitz.open(file_path) as doc:
        return "\n".join([page.get_text() for page in doc])

async def download_and_extract_pdf_text(url: str) -> str:
    try:
        response = await http_client.get(url, follow_redirects=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            return await asyncio.to_thread(extract_text_from_pdf, tmp_file.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF download or parsing failed: {e}")

def create_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return splitter.split_text(text)

def deduplicate_chunks(chunks: List[str]) -> List[str]:
    return list(OrderedDict.fromkeys(chunks))

async def embed_content_async(texts: List[str]) -> np.ndarray:
    resp = await openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([r.embedding for r in resp.data])

async def hybrid_search(query: str, chunks: List[str], embeddings: np.ndarray, bm25: BM25Okapi, top_k: int = 15) -> List[str]:
    query_embedding = await embed_content_async([query])
    vector_scores = cosine_similarity(query_embedding, embeddings)[0]
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores
    final_scores = 0.6 * vector_scores + 0.4 * bm25_norm
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]

async def generate_answer_async(question: str, context: str) -> str:
    prompt = (
        "You are a domain expert. Answer the question strictly based on the provided context. "
        "If the context does not contain enough information, reply with 'No relevant information found.' "
        "Do not make up facts.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

@api_router.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_rag_pipeline(request: RunRequest):
    text = await download_and_extract_pdf_text(request.documents)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Document is empty or unreadable")

    chunks = await asyncio.to_thread(create_chunks, text)
    embeddings = await embed_content_async(chunks)
    bm25_index = BM25Okapi([c.lower().split() for c in chunks])

    async def process_question(q: str) -> str:
        relevant_chunks = deduplicate_chunks(await hybrid_search(q, chunks, embeddings, bm25_index, top_k=15))
        context = "\n---\n".join(relevant_chunks)
        if len(context) > 9000:
            context = context[:9000]
        answer = await generate_answer_async(q, context)
        if "No relevant information" in answer:
            fallback_chunks = [chunks[i] for i in np.argsort(bm25_index.get_scores(q.lower().split()))[::-1][:5]]
            context_fb = "\n---\n".join(fallback_chunks)
            answer = await generate_answer_async(q, context_fb)
        return answer.replace("\n", " ").strip()

    answers = await asyncio.gather(*(process_question(q) for q in request.questions))
    answers = [a.replace("\n", " ").strip() for a in answers]
    return RunResponse(answers=answers)

app.include_router(api_router)
