import os
import asyncio
import httpx
import fitz  # PyMuPDF
import tempfile
import numpy as np
from openai import AsyncOpenAI, OpenAI
from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache
from dotenv import load_dotenv
import hashlib
import time

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STATIC_API_TOKEN = os.getenv("STATIC_API_TOKEN")

if not OPENAI_API_KEY or not STATIC_API_TOKEN:
    raise RuntimeError("Environment variables OPENAI_API_KEY and STATIC_API_TOKEN are required.")

# Global clients
http_client: httpx.AsyncClient | None = None
openai_client: AsyncOpenAI | None = None
cross_encoder: CrossEncoder | None = None
security = HTTPBearer()

# In-memory cache
document_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, openai_client, cross_encoder
    print("Initializing clients and models...")
    http_client = httpx.AsyncClient(timeout=180.0)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')  # GPU
    print("Initialization complete.")
    yield
    await http_client.aclose()
    print("Shutdown complete.")

app = FastAPI(
    title="Optimized RAG API",
    description="High-performance RAG pipeline",
    lifespan=lifespan
)
api_router = APIRouter(prefix="/api/v1")

# --- Models ---
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != STATIC_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return True

# --- PDF Text Extraction ---
async def download_and_extract_pdf_text(url: str) -> str:
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            def extract_text():
                with fitz.open(tmp_file.name) as doc:
                    return "\n".join(page.get_text() for page in doc)
            return await asyncio.to_thread(extract_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

# --- Chunking ---
def create_chunks(text: str) -> Dict:
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    parent_chunks = parent_splitter.split_text(text)
    child_chunks = []
    parent_child_map = {}
    for parent in parent_chunks:
        children = child_splitter.split_text(parent)
        for c in children:
            idx = len(child_chunks)
            child_chunks.append(c)
            parent_child_map[idx] = parent
    return {"child_chunks": child_chunks, "parent_child_map": parent_child_map}

# --- Embedding Utilities ---
def get_embeddings_batch_sync(texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    return [r.embedding for r in client.embeddings.create(input=texts, model="text-embedding-3-small").data]

async def embed_content_async(texts: List[str]) -> List[List[float]]:
    return await asyncio.to_thread(get_embeddings_batch_sync, texts)

# --- RRF ---
def reciprocal_rank_fusion(results: List[List[int]], k: int = 60) -> Dict[int, float]:
    fused_scores = {}
    for doc_scores in results:
        for rank, doc_id in enumerate(doc_scores):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rank + k)
    return dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))

# --- Hybrid Search ---
async def hybrid_search(query: str, child_chunks: List[str], embeddings: np.ndarray, bm25_index: BM25Okapi, top_k: int) -> List[int]:
    query_embedding = np.array(await embed_content_async([query]))
    vector_scores = cosine_similarity(query_embedding, embeddings)[0]
    vector_results = np.argsort(vector_scores)[::-1][:top_k].tolist()
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    keyword_results = np.argsort(bm25_scores)[::-1][:top_k].tolist()
    fused = reciprocal_rank_fusion([vector_results, keyword_results])
    return list(fused.keys())[:top_k]

# --- CrossEncoder Rerank ---
def rerank_with_cross_encoder(query: str, chunks: List[str]) -> List[str]:
    pairs = [[query, chunk] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    return [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]

# --- GPT Answer Generation ---
async def generate_answer_async(question: str, context: str) -> str:
    system_prompt = """You are a precision-focused AI assistant. Answer the user's question using only the provided excerpts."""
    user_prompt = f"""
<DOCUMENT_EXCERPTS>
---
{context}
---
</DOCUMENT_EXCERPTS>
<QUESTION>
{question}
</QUESTION>
"""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# --- Main Pipeline ---
@api_router.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_rag_pipeline(request: RunRequest):
    print("Pipeline started.")
    t0 = time.perf_counter()

    # Document caching via SHA256
    doc_hash = hashlib.sha256(request.documents.encode()).hexdigest()
    if doc_hash in document_cache:
        print("Using cached document.")
        cached = document_cache[doc_hash]
        child_chunks = cached["child_chunks"]
        parent_child_map = cached["parent_child_map"]
        chunk_embeddings = cached["chunk_embeddings"]
        bm25_index = cached["bm25_index"]
    else:
        print("Downloading and processing new document...")
        text = await download_and_extract_pdf_text(request.documents)
        if not text:
            raise HTTPException(status_code=400, detail="Empty document.")
        chunks_data = create_chunks(text)
        child_chunks = chunks_data["child_chunks"]
        parent_child_map = chunks_data["parent_child_map"]
        chunk_embeddings = np.array(await embed_content_async(child_chunks))
        tokenized = [c.lower().split() for c in child_chunks]
        bm25_index = BM25Okapi(tokenized)
        document_cache[doc_hash] = {
            "child_chunks": child_chunks,
            "parent_child_map": parent_child_map,
            "chunk_embeddings": chunk_embeddings,
            "bm25_index": bm25_index,
        }

    print("Processing questions...")

    async def process_question(q: str) -> str:
        print(f"→ Q: {q[:50]}...")
        candidate_indices = await hybrid_search(q, child_chunks, chunk_embeddings, bm25_index, top_k=20)
        candidates = [child_chunks[i] for i in candidate_indices]
        reranked = rerank_with_cross_encoder(q, candidates)[:10]
        parent_contexts = list(dict.fromkeys([parent_child_map[child_chunks.index(c)] for c in reranked]))
        context = "\n---\n".join(parent_contexts)
        return await generate_answer_async(q, context)

    answers = await asyncio.gather(*[process_question(q) for q in request.questions])

    print(f"Pipeline completed in {time.perf_counter() - t0:.2f}s.")
    return RunResponse(answers=answers)

app.include_router(api_router)
