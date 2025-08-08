import os
import asyncio
import httpx
import fitz  # PyMuPDF
import tempfile
import numpy as np
import hashlib
import time
from functools import partial
from contextlib import asynccontextmanager
from typing import List, Dict

from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
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
http_client: httpx.AsyncClient | None = None
openai_client: AsyncOpenAI | None = None
cross_encoder: CrossEncoder | None = None
security = HTTPBearer()
document_cache = {}

# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, openai_client, cross_encoder
    http_client = httpx.AsyncClient(timeout=180.0)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    yield
    await http_client.aclose()

# --- FastAPI App ---
app = FastAPI(title="Optimized RAG API", lifespan=lifespan)
api_router = APIRouter(prefix="/api/v1")

# --- Models ---
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- Auth ---
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

# --- Chunking ---
def create_chunks(text: str) -> Dict:
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    parent_chunks = parent_splitter.split_text(text)
    child_chunks = []
    parent_child_map = {}
    child_chunk_to_index_map = {}

    for parent in parent_chunks:
        children = child_splitter.split_text(parent)
        for c in children:
            idx = len(child_chunks)
            child_chunks.append(c)
            parent_child_map[idx] = parent
            child_chunk_to_index_map[c] = idx

    return {
        "child_chunks": child_chunks,
        "parent_child_map": parent_child_map,
        "child_chunk_to_index_map": child_chunk_to_index_map,
    }

# --- Embeddings ---
async def embed_content_async(texts: List[str]) -> np.ndarray:
    response = await openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    return np.array([r.embedding for r in response.data])

# --- Hybrid Search ---
def reciprocal_rank_fusion(results: List[List[int]], k: int = 60) -> Dict[int, float]:
    fused_scores = {}
    for doc_scores in results:
        for rank, doc_id in enumerate(doc_scores):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (rank + k)
    return dict(sorted(fused_scores.items(), key=lambda item: item[1], reverse=True))

async def hybrid_search(query: str, chunks: List[str], embeddings: np.ndarray, bm25: BM25Okapi, top_k: int) -> List[int]:
    query_embedding = await embed_content_async([query])
    vector_scores = cosine_similarity(query_embedding, embeddings)[0]
    vector_results = np.argsort(vector_scores)[::-1][:top_k].tolist()
    keyword_results = np.argsort(bm25.get_scores(query.lower().split()))[::-1][:top_k].tolist()
    fused_results = reciprocal_rank_fusion([vector_results, keyword_results])
    return list(fused_results.keys())[:top_k]

async def rerank_with_cross_encoder(query: str, chunks: List[str]) -> List[str]:
    pairs = [[query, chunk] for chunk in chunks]
    scores = await asyncio.to_thread(partial(cross_encoder.predict, sentences=pairs))
    return [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]

async def generate_answer_async(question: str, context: str) -> str:
    system_prompt = "You are a helpful assistant. Answer the question based ONLY on the provided excerpts. Be concise and accurate."
    user_prompt = f"<DOCUMENT_EXCERPTS>\n---\n{context}\n---\n</DOCUMENT_EXCERPTS>\n\n<QUESTION>\n{question}\n</QUESTION>"
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error during generation: {e}"

# --- RAG Pipeline ---
@api_router.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_rag_pipeline(request: RunRequest):
    t0 = time.perf_counter()

    doc_hash = hashlib.sha256(request.documents.encode()).hexdigest()
    if doc_hash in document_cache:
        cached = document_cache[doc_hash]
    else:
        text = await download_and_extract_pdf_text(request.documents)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Document is empty or unreadable")
        chunks_data = await asyncio.to_thread(create_chunks, text)
        embeddings = await embed_content_async(chunks_data["child_chunks"])
        bm25_index = BM25Okapi([c.lower().split() for c in chunks_data["child_chunks"]])
        cached = {**chunks_data, "chunk_embeddings": embeddings, "bm25_index": bm25_index}
        document_cache[doc_hash] = cached

    async def process_question(q: str) -> str:
        candidate_ids = await hybrid_search(q, cached["child_chunks"], cached["chunk_embeddings"], cached["bm25_index"], top_k=50)
        candidates = [cached["child_chunks"][i] for i in candidate_ids]
        reranked = await rerank_with_cross_encoder(q, candidates)
        top_chunks = reranked[:5]
        parent_ids = [cached["child_chunk_to_index_map"][c] for c in top_chunks]
        parent_contexts = list(dict.fromkeys([cached["parent_child_map"][i] for i in parent_ids]))
        return await generate_answer_async(q, "\n---\n".join(parent_contexts))

    answers = await asyncio.gather(*(process_question(q) for q in request.questions))
    print(f"✅ Completed in {time.perf_counter() - t0:.2f}s")
    return RunResponse(answers=answers)

# --- Mount API ---
app.include_router(api_router)
