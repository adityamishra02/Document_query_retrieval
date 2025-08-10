import os
import asyncio
import httpx
import fitz
import tempfile
from contextlib import asynccontextmanager
from typing import List, Dict
from collections import defaultdict

from openai import AsyncOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STATIC_API_TOKEN = os.getenv("STATIC_API_TOKEN")

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024
GENERATION_MODEL = "gpt-4o"

RETRIEVER_TOP_N = 10
COMPRESSOR_TOP_K = 5
RRF_K = 60

http_client: httpx.AsyncClient = None
openai_client: AsyncOpenAI = None
chroma_client: chromadb.Client = None
openai_ef = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, openai_client, chroma_client, openai_ef
    
    http_client = httpx.AsyncClient(timeout=120.0)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    chroma_client = chromadb.Client()

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )
    
    print("Application startup complete. Clients initialized.")
    yield
    
    await http_client.aclose()
    print("Application shutdown complete. Clients closed.")

app = FastAPI(
    title="Optimized RAG Pipeline",
    description="An advanced RAG implementation based on modern research.",
    lifespan=lifespan
)
api_router = APIRouter(prefix="/api/v1")
security = HTTPBearer()

class RunRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF document to process.")
    questions: List[str] = Field(..., description="List of questions to answer based on the document.")

class RunResponse(BaseModel):
    answers: List[str]

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != STATIC_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Token")
    return True

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with fitz.open(file_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""

async def download_and_extract_pdf_text(url: str) -> str:
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            return await asyncio.to_thread(extract_text_from_pdf, tmp_file.name)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to download PDF: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")

def reciprocal_rank_fusion(*ranked_lists: List[str], k: int = 60) -> List[str]:
    rrf_scores = defaultdict(float)
    for rank_list in ranked_lists:
        for rank, doc_id in enumerate(rank_list, 1):
            rrf_scores[doc_id] += 1 / (k + rank)
    
    return sorted(rrf_scores.keys(), key=rrf_scores.get, reverse=True)

async def hybrid_search(query: str, collection: chromadb.Collection, bm25_index: BM25Okapi, all_chunks: Dict[str, str]) -> List[str]:
    vector_results = await asyncio.to_thread(
        collection.query,
        query_texts=[query],
        n_results=RETRIEVER_TOP_N
    )
    vector_doc_ids = vector_results['ids'][0]

    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:RETRIEVER_TOP_N]
    bm25_doc_ids = [f"chunk_{i}" for i in top_bm25_indices]

    fused_ids = reciprocal_rank_fusion(vector_doc_ids, bm25_doc_ids, k=RRF_K)
    
    return [all_chunks[doc_id] for doc_id in fused_ids[:RETRIEVER_TOP_N]]

async def compress_context_async(question: str, context_chunks: List[str]) -> str:
    context_str = "\n---\n".join(context_chunks)
    prompt = f"""
Given the following question and a set of context documents, extract all sentences from the documents that are directly relevant to answering the question. If no sentences are relevant, respond with an empty string.

<question>
{question}
</question>

<documents>
{context_str}
</documents>

Relevant sentences:
"""
    try:
        resp = await openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during context compression: {e}")
        return context_str

async def generate_answer_async(question: str, context: str) -> str:
    prompt = f"""
You are an expert Question-Answering assistant. Your task is to answer the user's question with high accuracy.
Follow these instructions precisely:
1.  Base your answer *only* on the information contained within the provided <context> documents. Do not use any external knowledge.
2.  If the answer cannot be found in the provided context, you must respond with the exact phrase: "Based on the information provided, an answer cannot be determined."
3.  Be concise and do not repeat long passages from the context verbatim. Synthesize the information.

<context>
{context}
</context>

<question>
{question}
</question>

Answer:
"""
    try:
        resp = await openai_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during answer generation: {e}")
        return "Error: Could not generate an answer due to an internal issue."

@api_router.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_rag_pipeline(request: RunRequest):
    text = await download_and_extract_pdf_text(request.documents)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Document is empty or unreadable.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    all_chunks_dict = dict(zip(chunk_ids, chunks))

    collection_name = f"rag_{os.urandom(8).hex()}"
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )
    
    await asyncio.to_thread(
        collection.add,
        ids=chunk_ids,
        documents=chunks
    )
    
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    async def process_question(q: str) -> str:
        retrieved_chunks = await hybrid_search(q, collection, bm25_index, all_chunks_dict)
        
        compressed_context = await compress_context_async(q, retrieved_chunks)
        if not compressed_context.strip():
            return "Based on the information provided, an answer cannot be determined."

        return await generate_answer_async(q, compressed_context)

    try:
        answers = await asyncio.gather(*(process_question(q) for q in request.questions))
        return RunResponse(answers=answers)
    finally:
        chroma_client.delete_collection(name=collection_name)

app.include_router(api_router)
