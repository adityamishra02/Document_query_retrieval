import os
import asyncio
import httpx
import fitz
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

load_dotenv()
try:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    STATIC_API_TOKEN = os.getenv("STATIC_API_TOKEN")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    if not STATIC_API_TOKEN:
        raise ValueError("STATIC_API_TOKEN environment variable not set.")
except ValueError as e:
    print(f"Configuration Error: {e}")
    exit(1)

http_client: httpx.AsyncClient | None = None
openai_client: AsyncOpenAI | None = None
cross_encoder: CrossEncoder | None = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, openai_client, cross_encoder
    print("Loading models and initializing clients...")
    http_client = httpx.AsyncClient(timeout=180.0, follow_redirects=True)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("Initialization complete.")
    yield
    await http_client.aclose()
    print("Clients closed.")

app = FastAPI(
    title="Optimized RAG API",
    description="A high-performance RAG system implementing advanced strategies for accuracy and speed.",
    lifespan=lifespan
)
api_router = APIRouter(prefix="/api/v1")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != STATIC_API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

async def download_and_extract_pdf_text(url: str) -> str:
    try:
        async with http_client as client:
            response = await client.get(url)
            response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
            
            def extract_text_sync():
                text = ""
                with fitz.open(tmp_path) as doc:
                    for page in doc:
                        text += page.get_text() + "\n"
                return text.strip()

            return await asyncio.to_thread(extract_text_sync)
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Extraction Failed: {str(e)}")

def create_chunks(text: str) -> Dict[str, List[str]]:
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    parent_chunks = parent_splitter.split_text(text)
    child_chunks = []
    parent_child_map = {}

    for i, p_chunk in enumerate(parent_chunks):
        _child_chunks = child_splitter.split_text(p_chunk)
        for c_chunk in _child_chunks:
            child_idx = len(child_chunks)
            child_chunks.append(c_chunk)
            parent_child_map[child_idx] = p_chunk
            
    return {
        "child_chunks": child_chunks,
        "parent_chunks": parent_chunks,
        "parent_child_map": parent_child_map
    }

@lru_cache(maxsize=128)
def get_embedding_sync(text: str, model: str = "text-embedding-3-small") -> List[float]:
    sync_client = OpenAI(api_key=OPENAI_API_KEY)
    return sync_client.embeddings.create(input=[text], model=model).data[0].embedding

async def embed_content_async(content: List[str]) -> List[List[float]]:
    tasks = [asyncio.to_thread(get_embedding_sync, item) for item in content]
    embeddings = await asyncio.gather(*tasks)
    return embeddings

def reciprocal_rank_fusion(results: List[List[int]], k: int = 60) -> Dict[int, float]:
    fused_scores = {}
    for doc_scores in results:
        for rank, doc_id in enumerate(doc_scores):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (rank + k)
    
    reranked_results = {
        doc_id: score for doc_id, score in sorted(
            fused_scores.items(), key=lambda item: item[1], reverse=True
        )
    }
    return reranked_results

async def hybrid_search(
    query: str,
    child_chunks: List[str],
    chunk_embeddings: np.ndarray,
    bm25_index: BM25Okapi,
    top_k: int
) -> List[int]:
    query_embedding_list = await embed_content_async([query])
    query_embedding = np.array(query_embedding_list)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    vector_results = np.argsort(similarities)[::-1][:top_k].tolist()

    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    keyword_results = np.argsort(bm25_scores)[::-1][:top_k].tolist()
    
    fused_results = reciprocal_rank_fusion([vector_results, keyword_results])
    
    return list(fused_results.keys())[:top_k]

def rerank_with_cross_encoder(query: str, chunks: List[str]) -> List[str]:
    sentence_pairs = [[query, chunk] for chunk in chunks]
    scores = cross_encoder.predict(sentence_pairs)
    
    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    return ranked_chunks

async def generate_answer_async(question: str, context: str) -> str:
    system_prompt = """You are a precision-focused AI assistant. Your most important task is to answer the user's question with the highest possible accuracy, based *only* on the provided document excerpts.

- **Accuracy is the top priority.** You must include all critical details, conditions, exceptions, and specific numbers (like bed counts, percentages, or monetary limits).
- **Synthesize, do not just copy.** Create a clear and comprehensive answer. You may use a single sentence or a short paragraph as needed to be both complete and readable.
- **If the answer is not in the text, you must state:** "Based on the provided text, that information is not available." Do not guess or infer information.
"""

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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during answer generation with OpenAI: {e}")
        return f"Error generating answer: {str(e)}"

@api_router.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_rag_pipeline(request: RunRequest):
    print("Step 1: Downloading and parsing document...")
    document_text = await download_and_extract_pdf_text(request.documents)
    if not document_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")
    
    print("Step 2: Creating parent and child chunks...")
    chunk_data = create_chunks(document_text)
    child_chunks = chunk_data["child_chunks"]
    parent_child_map = chunk_data["parent_child_map"]
    if not child_chunks:
        raise HTTPException(status_code=400, detail="Document is too short to be processed.")

    print("Step 3: Embedding child chunks and creating BM25 index...")
    chunk_embeddings_list, question_embeddings_list = await asyncio.gather(
        embed_content_async(child_chunks),
        embed_content_async(request.questions)
    )
    chunk_embeddings = np.array(chunk_embeddings_list)
    question_embeddings = np.array(question_embeddings_list)
    
    tokenized_chunks = [chunk.lower().split() for chunk in child_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    
    print("Step 4: Processing all questions concurrently...")
    
    async def process_question(idx: int):
        question = request.questions[idx]
        print(f"  - Processing question: '{question[:50]}...'")
        
        candidate_indices = await hybrid_search(
            question, child_chunks, chunk_embeddings, bm25_index, top_k=50
        )
        candidate_chunks = [child_chunks[i] for i in candidate_indices]
        
        reranked_child_chunks = rerank_with_cross_encoder(question, candidate_chunks)
        
        final_top_k = 10
        top_child_chunks = reranked_child_chunks[:final_top_k]
        
        top_child_indices = [child_chunks.index(c) for c in top_child_chunks]
        
        relevant_parent_chunks = list(dict.fromkeys([parent_child_map[i] for i in top_child_indices]))
        
        context = "\n---\n".join(relevant_parent_chunks)
        
        return await generate_answer_async(question, context)

    tasks = [process_question(i) for i in range(len(request.questions))]
    answers = await asyncio.gather(*tasks)
    
    print("Step 5: All questions processed.")
    return RunResponse(answers=answers)

app.include_router(api_router)
