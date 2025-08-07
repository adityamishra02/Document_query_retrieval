import os
import asyncio
import httpx
import pdfplumber
import tempfile
import numpy as np
import google.generativeai as genai

from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    STATIC_API_TOKEN = os.getenv("STATIC_API_TOKEN")
    if not STATIC_API_TOKEN:
        raise ValueError("STATIC_API_TOKEN environment variable not set.")
except Exception as e:
    print(f"Error during initial configuration: {e}")

# --- Global Objects & Lifespan ---
http_client = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    # Increased timeout for potentially slower, more accurate model
    http_client = httpx.AsyncClient(timeout=120.0, follow_redirects=True)
    yield
    await http_client.aclose()

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)
api_router = APIRouter(prefix="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication ---
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != STATIC_API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- Pydantic Models ---
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- Core RAG Functions ---
async def download_and_extract_pdf_text(url: str) -> str:
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
            def extract_text_sync():
                text = ""
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text.strip()
            return await asyncio.to_thread(extract_text_sync)
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Extraction Failed: {str(e)}")

def chunk_text(text: str, max_tokens: int = 450, overlap: int = 100) -> List[str]:
    # ACCURACY TUNE: Increased overlap to better preserve context between chunks
    words = text.split()
    if not words: return []
    chunks = []
    for i in range(0, len(words), max_tokens - overlap):
        chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks

async def embed_content_async(content: List[str], task_type: str, title: str = "Document") -> List[List[float]]:
    def embed_sync():
        try:
            response = genai.embed_content(
                model="models/embedding-001", content=content, task_type=task_type,
                title=title if task_type == "retrieval_document" else None
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error embedding content: {e}")
            return [[0.0] * 768] * len(content)
    return await asyncio.to_thread(embed_sync)

def find_relevant_chunks_from_embedding(embedding: np.ndarray, chunks: List[str], chunk_embeddings: np.ndarray, top_k: int) -> List[str]:
    actual_top_k = min(top_k, len(chunks))
    if actual_top_k == 0: return []
    similarities = cosine_similarity(embedding.reshape(1, -1), chunk_embeddings)[0]
    top_indices = np.argpartition(similarities, -actual_top_k)[-actual_top_k:]
    sorted_indices = sorted(top_indices, key=lambda i: similarities[i], reverse=True)
    return [chunks[i] for i in sorted_indices]

async def generate_answer_async(question: str, context: str) -> str:
    # ACCURACY TUNE: Using a more powerful model for better reasoning
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # ACCURACY TUNE: More explicit and forceful prompt
    prompt = f"""You are a highly analytical assistant. Your task is to answer the user's question based *exclusively* on the provided "DOCUMENT EXCERPTS".
Do not use any outside knowledge. Reply like a human in a single proper sentence.

DOCUMENT EXCERPTS:
---
{context}
---

QUESTION: {question}

Synthesize a concise and direct answer from the excerpts.
"""
    try:
        def generate_sync():
            return model.generate_content(prompt).text.strip()
        return await asyncio.to_thread(generate_sync)
    except Exception as e:
        print(f"Error during answer generation: {str(e)}")
        return f"Error generating answer: {str(e)}"

# --- API Endpoints ---
@api_router.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_hackrx(request: RunRequest):
    document_text = await download_and_extract_pdf_text(request.documents)
    if not document_text: raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")
    
    chunks = chunk_text(document_text)
    if not chunks: raise HTTPException(status_code=400, detail="Document is too short to be processed.")

    # Run embedding tasks in parallel for efficiency
    chunk_embeddings_list, question_embeddings_list = await asyncio.gather(
        embed_content_async(chunks, task_type="retrieval_document"),
        embed_content_async(request.questions, task_type="retrieval_query")
    )
    chunk_embeddings = np.array(chunk_embeddings_list)
    question_embeddings = np.array(question_embeddings_list)
    
    async def process_question(idx: int):
        # ACCURACY TUNE: Retrieve more chunks to create a richer context
        candidate_chunks = find_relevant_chunks_from_embedding(question_embeddings[idx], chunks, chunk_embeddings, top_k=15)
        # ACCURACY TUNE: Use more of the retrieved chunks in the final context
        relevant_context = "\n---\n".join(candidate_chunks[:7])
        return await generate_answer_async(request.questions[idx], relevant_context)

    tasks = [process_question(i) for i in range(len(request.questions))]
    answers = await asyncio.gather(*tasks)
    
    return RunResponse(answers=answers)

app.include_router(api_router)

@app.get("/", response_class=FileResponse)
async def read_index():
    # This serves your index.html file
    return "index.html"
