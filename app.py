import os
import asyncio
import httpx
import pdfplumber
import tempfile
import numpy as np
import google.generativeai as genai

from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Google AI: {e}. Ensure GOOGLE_API_KEY is set.")

http_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)
api_router = APIRouter(prefix="/api/v1")

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

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

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), max_tokens - overlap):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
    return chunks

async def embed_content_async(content: List[str], task_type: str, title: str = "Document") -> List[List[float]]:
    def embed_sync():
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=content,
                task_type=task_type,
                title=title if task_type == "retrieval_document" else None
            )
            return response["embedding"]
        except Exception as e:
            print(f"Error embedding content: {e}")
            return [[0.0] * 768] * len(content)
    return await asyncio.to_thread(embed_sync)

async def find_relevant_chunks(question: str, chunks: List[str], chunk_embeddings: np.ndarray, top_k: int = 5) -> List[str]:
    actual_top_k = min(top_k, len(chunks))
    if actual_top_k == 0:
        return []
    question_embedding_list = await embed_content_async([question], task_type="retrieval_query")
    if not question_embedding_list or not any(question_embedding_list[0]):
        return chunks[:actual_top_k]
    question_embedding = np.array(question_embedding_list[0])
    similarities = cosine_similarity(question_embedding.reshape(1, -1), chunk_embeddings)[0]
    top_indices = np.argpartition(similarities, -actual_top_k)[-actual_top_k:]
    sorted_top_indices = sorted(top_indices, key=lambda i: similarities[i], reverse=True)
    return [chunks[i] for i in sorted_top_indices]

async def generate_answer_async(question: str, context: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
You are an expert question-answering assistant. Your answer must be based *only* on the information in the provided document excerpts.
Be accurate and concise. Answer in 1-2 sentences. Do not use markdown or other text styling.

DOCUMENT EXCERPTS:
\"\"\"
{context}
\"\"\"

QUESTION: {question}

Answer the question using only the document excerpts above. If the answer is not in the document, state that clearly.
"""
    try:
        def generate_sync():
            response = model.generate_content(prompt)
            return response.text.strip()
        return await asyncio.to_thread(generate_sync)
    except Exception as e:
        print(f"Error during answer generation: {str(e)}")
        return f"Error generating answer: {str(e)}"

@api_router.post("/hackrx/run", response_model=RunResponse)
async def run_hackrx(request: RunRequest):
    document_text = await download_and_extract_pdf_text(request.documents)
    if not document_text:
        raise HTTPException(status_code=400, detail="Could not extract any text from the PDF.")
    
    chunks = chunk_text(document_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document is too short to be processed.")

    chunk_embeddings_list = await embed_content_async(chunks, task_type="retrieval_document")
    chunk_embeddings = np.array(chunk_embeddings_list)
    
    async def process_question(question: str):
        candidate_chunks = await find_relevant_chunks(question, chunks, chunk_embeddings, top_k=10)
        relevant_context = "\n---\n".join(candidate_chunks[:5])
        final_answer = await generate_answer_async(question, relevant_context)
        return final_answer

    tasks = [process_question(q) for q in request.questions]
    answers = await asyncio.gather(*tasks)
    
    return RunResponse(answers=answers)

app.include_router(api_router)
