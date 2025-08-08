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

# --- Configuration and Initialization ---

# Load environment variables from .env file
load_dotenv()

# Securely fetch API keys and tokens from environment variables
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

# Global clients and models to be initialized at startup
http_client: httpx.AsyncClient | None = None
openai_client: AsyncOpenAI | None = None
cross_encoder: CrossEncoder | None = None
security = HTTPBearer()

# --- Lifespan Management (Startup and Shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. Initializes resources on startup
    and cleans them up on shutdown.
    """
    global http_client, openai_client, cross_encoder
    print("Loading models and initializing clients...")
    # Initialize a single, reusable HTTP client for the application's lifetime
    http_client = httpx.AsyncClient(timeout=180.0, follow_redirects=True)
    # Initialize the asynchronous OpenAI client
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    # Load the CrossEncoder model for reranking. It will use GPU if available.
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("Initialization complete.")
    yield
    # Gracefully close the HTTP client on shutdown
    await http_client.aclose()
    print("Clients closed.")

# --- FastAPI Application Setup ---

app = FastAPI(
    title="Optimized RAG API",
    description="A high-performance RAG system implementing advanced strategies for accuracy and speed.",
    lifespan=lifespan
)
api_router = APIRouter(prefix="/api/v1")

# --- Security and Authentication ---

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the provided static bearer token against the environment variable.
    """
    if credentials.scheme != "Bearer" or credentials.credentials != STATIC_API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- Pydantic Models (Data Contracts) ---

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- Core RAG Logic ---

async def download_and_extract_pdf_text(url: str) -> str:
    """
    Asynchronously downloads a PDF from a URL and extracts its text content.
    """
    try:
        # Use the global httpx client to download the PDF content
        response = await http_client.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Use a temporary file to store the PDF content for processing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

            # Run the synchronous PyMuPDF text extraction in a separate thread
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

def create_chunks(text: str) -> Dict:
    """
    Splits text into parent and child chunks for a multi-layered retrieval strategy.
    """
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    parent_chunks = parent_splitter.split_text(text)
    child_chunks = []
    parent_child_map = {}

    for p_chunk in parent_chunks:
        _child_chunks = child_splitter.split_text(p_chunk)
        for c_chunk in _child_chunks:
            # Map each child chunk back to its parent chunk
            child_idx = len(child_chunks)
            child_chunks.append(c_chunk)
            parent_child_map[child_idx] = p_chunk

    return {
        "child_chunks": child_chunks,
        "parent_child_map": parent_child_map
    }

@lru_cache(maxsize=128)
def get_embedding_sync(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Synchronous function to get embeddings. Cached to avoid re-computing for the same text.
    NOTE: This uses a synchronous client, suitable for threading.
    """
    # Using a synchronous client here because this function is run in a thread.
    sync_client = OpenAI(api_key=OPENAI_API_KEY)
    # Corrected to access the embedding from the first item in the data list.
    return sync_client.embeddings.create(input=[text], model=model).data[0].embedding

async def embed_content_async(content: List[str]) -> List[List[float]]:
    """
    Asynchronously embeds a list of text content by running sync calls in parallel threads.
    """
    tasks = [asyncio.to_thread(get_embedding_sync, item) for item in content]
    embeddings = await asyncio.gather(*tasks)
    return embeddings

def reciprocal_rank_fusion(results: List[List[int]], k: int = 60) -> Dict[int, float]:
    """
    Fuses multiple ranked lists of document indices into a single ranked list.
    """
    fused_scores = {}
    for doc_scores in results:
        for rank, doc_id in enumerate(doc_scores):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            # Add the reciprocal rank score
            fused_scores[doc_id] += 1 / (rank + k)

    # Sort the fused scores in descending order to get the final ranking
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
    """
    Performs hybrid search by combining vector search and BM25 keyword search.
    """
    # 1. Vector Search
    query_embedding_list = await embed_content_async([query])
    query_embedding = np.array(query_embedding_list)
    # Corrected to handle the 2D output of cosine_similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    vector_results = np.argsort(similarities)[::-1][:top_k].tolist()

    # 2. Keyword Search (BM25)
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    keyword_results = np.argsort(bm25_scores)[::-1][:top_k].tolist()

    # 3. Fuse results using RRF
    fused_results = reciprocal_rank_fusion([vector_results, keyword_results])

    return list(fused_results.keys())[:top_k]

def rerank_with_cross_encoder(query: str, chunks: List[str]) -> List[str]:
    """
    Reranks a list of chunks based on their relevance to a query using a CrossEncoder model.
    """
    sentence_pairs = [[query, chunk] for chunk in chunks]
    # The predict method is computationally intensive, a key area for GPU acceleration.
    scores = cross_encoder.predict(sentence_pairs)

    # Sort chunks by their reranked scores in descending order
    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    return ranked_chunks

async def generate_answer_async(question: str, context: str) -> str:
    """
    Generates an answer using an OpenAI chat model based on the provided context and question.
    """
    system_prompt = """You are a precision-focused AI assistant. Your most important task is to answer the user's question with the highest possible accuracy, based *only* on the provided document excerpts. Answer like a human in single proper sentence.

- **Accuracy is the top priority.** You must include all critical details, conditions, exceptions, and specific numbers (like bed counts, percentages, or monetary limits).
- **Synthesize, do not just copy.** Create a clear and comprehensive answer. 
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
        # Use the global async OpenAI client
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # Set to 0 for deterministic, fact-based answers
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during answer generation with OpenAI: {e}")
        return f"Error generating answer: {str(e)}"

# --- API Endpoint ---

@api_router.post("/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_token)])
async def run_rag_pipeline(request: RunRequest):
    """
    The main RAG pipeline endpoint.
    """
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
    # Embed all child chunks at once
    chunk_embeddings_list = await embed_content_async(child_chunks)
    chunk_embeddings = np.array(chunk_embeddings_list)

    # Create the BM25 index from the tokenized chunks
    tokenized_chunks = [chunk.lower().split() for chunk in child_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    print("Step 4: Processing all questions concurrently...")

    async def process_question(question: str):
        """Helper function to process a single question through the RAG pipeline."""
        print(f"  - Processing question: '{question[:50]}...'")

        # Retrieve candidate chunks using hybrid search
        candidate_indices = await hybrid_search(
            question, child_chunks, chunk_embeddings, bm25_index, top_k=20
        )
        candidate_chunks = [child_chunks[i] for i in candidate_indices]

        # Rerank the candidates for higher relevance
        reranked_child_chunks = rerank_with_cross_encoder(question, candidate_chunks)

        # Select the top 10 most relevant child chunks after reranking
        final_top_k = 10
        top_child_chunks = reranked_child_chunks[:final_top_k]

        # Find the original indices of these top child chunks
        top_child_indices = [child_chunks.index(c) for c in top_child_chunks]

        # Retrieve the corresponding parent chunks to build a rich context
        # Use dict.fromkeys to get unique parent chunks in order
        relevant_parent_chunks = list(dict.fromkeys([parent_child_map[i] for i in top_child_indices]))

        # Join parent chunks to form the final context
        context = "\n---\n".join(relevant_parent_chunks)

        # Generate the final answer
        return await generate_answer_async(question, context)

    # Run the processing for all questions in parallel
    tasks = [process_question(q) for q in request.questions]
    answers = await asyncio.gather(*tasks)

    print("Step 5: All questions processed.")
    return RunResponse(answers=answers)

# Include the API router in the main FastAPI application
app.include_router(api_router)