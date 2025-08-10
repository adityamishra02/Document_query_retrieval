import os
import asyncio
import httpx
import fitz
import tempfile
from contextlib import asynccontextmanager
from typing import List, Dict

from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, APIRouter, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

# --- Environment and Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STATIC_API_TOKEN = os.getenv("STATIC_API_TOKEN")

if not OPENAI_API_KEY or not STATIC_API_TOKEN:
    raise RuntimeError("API keys (OPENAI_API_KEY, STATIC_API_TOKEN) are missing from.env file")

# --- Model and RAG Pipeline Configuration ---
# Using state-of-the-art models for maximum performance as per research findings.
# GPT-4o is selected for its superior speed, cost-effectiveness, and reasoning. [4]
# text-embedding-3-large is chosen for its peak semantic representation capabilities. [2, 3]
GPT_GENERATION_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536 # Balances performance and cost, can be tuned. [2]

# Advanced chunking parameters for Parent-Child Retriever Strategy. [1, 9]
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

# Contextual compression parameters
# The retriever will fetch a larger number of documents initially.
RETRIEVER_K = 20
# The compressor will filter down to the most relevant documents.
COMPRESSOR_K = 5
# Similarity threshold for the embeddings filter.
SIMILARITY_THRESHOLD = 0.78

# --- Advanced Prompt Template ---
# This structured prompt is engineered to maximize LLM accuracy by providing clear roles,
# instructions, and context structure, while minimizing hallucinations. [6, 7, 8, 10]
ADVANCED_PROMPT_TEMPLATE = """
You are a world-class expert research analyst. Your task is to answer the user's question with precision and clarity, based *only* on the information contained within the provided documents.

**Instructions:**
1.  **Analyze the Documents:** Carefully read and analyze the content of the documents provided below.
2.  **Synthesize the Answer:** Formulate a comprehensive answer to the user's question. Your answer must be directly supported by the information in the documents.
3.  **Cite Your Sources:** For each statement you make, you must cite the document index it came from. Use the format `[doc-X]` where X is the document number.
4.  **Think Step-by-Step:** First, identify the key facts and information from the documents relevant to the question. Second, construct your answer based on these facts. This ensures accuracy and adherence to the provided context.
5.  **Handle Missing Information:** If the answer cannot be found within the provided documents, you must respond with the exact phrase: "Based on the information provided, an answer cannot bedetermined." Do not use any external knowledge or make assumptions.

**Provided Documents:**
<documents>
{context}
</documents>

**User's Question:**
{question}

**Your Answer:**
"""

# --- FastAPI Application Setup ---
http_client: httpx.AsyncClient | None = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=120.0)
    yield
    await http_client.aclose()

app = FastAPI(title="Optimized RAG Pipeline", lifespan=lifespan)
api_router = APIRouter(prefix="/api/v1")

# --- Pydantic Models ---
class RunRequest(BaseModel):
    documents: str = Field(..., description="URL of the PDF document to process.")
    questions: List[str] = Field(..., description="List of questions to answer based on the document.")

class RunResponse(BaseModel):
    answers: List[str]

# --- Security ---
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme!= "Bearer" or credentials.credentials!= STATIC_API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# --- Core Logic ---
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        with fitz.open(file_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        # Log the error for debugging
        print(f"Error opening or parsing PDF: {e}")
        raise

async def download_and_extract_pdf_text(url: str) -> str:
    """Downloads a PDF from a URL and extracts its text content."""
    try:
        response = await http_client.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            file_path = tmp_file.name

        # Use asyncio.to_thread for synchronous file I/O and parsing
        text = await asyncio.to_thread(extract_text_from_pdf, file_path)
        os.unlink(file_path) # Clean up the temporary file
        return text
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"PDF download failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")

async def generate_answer_async(question: str, context: str, llm: ChatOpenAI) -> str:
    """Generates an answer using the LLM with the advanced prompt."""
    prompt = ADVANCED_PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        resp = await llm.ainvoke(prompt)
        return resp.content.strip()
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return f"Error generating answer: {e}"

@api_router.post("/hackrx/run", response_model=RunResponse, dependencies=)
async def run_rag_pipeline(request: RunRequest):
    """
    Executes the full, optimized RAG pipeline.
    This pipeline incorporates advanced chunking, state-of-the-art models,
    and contextual compression to achieve maximum performance.
    """
    # 1. Document Ingestion
    text = await download_and_extract_pdf_text(request.documents)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Document is empty or unreadable.")

    # 2. Instantiate Models
    # Using the most powerful embedding and generation models available.
    embeddings_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS
    )
    llm = ChatOpenAI(model=GPT_GENERATION_MODEL, temperature=0)

    # 3. Advanced Chunking & Retrieval Setup: Parent-Child Strategy
    # This strategy balances retrieval precision with contextual richness. [1]
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    
    # The vector store holds the small, precise "child" chunks.
    vectorstore = Chroma(
        collection_name="parent_child_retrieval",
        embedding_function=embeddings_model
    )
    # The document store holds the large, context-rich "parent" chunks.
    docstore = InMemoryStore()

    base_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": RETRIEVER_K}
    )
    
    # Index the document content
    docs =
    await asyncio.to_thread(base_retriever.add_documents, docs, ids=None)

    # 4. Contextual Compression Setup
    # This step refines the retrieved documents, filtering out noise and keeping
    # only the most relevant information for the LLM. [5]
    compressor = EmbeddingsFilter(
        embeddings=embeddings_model,
        similarity_threshold=SIMILARITY_THRESHOLD,
        k=COMPRESSOR_K
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 5. Process Questions Concurrently
    async def process_question(q: str) -> str:
        # Retrieve and compress documents for the given question.
        retrieved_docs = await compression_retriever.aget_relevant_documents(q)
        
        # Format the context for the prompt, adding index numbers for citation.
        context_str = "\n\n".join(
            [f"<document index='{i+1}'>\n{doc.page_content}\n</document>" for i, doc in enumerate(retrieved_docs)]
        )
        
        return await generate_answer_async(q, context_str, llm)

    answers = await asyncio.gather(*(process_question(q) for q in request.questions))
    
    # Clean up the vector store after processing the request
    await asyncio.to_thread(vectorstore.delete_collection)

    return RunResponse(answers=answers)

app.include_router(api_router)
