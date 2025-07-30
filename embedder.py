import os
import requests
import pickle
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from text_splitter import split_text

# PDF URL
PDF_URL = "https://hackrx.in/policies/EDLHLGA23009V012223.pdf"

# Step 1: Download PDF
def download_pdf(url: str, filename: str = "document.pdf"):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    print("[*] PDF downloaded.")
    return filename

# Step 2: Extract text using PyMuPDF
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# Step 3: Generate embeddings
def get_embeddings(text_list):
    return model.encode(text_list, convert_to_numpy=True)

# Step 4: Build FAISS index
def build_faiss_index(text_chunks):
    print("[*] Creating FAISS index...")
    vectors = get_embeddings(text_chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

# ---- Main ---- #
if __name__ == "__main__":
    print("[*] Starting pipeline...")

    # Download and extract
    pdf_file = download_pdf(PDF_URL)
    raw_text = extract_text_from_pdf(pdf_file)

    # Split into chunks
    chunks = split_text(raw_text)

    # Load embedding model
    print("[*] Loading MiniLM model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Build FAISS
    index, vectors = build_faiss_index(chunks)

    # Save everything
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(index, f)
    with open("vectors.pkl", "wb") as f:
        pickle.dump(vectors, f)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("[*] All done! FAISS index saved.")
