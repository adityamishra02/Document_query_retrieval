import os
import requests
import pickle
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load OpenAI API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load MiniLM model
print("[*] Loading local MiniLM model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def download_pdf(url):
    response = requests.get(url)
    filename = "temp.pdf"
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text_into_chunks(text, max_chunk_size=500):
    words = text.split()
    chunks = []
    chunk = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > max_chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []
            current_len = 0
        chunk.append(word)
        current_len += len(word) + 1

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def get_embeddings(text_list, model):
    return model.encode(text_list, convert_to_numpy=True)

def build_faiss_index(chunks, model):
    print("[*] Creating FAISS index...")
    vectors = get_embeddings(chunks, model)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, vectors

def find_similar_chunks(query, chunks, index, model, top_k=5):
    query_vector = get_embeddings([query], model)
    _, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0]]

def get_answer_from_openai(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

def run_pipeline(pdf_url, questions):
    print("[*] Downloading and processing document...")
    path = download_pdf(pdf_url)
    text = extract_text_from_pdf(path)
    chunks = split_text_into_chunks(text)
    index, _ = build_faiss_index(chunks, model)

    results = {}
    for question in questions:
        relevant_chunks = find_similar_chunks(question, chunks, index, model)
        context = "\n".join(relevant_chunks)
        answer = get_answer_from_openai(question, context)
        results[question] = answer

    return results
