import faiss

def save_index(index, path="faiss_index.pkl"):
    faiss.write_index(index, path)

def load_index(path="faiss_index.pkl"):
    return faiss.read_index(path)
