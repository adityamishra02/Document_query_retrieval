from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

def embed_text(texts):
    embeddings = model.get_embeddings(texts)
    return [e.values for e in embeddings]
