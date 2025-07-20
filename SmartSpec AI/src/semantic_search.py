# src/semantic_search.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the embedding model (e.g., MiniLM)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize index and chunk mapping
index = None
chunks = []

def embed_text(text):
    """Embed a single text (query or chunk)."""
    embedding = embedder.encode([text])[0]
    return embedding.astype(np.float32)

def build_index(text_chunks):
    """Embed all chunks and build FAISS index."""
    global index, chunks
    embeddings = [embed_text(chunk) for chunk in text_chunks]
    dimension = embeddings[0].shape[0]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    chunks = text_chunks  # Save mapping

def search(query_text, top_k=3):
    """Search most similar chunks to the query."""
    if index is None:
        raise ValueError("Index not built yet.")

    query_vec = embed_text(query_text).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i in indices[0]:
        results.append(chunks[i])
    return results
