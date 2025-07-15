import os
import torch
import faiss
import numpy as np
from src.encoder.encoder import Encoder, extract_text_from_pdf, preprocess_text, chunk_text, build_vocab, tokenize_chunk

# === 1. Load and Preprocess the Document ===
pdf_path = r'C:\Users\ADMIN\Desktop\AI PROJECT\TKPV DRDO.pdf'
raw_text = extract_text_from_pdf(pdf_path)
clean_text = preprocess_text(raw_text)
chunks = chunk_text(clean_text, max_chunk_size=100)  # Split into chunks

# === 2. Build Vocabulary from Chunks ===
vocab = build_vocab(chunks)  # You should already have this function
vocab_size = len(vocab) + 2

# === 3. Initialize Your Encoder ===
encoder = Encoder(d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1)

# === 4. Generate Embeddings for Each Chunk ===
embeddings = []
for chunk in chunks:
    token_ids = tokenize_chunk(chunk, vocab, max_len=100)  # Your helper function
    token_ids = token_ids.unsqueeze(0).unsqueeze(2).repeat(1, 1, 512).float()  # Convert to float
    
    with torch.no_grad():
        embedding = encoder(token_ids, None).mean(dim=1).squeeze()
    embeddings.append(embedding.numpy())

# === 5. Store Embeddings in FAISS Index ===
embedding_matrix = np.vstack(embeddings).astype('float32')
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)

# === 6. Save the FAISS Index Offline ===
os.makedirs('faiss_indexes', exist_ok=True)
faiss.write_index(index, 'faiss_indexes/document_index.index')

print(f"✅ FAISS index successfully saved with {len(embeddings)} document chunks!")
