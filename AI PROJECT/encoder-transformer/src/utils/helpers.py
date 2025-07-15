
import faiss
import numpy as np
import torch
import os
from src.encoder.encoder import Encoder  # Using your own Transformer Encoder

# Your existing function (keep as is)
def create_padding_mask(seq): 
    return (seq != 0).unsqueeze(1).unsqueeze(2)

# 1️⃣ Load FAISS Index and ID Map
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    id_map_path = index_path.replace('.index', '_id_map.txt')
    id_to_chunk = {}
    if os.path.exists(id_map_path):
        with open(id_map_path, 'r', encoding='utf-8') as f:
            for line in f:
                idx, chunk = line.strip().split('\t', 1)
                id_to_chunk[int(idx)] = chunk
    else:
        print(f"[WARNING] No ID map found at {id_map_path}.")
    return index, id_to_chunk

# 2️⃣ Generate Embeddings using YOUR Transformer Encoder (no BERT)
def generate_embeddings(chunks, encoder, device, tokenizer_func, seq_len=10, d_model=512):
    tokenized_inputs = []
    for text in chunks:
        token_ids = tokenizer_func(text, max_length=seq_len)
        tokenized_inputs.append(token_ids)

    input_tensor = torch.tensor(tokenized_inputs).to(device)  # (batch_size, seq_len)
    input_tensor = input_tensor.unsqueeze(-1).repeat(1, 1, d_model)  # Expand to (batch_size, seq_len, d_model)
    
    mask = torch.ones(input_tensor.shape[0], seq_len).to(device)
    with torch.no_grad():
        embeddings = encoder(input_tensor, mask)
    mean_embeddings = embeddings.mean(dim=1).cpu().numpy().astype('float32')  # (batch_size, d_model)
    return mean_embeddings

# 3️⃣ Simple Tokenizer (No pretrained tokenizer—basic split)
def tokenize_input(text, max_length=10):
    tokens = text.lower().split()[:max_length]
    token_ids = [hash(w) % 10000 for w in tokens]  # Simple hash-based token IDs
    # Padding if needed
    if len(token_ids) < max_length:
        token_ids += [0] * (max_length - len(token_ids))
    return token_ids
