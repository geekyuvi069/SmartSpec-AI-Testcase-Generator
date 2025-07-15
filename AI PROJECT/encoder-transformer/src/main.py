import sys
import os
import torch

# ===== Add Project Root to sys.path =====
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ===== Imports from src =====
from src.encoder.encoder import Encoder
from src.utils.pdf_extractor import extract_text_from_pdf
from src.utils.text_preprocessor import clean_text, split_into_chunks
from src.decoder.train_decoder import train_decoder_main
from src.query_answering import run_query_answering  # <-- Added for interactive inference

# ===== Main Function =====
def main():
    # ------------------ Hyperparameters ------------------
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    seq_len = 10
    batch_size = 2

    # ------------------ PDF Extraction ------------------
    pdf_path = r'C:\Users\ADMIN\Desktop\AI PROJECT\encoder-transformer\src\document\srs_example_2010_group2.pdf'
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    chunks = split_into_chunks(cleaned_text, chunk_size=500)

    print(f"✅ Number of document chunks extracted: {len(chunks)}")

    # ------------------ Encoder Demo (Optional) ------------------
    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len)
    encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)
    output = encoder(x, mask)
    print("✅ Encoder demo output shape:", output.shape)

    # ------------------ Train the Decoder ------------------
    print("\n🚀 Starting Decoder Training...\n")
    train_decoder_main()
    print("\n✅ Decoder Training Completed!\n")

    # ------------------ Run Offline Query-Answering ------------------
    print("\n🤖 Launching Offline AI Test Case Generator...\n")

    # Pass None for encoder checkpoint to avoid loading error
    encoder_ckpt = None  # <-- This ensures no missing file error
    # Path to the trained decoder
decoder_ckpt = 'src/decoder_checkpoints/decoder.pth'

   # If you're not using encoder checkpoint, pass None
run_query_answering(encoder_ckpt=None, decoder_ckpt=decoder_ckpt)

# ===== Entry Point =====
if __name__ == "__main__":
    main()
