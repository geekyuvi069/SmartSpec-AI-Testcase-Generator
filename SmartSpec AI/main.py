import os
from datetime import datetime

from src.preprocessing import extract_text_from_pdf, clean_text, split_into_chunks
from src.vocabulary import Vocabulary
from src.encoder import TransformerEncoder
from src.decoder import TransformerDecoder

DATA_FOLDER = "data"

def get_latest_uploaded_file():
    """Get the most recent PDF file in data/ subfolders."""
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"'{DATA_FOLDER}' directory does not exist.")

    # List all subfolders
    subfolders = [
        os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER)
        if os.path.isdir(os.path.join(DATA_FOLDER, f))
    ]

    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in '{DATA_FOLDER}'.")

    # Sort subfolders by date (descending)
    subfolders.sort(reverse=True)

    # Look for PDFs in each subfolder
    for folder in subfolders:
        pdf_files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".pdf")
        ]
        if pdf_files:
            # Return the newest file
            pdf_files.sort(reverse=True)
            return pdf_files[0]

    raise FileNotFoundError(f"No PDF files found in '{DATA_FOLDER}'. Please upload a file first.")

def main():
    print("=== SmartSpec AI - Document Processing ===")

    try:
        pdf_path = get_latest_uploaded_file()
    except FileNotFoundError as e:
        print(e)
        print("⚠️  No PDF found. Please upload via the web interface first.")
        return

    print(f"✅ Using file: {pdf_path}")

    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    print("✅ Extracted raw text.")

    # Clean text
    cleaned_text = clean_text(raw_text)
    print("✅ Cleaned text.")

    # Split into chunks
    chunks = split_into_chunks(cleaned_text)
    print(f"✅ Split into {len(chunks)} chunks.")

    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(chunks)
    print(f"✅ Vocabulary built with {len(vocab.word_to_id)} tokens.")

    # Initialize encoder and decoder
    vocab_size = len(vocab.word_to_id)
    embed_size = 256
    num_heads = 8
    num_layers = 6
    encoder = TransformerEncoder(vocab_size, embed_size, num_heads, num_layers)
    decoder = TransformerDecoder(vocab_size, embed_size, num_heads, num_layers)
    print("✅ Transformer encoder and decoder initialized.")

    # Display chunks (or do other processing)
    print("\n=== First Chunk Preview ===\n")
    if chunks:
        print(chunks[0][:500])
    else:
        print("No chunks found.")

    print("\n✅ Processing complete.\n")

if __name__ == "__main__":
    main()
