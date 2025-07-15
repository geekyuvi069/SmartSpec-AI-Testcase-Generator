import os
import torch
import json
from src.encoder.encoder import Encoder
from src.decoder.decoder import SimpleTransformerDecoder
from src.utils.text_preprocessor import clean_text, split_into_chunks
from src.utils.pdf_extractor import extract_text_from_pdf

# ===== Load vocabulary from JSON =====
def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ===== Convert input text to token sequence =====
def text_to_sequence(text, vocab, max_len=50):
    tokens = [vocab.get(word, vocab.get('<UNK>', 1)) for word in text.lower().split()]
    if len(tokens) < max_len:
        tokens += [vocab.get('<PAD>', 0)] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_len)

# ===== Convert token sequence back to text =====
def sequence_to_text(seq, inv_vocab):
    return ' '.join([inv_vocab.get(idx, '<UNK>') for idx in seq if idx != 0])

# ===== Load model weights from checkpoint files =====
def load_model_weights(encoder, decoder, encoder_ckpt=None, decoder_ckpt=None):
    if encoder and encoder_ckpt and os.path.exists(encoder_ckpt):
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location='cpu'))
        print(f"✅ Encoder weights loaded from {encoder_ckpt}")
    else:
        print("⚠️ Skipping encoder checkpoint (not provided or file missing).")

    if decoder and decoder_ckpt and os.path.exists(decoder_ckpt):
        decoder.load_state_dict(torch.load(decoder_ckpt, map_location='cpu'))
        print(f"✅ Decoder weights loaded from {decoder_ckpt}")
    else:
        print("⚠️ Decoder checkpoint not found.")

# ===== Generate test case from query using encoder-decoder =====
def answer_query(query, encoder, decoder, vocab, inv_vocab, device, max_len=50):
    query_seq = text_to_sequence(query, vocab, max_len).to(device)
    mask = torch.ones(1, max_len).to(device)

    with torch.no_grad():
        if encoder:
            memory = encoder(query_seq.float(), mask)
        else:
            memory = torch.zeros(1, max_len, 512).to(device)

        tgt_seq = torch.tensor([[vocab.get('<PAD>', 0)]], dtype=torch.long).to(device)
        outputs = []

        for _ in range(max_len):
            logits = decoder(tgt_seq, memory)  # logits: (1, tgt_seq_len, vocab_size)
            next_token = logits[:, -1, :].argmax(-1).item()

            # ==== Stop if predicted token is <PAD> or <UNK> ====
            if next_token in [vocab.get('<PAD>', 0), vocab.get('<UNK>', 1)]:
                break

            outputs.append(next_token)
            tgt_seq = torch.cat([tgt_seq, torch.tensor([[next_token]], device=device)], dim=1)

        return sequence_to_text(outputs, inv_vocab)

# ===== Run interactive test case generation =====
def run_query_answering(encoder_ckpt=None, decoder_ckpt=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Default paths ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(base_dir, 'data', 'vocab.json')
    default_encoder_ckpt = os.path.join(base_dir, 'encoder_checkpoints', 'encoder.pth')
    default_decoder_ckpt = os.path.join(base_dir, 'decoder_checkpoints', 'decoder.pth')

    encoder_ckpt = encoder_ckpt or default_encoder_ckpt
    decoder_ckpt = decoder_ckpt or default_decoder_ckpt

    # === Load vocab and inverse vocab ===
    vocab = load_vocab(vocab_path)
    inv_vocab = {idx: word for word, idx in vocab.items()}  # Use int keys
    vocab_size = len(vocab)

    # === Load models ===
    encoder = Encoder(d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1).to(device) if os.path.exists(encoder_ckpt) else None
    decoder = SimpleTransformerDecoder(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=50
    ).to(device)

    load_model_weights(encoder, decoder, encoder_ckpt, decoder_ckpt)

    print("\n💡 Type your query to generate test case (or type 'exit' to quit):")
    while True:
        query = input("\n📝 Query: ").strip()
        if query.lower() == "exit":
            print("👋 Exiting.")
            break
        if not query:
            print("⚠️ Empty query. Please try again.")
            continue
        answer = answer_query(query, encoder, decoder, vocab, inv_vocab, device)
        print("✅ Test Case:", answer)
