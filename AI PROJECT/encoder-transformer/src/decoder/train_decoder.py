import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.decoder.decoder import SimpleTransformerDecoder


# === Dataset ===
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.input_sequences[idx], dtype=torch.long), torch.tensor(self.target_sequences[idx], dtype=torch.long)


# === Utilities ===
def build_vocab_from_json(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for item in data:
        for field in [item["chunk"], item["test_case"]]:
            for word in field.lower().split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
    return vocab


def text_to_sequence(text, vocab, max_len=50):
    tokens = [vocab.get(word, vocab["<UNK>"]) for word in text.lower().split()]
    if len(tokens) < max_len:
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


# === Training Loop ===
def train_decoder(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Shift targets for teacher forcing (decoder_input = targets[:, :-1], label = targets[:, 1:])
            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]

            optimizer.zero_grad()
            memory = model.embedding(inputs)  # Simulate encoder output (use actual encoder in full system)
            logits = model(decoder_input, memory)

            loss = criterion(logits.view(-1, logits.size(-1)), decoder_target.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


# === Main Function ===
def train_decoder_main():
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 16
    num_epochs = 10
    max_len = 50
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'training_data.json')

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab = build_vocab_from_json(data)
    vocab_size = len(vocab)

    input_sequences = [text_to_sequence(item["chunk"], vocab, max_len) for item in data]
    target_sequences = [text_to_sequence(item["test_case"], vocab, max_len + 1) for item in data]  # +1 for shifting

    dataset = TextDataset(input_sequences, target_sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleTransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=num_heads,
        num_layers=num_layers,
        dim_feedforward=d_ff,
        dropout=dropout,
        max_seq_len=max_len + 1
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_decoder(model, dataloader, criterion, optimizer, device, num_epochs)

    # Save decoder checkpoint
    checkpoint_dir = os.path.join(base_dir, 'decoder_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'decoder.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Decoder model saved at: {checkpoint_path}")

    # Save vocab
    vocab_dir = os.path.join(base_dir, 'data')
    os.makedirs(vocab_dir, exist_ok=True)
    vocab_path = os.path.join(vocab_dir, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as vf:
        json.dump(vocab, vf, indent=2)
    print(f"✅ Vocab saved at: {vocab_path}")


# === Run if called directly ===
if __name__ == "__main__":
    train_decoder_main()
