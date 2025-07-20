import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

class Trainer:
    def __init__(self, encoder, decoder, vocab, device):
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
        self.optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=1e-3
        )

    def prepare_batch(self, text_batch):
        """
        Converts a list of text strings into input/output tensors.
        """
        sequences = [torch.tensor(self.vocab.text_to_sequence(t), dtype=torch.long) for t in text_batch]

        # Encoder input: full sequence except last token
        encoder_inputs = [seq[:-1] for seq in sequences]
        # Decoder input: BOS + all tokens except last
        decoder_inputs = [torch.cat([torch.tensor([self.vocab.bos_id]), seq[:-1]]) for seq in sequences]
        # Decoder target: full sequence
        decoder_targets = [seq for seq in sequences]

        # Pad sequences
        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.vocab.pad_id).to(self.device)
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.vocab.pad_id).to(self.device)
        decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=self.vocab.pad_id).to(self.device)

        return encoder_inputs, decoder_inputs, decoder_targets

    def train_epoch(self, text_chunks, batch_size=4):
        """
        Runs one epoch over all chunks.
        """
        self.encoder.train()
        self.decoder.train()

        total_loss = 0.0
        num_batches = (len(text_chunks) + batch_size - 1) // batch_size

        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            enc_in, dec_in, dec_target = self.prepare_batch(batch)

            self.optimizer.zero_grad()

            enc_output = self.encoder(enc_in)
            dec_output = self.decoder(dec_in, enc_output)

            # Flatten outputs and targets
            logits = dec_output.reshape(-1, dec_output.size(-1))
            targets = dec_target.reshape(-1)

            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_models(self, encoder_path, decoder_path):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load_models(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
