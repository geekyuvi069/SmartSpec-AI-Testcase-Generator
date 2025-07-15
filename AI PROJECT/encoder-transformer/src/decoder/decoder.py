import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention with masking
        _tgt = self.norm1(tgt + self.dropout(
            self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=False)[0]
        ))

        # Cross-attention with encoder memory
        _tgt = self.norm2(_tgt + self.dropout(
            self.cross_attn(_tgt, memory, memory, attn_mask=memory_mask, need_weights=False)[0]
        ))

        # Feedforward + residual + norm
        ff = self.linear2(self.dropout(self.activation(self.linear1(_tgt))))
        output = self.norm3(_tgt + self.dropout(ff))

        return output


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_len=100):
        super(SimpleTransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, size):
        # Mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask  # (size, size)

    def forward(self, tgt_seq, memory):
        # tgt_seq: (batch, seq_len)
        x = self.embedding(tgt_seq)
        x = self.positional_encoding(x)

        seq_len = tgt_seq.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask)

        logits = self.fc_out(x)  # (batch, seq_len, vocab_size)
        return logits 
    