import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embed_size]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden_size=512, dropout=0.1, max_len=5000):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embed_size: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of encoder layers.
            ff_hidden_size: Hidden size of feedforward network.
            dropout: Dropout probability.
            max_len: Maximum sequence length.
        """
        super(TransformerEncoder, self).__init__()

        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ff_hidden_size,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Input tensor [src_seq_len, batch_size]
            src_mask: Optional mask.
            src_key_padding_mask: Optional padding mask.
        Returns:
            Encoded memory [src_seq_len, batch_size, embed_size]
        """
        src_emb = self.embedding(src) * math.sqrt(self.embed_size)
        src_emb = self.positional_encoding(src_emb)

        memory = self.transformer_encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        return memory
