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

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden_size=512, dropout=0.1, max_len=5000):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embed_size: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of decoder layers.
            ff_hidden_size: Hidden size of feedforward network.
            dropout: Dropout probability.
            max_len: Maximum sequence length for positional encoding.
        """
        super(TransformerDecoder, self).__init__()

        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=ff_hidden_size,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: Target sequence [tgt_seq_len, batch_size]
            memory: Encoder output [src_seq_len, batch_size, embed_size]
            tgt_mask: Mask for target sequence.
            memory_mask: Mask for encoder output.
            tgt_key_padding_mask: Padding mask for target keys.
            memory_key_padding_mask: Padding mask for memory keys.
        Returns:
            Output logits [tgt_seq_len, batch_size, vocab_size]
        """
        # Embed and add positional encoding
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_size)
        tgt_emb = self.positional_encoding(tgt_emb)

        # Pass through Transformer decoder
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Final linear projection
        output = self.fc_out(output)

        return output
