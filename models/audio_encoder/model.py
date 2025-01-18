import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(AudioEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_dim)
        x = x.permute(1, 0, 2)
        return x
