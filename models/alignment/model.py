import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, audio_dim, video_dim, hidden_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads)

    def forward(self, audio_features, video_features):
        # Project inputs to hidden_dim
        audio_proj = self.audio_proj(audio_features)  # (batch, seq_a, hidden)
        video_proj = self.video_proj(video_features)  # (batch, seq_v, hidden)

        # Permute for MultiheadAttention
        audio_proj = audio_proj.permute(1, 0, 2)  # (seq_a, batch, hidden)
        video_proj = video_proj.permute(1, 0, 2)  # (seq_v, batch, hidden)

        # Apply cross-attention
        aligned_audio, _ = self.cross_attention(audio_proj, video_proj, video_proj)
        aligned_audio = aligned_audio.permute(1, 0, 2)  # (batch, seq_a, hidden)
        return aligned_audio
