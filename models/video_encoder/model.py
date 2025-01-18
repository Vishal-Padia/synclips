import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim):
        super(VideoEncoder, self).__init__()
        self.conv1 = nn.Conv3d(
            input_channels, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )
        self.conv2 = nn.Conv3d(
            hidden_dim, hidden_dim * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )
        self.conv3 = nn.Conv3d(
            hidden_dim * 2, output_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, channels, depth, height, width)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        return x  # (batch_size, output_dim, depth` , height`, width`)
