import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_channels):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim * 8 * 8)
        self.conv1 = nn.ConvTranspose2d(
            hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.ConvTranspose2d(
            hidden_dim // 2, output_channels, kernel_size=4, stride=2, paddin=1
        )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)  # (batch_size, hidden_dim * 8 * 8)
        x = x.view(-1, 512, 8, 8)  # (batch_size, hidden_dim, 8, 8)
        x = self.relu(self.conv1(x))
        x = self.tanh(self.conv2(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, hidden_dim, kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1
        )
        self.fc = nn.Linear(hidden_dim * 2 * 8 * 8, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))  # (batch_size, hidden_dim, 16, 16)
        x = self.leaky_relu(self.conv2(x))  # (batch_size, hidden_dim * 2, 8, 8)
        x = x.view(x.size(0), -1)  # (batch_size, hidden_dim * 2 * 8 * 8)
        x = self.fc(x)  # (batch_size, 1)

        return x
