import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_utils import LipSyncDataset
from models.alignment.model import CrossAttention
from models.audio_encoder.model import AudioEncoder
from models.video_encoder.model import VideoEncoder
from torch.utils.data import DataLoader, ConcatDataset
from models.video_decoder.model import Generator, Discriminator
from torchvision.transforms import Compose, ToTensor, Normalize

# define data path
processed_data_dir = "data/processed"

# define transforms
transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# create a list of all aligned_data.json files
aligned_data_files = []
for speaker in os.listdir(processed_data_dir):
    speaker_dir = os.path.join(processed_data_dir, speaker_dir)
    for video_title in os.listdir(speaker_dir):
        video_dir = os.path.join(speaker_dir, video_title)
        aligned_data_file = os.path.join(video_dir, "aligned_data.json")
        if os.path.exists(aligned_data_file):
            aligned_data_files.append(aligned_data_file)

# Create a combined dataset
datasets = [
    LipSyncDataset(aligned_data_file, transform=transform)
    for aligned_data_file in aligned_data_files
]
combined_dataset = ConcatDataset(datasets)

# Create a dataloader
dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

# Initialize models
audio_encoder = AudioEncoder(input_dim=13, hidden_dim=128, num_heads=8, num_layers=4)
video_encoder = VideoEncoder(input_channels=3, hidden_dim=64, output_dim=128)
alignment_module = CrossAttention(
    audio_dim=128, video_dim=128, hidden_dim=128, num_heads=8
)
generator = Generator(input_dim=128, hidden_dim=512, output_channels=3)
discriminator = Discriminator(input_channels=3, hidden_dim=64)

# Move everything to the availabel device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_encoder.to(device)
video_encoder.to(device)
alignment_module.to(device)
generator.to(device)
discriminator.to(device)

# Define loss function
criterion_l1 = nn.L1Loss()  # For pixel-wise accuracy
criterion_gan = nn.BCEWithLogitsLoss()  # For adversarial loss

# Define Optimizer
optimizer_g = optim.Adam(
    list(audio_encoder.parameters())
    + list(video_encoder.parameters())
    + list(alignment_module.parameters())
    + list(generator.parameters()),
    lr=0.0002,
    betas=(0.5, 0.999),
)

optimizer_d = optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    for i, (frames, mfccs) in enumerate(dataloader):
        # move the data to the device
        frames = frames.to(device)
        mfccs = mfccs.to(device)

        # Training discriminator
        optimizer_d.zero_grad()

        # forward pass
        audio_features = audio_encoder(mfccs)
        video_features = video_encoder(frames)
        aligned_audio = alignment_module(audio_features, video_features)
        generated_frames = generator(aligned_audio)

        # real and fake labels
        real_labels = torch.ones(frames.size(0), 1).to(device)
        fake_labels = torch.zeros(frames.size(0), 1).to(device)

        # Discriminator loss on real frames
        real_output = discriminator(frames)
        loss_d_real = criterion_gan(real_output, real_labels)

        # Discriminator loss on fake frames
        fake_output = discriminator(generated_frames.detach())
        loss_d_fake = criterion_gan(fake_output, fake_labels)

        # Total discriminator loss
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        # Training generator
        optimizer_g.zero_grad()

        # Generator loss (adversarial)
        fake_output = discriminator(generated_frames)
        loss_g_gan = criterion_gan(fake_output, real_labels)

        # Generator loss (L1)
        loss_g_l1 = criterion_l1(generated_frames, frames)

        # Total generator loss
        loss_g = loss_g_gan + 100 * loss_g_l1  # weighted L1 loss
        loss_g.backward()
        optimizer_g.step()

        # Print Losses
        if i % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                f"Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}"
            )

        # Save model checkpoints
    if (epoch + 1) % 10 == 0:
        torch.save(
            audio_encoder.state_dict(),
            f"outputs/checkpoints/audio_encoder_epoch_{epoch+1}.pth",
        )
        torch.save(
            video_encoder.state_dict(),
            f"outputs/checkpoints/video_encoder_epoch_{epoch+1}.pth",
        )
        torch.save(
            alignment_module.state_dict(),
            f"outputs/checkpoints/alignment_module_epoch_{epoch+1}.pth",
        )
        torch.save(
            generator.state_dict(), f"outputs/checkpoints/generator_epoch_{epoch+1}.pth"
        )
        torch.save(
            discriminator.state_dict(),
            f"outputs/checkpoints/discriminator_epoch_{epoch+1}.pth",
        )
