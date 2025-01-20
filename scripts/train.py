import os
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.data_utils import LipSyncDataset
from models.alignment.model import CrossAttention
from utils.custom_collate import custom_collate_fn
from models.audio_encoder.model import AudioEncoder
from models.video_encoder.model import VideoEncoder
from torch.utils.data import DataLoader, ConcatDataset, Subset
from models.video_decoder.model import Generator, Discriminator
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

# Hyperparameters
learning_rate = 0.0002
epochs = 100
batch_size = 32
checkpoint_dir = "outputs/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize wandb
wandb.login()
run = wandb.init(
    project="SyncLips",
    config={"lr": learning_rate, "epochs": epochs, "batch_size": batch_size},
)

# Dataset and transforms
transform = Compose(
    [
        Resize((32, 32)),  # Match generator output size
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

processed_data_dir = "data/processed"
aligned_data_files = [
    os.path.join(processed_data_dir, speaker, video_title, "aligned_data.json")
    for speaker in os.listdir(processed_data_dir)
    for video_title in os.listdir(os.path.join(processed_data_dir, speaker))
    if os.path.exists(
        os.path.join(processed_data_dir, speaker, video_title, "aligned_data.json")
    )
]

# Create dataset
combined_dataset = ConcatDataset(
    [LipSyncDataset(f, transform) for f in aligned_data_files]
)
dataloader = DataLoader(
    combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
)

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_encoder = AudioEncoder(
    input_dim=13, hidden_dim=128, num_heads=8, num_layers=4
).to(device)
video_encoder = VideoEncoder(input_channels=3, hidden_dim=64, output_dim=128).to(device)
alignment_module = CrossAttention(
    audio_dim=128, video_dim=128, hidden_dim=128, num_heads=8
).to(device)
generator = Generator(input_dim=128, hidden_dim=512, output_channels=3).to(device)
discriminator = Discriminator(input_channels=3, hidden_dim=64).to(device)

# Optimizers
optimizer_g = optim.Adam(
    list(audio_encoder.parameters())
    + list(video_encoder.parameters())
    + list(alignment_module.parameters())
    + list(generator.parameters()),
    lr=learning_rate,
    betas=(0.5, 0.999),
)
optimizer_d = optim.Adam(
    discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
)

# Loss functions
criterion_l1 = nn.L1Loss()
criterion_gan = nn.BCEWithLogitsLoss()

best_loss_g = float("inf")

# Training loop
for epoch in range(epochs):
    for i, (frames, mfccs) in enumerate(dataloader):
        # Move data to device
        frames = frames.to(device)  # (batch_size, 3, 32, 32) after transform
        mfccs = mfccs.to(device)  # (batch_size, seq_len, 13)

        # --- Video Preprocessing ---
        # Create video sequence: (batch, 3, 16, 32, 32)
        video_sequence = (
            frames.unsqueeze(2).repeat(1, 1, 16, 1, 1).permute(0, 1, 2, 3, 4)
        )

        # --- Forward Pass ---
        # Audio features
        audio_features = audio_encoder(mfccs)  # (batch, seq_len, 128)

        # Video features
        video_features = video_encoder(video_sequence)  # (batch, 128, 2, 8, 8)

        # Cross-attention
        batch_size, feat_dim, d, h, w = video_features.shape
        video_features_flat = video_features.permute(0, 2, 3, 4, 1).reshape(
            batch_size, -1, feat_dim
        )
        aligned_audio = alignment_module(
            audio_features, video_features_flat
        )  # (batch, seq_len, 128)

        # Generate frames
        seq_len = aligned_audio.shape[1]
        generated_frames = generator(
            aligned_audio.reshape(-1, 128)
        )  # (batch*seq_len, 3, 32, 32)
        generated_frames = generated_frames.reshape(batch_size, seq_len, 3, 32, 32)

        # --- Discriminator Inputs ---
        # Real frames (original frames resized to 32x32)
        real_frames = frames.unsqueeze(1).repeat(
            1, seq_len, 1, 1, 1
        )  # (batch, seq_len, 3, 32, 32)
        real_frames = real_frames.reshape(-1, 3, 32, 32)  # (batch*seq_len, 3, 32, 32)

        # Generated frames
        generated_frames = generated_frames.reshape(
            -1, 3, 32, 32
        )  # (batch*seq_len, 3, 32, 32)

        # --- Train Discriminator ---
        optimizer_d.zero_grad()

        # Real loss
        real_pred = discriminator(real_frames)
        loss_real = criterion_gan(real_pred, torch.ones_like(real_pred))

        # Fake loss
        fake_pred = discriminator(generated_frames.detach())
        loss_fake = criterion_gan(fake_pred, torch.zeros_like(fake_pred))

        loss_d = (loss_real + loss_fake) / 2
        loss_d.backward()
        optimizer_d.step()

        # --- Train Generator ---
        optimizer_g.zero_grad()

        # Adversarial loss
        fake_pred = discriminator(generated_frames)
        loss_g_gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))

        # Reconstruction loss
        loss_g_l1 = criterion_l1(generated_frames, real_frames)

        # Total loss
        loss_g = loss_g_gan + 100 * loss_g_l1
        loss_g.backward()
        optimizer_g.step()

        # --- Logging & Checkpoints ---
        if i % 10 == 0:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss_d": loss_d.item(),
                    "loss_g": loss_g.item(),
                    "loss_g_gan": loss_g_gan.item(),
                    "loss_g_l1": loss_g_l1.item(),
                }
            )

            if loss_g < best_loss_g:
                best_loss_g = loss_g
                torch.save(
                    audio_encoder.state_dict(),
                    os.path.join(checkpoint_dir, "best_audio_encoder.pth"),
                )
                torch.save(
                    video_encoder.state_dict(),
                    os.path.join(checkpoint_dir, "best_video_encoder.pth"),
                )
                torch.save(
                    alignment_module.state_dict(),
                    os.path.join(checkpoint_dir, "best_alignment_module.pth"),
                )
                torch.save(
                    generator.state_dict(),
                    os.path.join(checkpoint_dir, "best_generator_model.pth"),
                )

            # Save model checkpoints
            if (epoch + 1) % 10 == 0:
                torch.save(
                    audio_encoder.state_dict(),
                    os.path.join(checkpoint_dir, f"audio_encoder_epoch_{epoch+1}.pth"),
                )
                torch.save(
                    video_encoder.state_dict(),
                    os.path.join(checkpoint_dir, f"video_encoder_epoch_{epoch+1}.pth"),
                )
                torch.save(
                    alignment_module.state_dict(),
                    os.path.join(
                        checkpoint_dir, f"alignment_module_epoch_{epoch+1}.pth"
                    ),
                )
                torch.save(
                    generator.state_dict(),
                    os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.pth"),
                )
                torch.save(
                    discriminator.state_dict(),
                    os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch+1}.pth"),
                )

                # Log checkpoints to wandb
                wandb.save(
                    os.path.join(checkpoint_dir, f"audio_encoder_epoch_{epoch+1}.pth")
                )
                wandb.save(
                    os.path.join(checkpoint_dir, f"video_encoder_epoch_{epoch+1}.pth")
                )
                wandb.save(
                    os.path.join(
                        checkpoint_dir, f"alignment_module_epoch_{epoch+1}.pth"
                    )
                )
                wandb.save(
                    os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.pth")
                )
                wandb.save(
                    os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch+1}.pth")
                )

# Finish the wandb run
wandb.finish()
