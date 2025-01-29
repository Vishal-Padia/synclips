import os
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from models.tts.model import TTSModule
from models.video_decoder.model import Generator
from models.alignment.model import CrossAttention
from models.video_encoder.model import VideoEncoder
from models.audio_encoder.model import AudioEncoder


@dataclass
class ModelConfig:
    """Configuration for loading models."""

    checkpoint_dir: str
    audio_encoder_path: Optional[str] = None
    video_encoder_path: Optional[str] = None
    alignment_module_path: Optional[str] = None
    generator_path: Optional[str] = None

    def __post_init__(self):
        """Initialize paths if not provided."""
        if not self.audio_encoder_path:
            self.audio_encoder_path = os.path.join(
                self.checkpoint_dir, "best_audio_encoder.pth"
            )
        if not self.video_encoder_path:
            self.video_encoder_path = os.path.join(
                self.checkpoint_dir, "best_video_encoder.pth"
            )
        if not self.alignment_module_path:
            self.alignment_module_path = os.path.join(
                self.checkpoint_dir, "best_alignment_module.pth"
            )
        if not self.generator_path:
            self.generator_path = os.path.join(
                self.checkpoint_dir, "best_generator_model.pth"
            )


class InferencePipeline:
    def __init__(
        self,
        config: ModelConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize inference pipeline with models."""
        self.device = device
        self.config = config

        # Initialize TTS module
        self.tts_module = TTSModule(device=device)

        # Load other models
        self._load_models()

    def _load_models(self):
        """Load all required models."""

        # Initialize models
        self.audio_encoder = AudioEncoder(
            input_dim=13, hidden_dim=128, num_heads=8, num_layers=4
        ).to(self.device)
        self.video_encoder = VideoEncoder(
            input_channels=3, hidden_dim=64, output_dim=128
        ).to(self.device)
        self.alignment_module = CrossAttention(
            audio_dim=128, video_dim=128, hidden_dim=128, num_heads=8
        ).to(self.device)
        self.generator = Generator(input_dim=128, hidden_dim=512, output_channels=3).to(
            self.device
        )

        # Load state dicts
        self.audio_encoder.load_state_dict(
            torch.load(self.config.audio_encoder_path, map_location=self.device)
        )
        self.video_encoder.load_state_dict(
            torch.load(self.config.video_encoder_path, map_location=self.device)
        )
        self.alignment_module.load_state_dict(
            torch.load(self.config.alignment_module_path, map_location=self.device)
        )
        self.generator.load_state_dict(
            torch.load(self.config.generator_path, map_location=self.device)
        )

        # Set to eval mode
        self.audio_encoder.eval()
        self.video_encoder.eval()
        self.alignment_module.eval()
        self.generator.eval()

    def process_text(
        self,
        text: str,
        reference_video_path: str,
        output_dir: str,
        temp_dir: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input text to generate lip-synced video.

        Args:
            text: Input text to convert to speech
            reference_video_path: Path to reference video for face extraction
            output_dir: Directory to save outputs
            temp_dir: Directory for temporary files

        Returns:
            Tuple of generated video frames and audio
        """
        # Create directories
        output_dir = Path(output_dir)
        temp_dir = Path(temp_dir) if temp_dir else output_dir / "temp"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # Generate speech from text
        speech_waveform, sample_rate = self.tts_module.generate_speech(
            text, output_path=str(output_dir / "generated_speech.wav")
        )

        # Extract MFCC features from generated speech
        from utils.audio_utils import extract_mfcc

        mfcc = extract_mfcc(str(output_dir / "generated_speech.wav"))
        mfcc_tensor = (
            torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Extract reference face
        from utils.face_utils import crop_faces_concurrently
        import cv2

        # Extract first frame from reference video
        cap = cv2.VideoCapture(reference_video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read reference video")

        # Save frame and crop face
        cv2.imwrite(str(temp_dir / "reference_frame.jpg"), frame)
        crop_faces_concurrently(
            frame_dir=str(temp_dir), cropped_face_dir=str(temp_dir / "cropped")
        )

        # Load cropped face
        reference_face = cv2.imread(str(temp_dir / "cropped" / "reference_frame.jpg"))
        reference_face = (
            torch.tensor(reference_face)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        # Generate video frames
        with torch.no_grad():
            # Get audio features
            audio_features = self.audio_encoder(mfcc_tensor)

            # Create video sequence from reference frame
            video_sequence = reference_face.unsqueeze(2).repeat(1, 1, 16, 1, 1)

            # Get video features
            video_features = self.video_encoder(video_sequence)

            # Flatten video features
            batch_size, feat_dim, d, h, w = video_features.shape
            video_features_flat = video_features.permute(0, 2, 3, 4, 1).reshape(
                batch_size, -1, feat_dim
            )

            # Cross-attention
            aligned_audio = self.alignment_module(audio_features, video_features_flat)

            # Generate frames
            generated_frames = self.generator(aligned_audio.reshape(-1, 128))
            generated_frames = generated_frames.reshape(
                -1, aligned_audio.shape[1], 3, 32, 32
            )

        return generated_frames, speech_waveform

    def save_output_video(
        self,
        frames: torch.Tensor,
        audio: torch.Tensor,
        output_path: str,
        fps: int = 25,
        sample_rate: int = 16000,
    ):
        """Save generated frames and audio as a video."""
        import cv2
        import tempfile
        import subprocess

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Save frames
            frames = frames.cpu().numpy()
            for i, frame in enumerate(frames[0]):  # Assuming batch size 1
                frame = ((frame.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                cv2.imwrite(str(temp_dir / f"frame_{i:04d}.jpg"), frame)

            # Save audio
            temp_audio_path = temp_dir / "temp_audio.wav"
            from models.tts.model import TTSModule

            TTSModule.save_audio(audio, sample_rate, str(temp_audio_path))

            # Combine frames and audio using ffmpeg
            frame_pattern = str(temp_dir / "frame_%04d.jpg")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    frame_pattern,
                    "-i",
                    str(temp_audio_path),
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-strict",
                    "experimental",
                    "-shortest",
                    output_path,
                ]
            )
