import torch
import torchaudio

import numpy as np

from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


class TTSModule:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize TTS module with pretrained models."""
        self.device = device
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(
            device
        )

        # Load speaker embeddings from a dataset
        dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(dataset[7306]["xvector"]).unsqueeze(0)

    def preprocess_text(self, text):
        """Preprocess input text for TTS."""
        inputs = self.processor(text=text, return_tensors="pt")
        return inputs.to(self.device)

    def generate_speech(self, text, output_path=None, sample_rate=16000):
        """
        Generate speech from input text.

        Args:
            text (str): Input text to convert to speech
            output_path (str, optional): Path to save the audio file
            sample_rate (int): Sample rate of the output audio

        Returns:
            torch.Tensor: Generated speech waveform
            int: Sample rate
        """
        # Preprocess text
        inputs = self.preprocess_text(text)

        # Generate speech
        speech = self.model.generate_speech(
            inputs["input_ids"],
            self.speaker_embeddings.to(self.device),
            vocoder=self.vocoder,
        )

        # Save audio if output path is provided
        if output_path:
            torchaudio.save(output_path, speech.unsqueeze(0), sample_rate=sample_rate)

        return speech, sample_rate

    def batch_generate_speech(self, texts, output_dir=None, sample_rate=16000):
        """
        Generate speech for multiple texts.

        Args:
            texts (list): List of input texts
            output_dir (str, optional): Directory to save audio files
            sample_rate (int): Sample rate of the output audio

        Returns:
            list: List of (waveform, sample_rate) tuples
        """
        results = []

        for i, text in enumerate(texts):
            output_path = f"{output_dir}/speech_{i}.wav" if output_dir else None
            speech, sr = self.generate_speech(text, output_path, sample_rate)
            results.append((speech, sr))

        return results

    @staticmethod
    def save_audio(waveform, sample_rate, output_path):
        """Save audio waveform to file."""
        torchaudio.save(output_path, waveform.unsqueeze(0), sample_rate)
