import librosa
import numpy as np


def extract_mfcc(audio_path, n_mfcc=13):
    """
    Extract MFFC from the audio

    Args:
    audio_path(str): The input audio path
    n_mfcc(int): The number of MFCC
    """
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc
