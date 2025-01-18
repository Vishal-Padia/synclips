import os
import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


def load_alignments(align_path):
    """
    Load alignment file and return a list of (start_time, end_time, word) tuples.

    Args:
    align_path (str): Path to the alignment file

    Returns:
    list: List of tuples containing (start_time, end_time, word)
    """
    alignments = []
    with open(align_path, "r") as f:
        for line in f:
            start_time, end_time, word = line.strip().split()
            alignments.append((float(start_time), float(end_time), word))

    return alignments


def align_frames_and_audio(frame_dir, alignments, output_file):
    """
    Align video frames with audio segments using alignment file.

    Args:
    frame_dir (str): Directory containing the extracted frames
    alignments (list): List of (start_time, end_time, word) tuples
    output_file (str): Path to save the aligned data (json file)
    """
    aligned_data = []

    # iterate over frames and match them to words
    for frame_file in sorted(os.listdir(frame_dir)):
        frame_path = os.path.join(frame_dir, frame_file)
        if not os.path.exists(frame_path):  # Skip if frame file doesn't exist
            print(f"Frame {frame_path} does not exist. Skipping.")
            continue
        frame_number = int(frame_file.split("_")[1].split(".")[0])

        # find the word corresponding to the frame's timestamp
        for start_time, end_time, word in alignments:
            if start_time <= frame_number * 1000 < end_time:
                aligned_data.append(
                    {
                        "frame_path": frame_path,
                        "word": word,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )
                break

    # save the data to a json file
    with open(output_file, "w") as f:
        json.dump(aligned_data, f, indent=4)


class LipSyncDataset(Dataset):
    def __init__(self, aligned_data_file, transform=None):
        """
        Args:
        aligned_data_file (str): Path to the aligned data json file
        transform (callable, optional): Optional transform to be applied on a sample
        """
        with open(aligned_data_file, "r") as f:
            self.aligned_data = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.aligned_data)

    def __getitem__(self, idx):

        # load frame
        frame_path = self.aligned_data[idx]["frame_path"]
        frame = Image.open(frame_path).convert("RGB")

        # load MFCCs
        mfcc_path = frame_path.split("cropped_faces")[0] + "audio_features\\mfcc.npy"
        mfcc = np.load(mfcc_path)

        # apply transformations
        if self.transform:
            frame = self.transform(frame)

        # conver to tensors
        frame = torch.tensor(np.array(frame), dtype=torch.float32).permute(
            2, 0, 1
        )  # (C, H, W)
        mfcc = torch.tensor(mfcc, dtype=torch.float32)  # (n_mfcc, time_steps)

        return frame, mfcc
