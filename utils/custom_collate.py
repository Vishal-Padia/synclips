import torch
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch):
    """
    Collate function to handle variable-length MFCCs.

    Args:
    batch (list): List of tuples (frame, mfcc) from the dataset.

    Returns:
    frames (torch.Tensor): Stacked frames tensor of shape (batch_size, C, H, W)
    mfccs (torch.Tensor): Padded MFCCs tensor of shape (batch_size, time_steps, n_mfcc)
    """
    # Separate frames and MFCCs
    frames = [item[0] for item in batch]  # Add depth dimension
    mfccs = [item[1].t() for item in batch]  # Transpose MFCC

    # Stack frames (they are already the same size)
    frames_stacked = torch.stack(frames, dim=0)  # (batch_size, C, depth=1, H, W)

    # Pad MFCCs to the maximum length in the batch
    mfccs_padded = pad_sequence(mfccs, batch_first=True, padding_value=0)

    return frames_stacked, mfccs_padded
