import os
import json


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


def align_frames_and_audio(frame_dir, alignments, output_file, frame_rate=25):
    """
    Align video frames with audio segments using alignment file.

    Args:
    frame_dir (str): Directory containing the extracted frames
    alignments (list): List of (start_time, end_time, word) tuples
    output_file (str): Path to save the aligned data (json file)
    frame_rate (int): Frame rate of the video
    """
    aligned_data = []

    # iterate over frames and match them to words
    for frame_file in sorted(os.listdir(frame_dir)):
        frame_path = os.path.join(frame_dir, frame_file)
        if not os.path.exists(frame_path):  # Skip if frame file doesn't exist
            print(f"Frame {frame_path} does not exist. Skipping.")
            continue
        frame_number = int(frame_file.split("_")[1].split(".")[0])
        frame_time = frame_number / frame_rate

        # find the word corresponding to the frame's timestamp
        for start_time, end_time, word in alignments:
            if start_time <= frame_time < end_time:
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
