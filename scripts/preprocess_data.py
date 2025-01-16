import os
import cv2

import numpy as np

from utils.face_utils import crop_face
from utils.audio_utils import extract_mfcc
from utils.data_utils import load_alignments, align_frames_and_audio


def extract_frames(video_path, output_dir, frame_rate=10):
    """
    Extract frames from the videos and then save it to a directory

    Args:
    video_path (str): The path to the video
    output_dir (str): The output directory name
    frame_rate (int): The frame rate at which we need to save images
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        # save images acc to the frame rate
        if frame_count % frame_rate == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:04d}.jpg"), frame)

        # increment frame count
        frame_count += 1

    cap.release()


def preprocess_speaker_data(speaker_dir, output_dir, frame_rate=10):
    """
    Preprocess data for a single speaker

    Args:
    speaker_dir (str): Directory containing raw data for a speaker
    output_dir (str): Directory to save preprocessed data
    frame_rate (int): Frame rate for extracting video frames
    """
    # Paths
    video_files = os.listdir(os.path.join(speaker_dir, "video"))
    audio_files = os.listdir(os.path.join(speaker_dir, "audio"))
    align_files = os.listdir(os.path.join(speaker_dir, "alignments"))

    # Ensure the number of files matches
    if not (len(video_files) == len(audio_files) == len(align_files)):
        raise ValueError("Mismatch in the number of video, audio, alignment files")

    # process each file
    for video_file, audio_file, align_file in zip(
        video_files, audio_files, align_files
    ):
        # construct full paths
        video_path = os.path.join(speaker_dir, "video", video_file)
        audio_path = os.path.join(speaker_dir, "audio", audio_file)
        align_path = os.path.join(speaker_dir, "alignments", align_file)

        # create output directories for this file
        file_name = os.path.splitext(video_file)[0]
        file_output_dir = os.path.join(output_dir, file_name)
        frame_dir = os.path.join(file_output_dir, "frames")
        cropped_face_dir = os.path.join(file_output_dir, "cropped_faces")
        audio_features_dir = os.path.join(file_output_dir, "audio_features")

        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(cropped_face_dir, exist_ok=True)
        os.makedirs(audio_features_dir, exist_ok=True)

        # Extracting frames
        extract_frames(
            video_path=video_path, output_dir=frame_dir, frame_rate=frame_rate
        )

        # Cropping faces
        for frame_file in os.listdir(frame_dir):
            frame_path = os.path.join(frame_dir, frame_file)
            cropped_face_path = os.path.join(cropped_face_dir, frame_file)
            crop_face(image_path=frame_path, output_path=cropped_face_path)

        # Extracting MFCCs
        mfcc = extract_mfcc(audio_path=audio_path)
        np.save(os.path.join(audio_features_dir, "mfcc.npy"), mfcc)

        # Align frames and audio
        alignments = load_alignments(align_path=align_path)
        aligned_data_file = os.path.join(file_output_dir, "aligned_data.json")
        align_frames_and_audio(
            frame_dir=cropped_face_dir,
            alignments=alignments,
            output_file=aligned_data_file,
            frame_rate=frame_rate,
        )
