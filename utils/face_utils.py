import os
import cv2
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor, as_completed


def crop_face(image_path, output_path):
    """
    Cropping the face using MTCNN model

    Args:
    image_path (str): The input image path
    output_path (str): The output image path
    """
    detector = MTCNN()
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    if result:
        x, y, width, height = result[0]["box"]
        face = image[y : y + height, x : x + width]
        cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))


def crop_faces_concurrently(frame_dir, cropped_face_dir):
    """
    Crop faces from all frames in a directory concurrently

    Args:
    frame_dir (str): Directory containing the input frames
    cropped_face_dir (str): Directory to save the cropped faces
    """
    # Get list of frame files
    frame_files = os.listdir(frame_dir)

    # Use ThreadPoolExecutor for concurrent face cropping
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit tasks for each frame
        futures = [
            executor.submit(
                crop_face,
                os.path.join(frame_dir, frame_file),
                os.path.join(cropped_face_dir, frame_file),
            )
            for frame_file in frame_files
        ]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()  # Check for exceptions
            except Exception as e:
                print(f"Error cropping face: {e}")
