import cv2
from mtcnn import MTCNN


def crop_face(image_path, output_path):
    """
    Cropping the face using MTCNN model

    Args:
    image_path(str): The input image path
    output_path(str): The output image path
    """
    detector = MTCNN()
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    if result:
        x, y, width, height = result[0]["box"]
        face = image[y : y + height, x : x + width]
        cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    else:
        print(f"No faces detected in {image_path}, Skipping this frame")
        return
