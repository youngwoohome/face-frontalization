import cv2
import numpy as np
import math

def calculate_face_rotation(image_input):
    """
    Calculate the rotation angle of a face in an image using only OpenCV.
    
    Args:
        image_input: Input image (str: file path or numpy.ndarray: image array)
    
    Returns:
        float: Face rotation angle in degrees
    """
    
    # Load image (file path or memory image)
    if isinstance(image_input, str):
        # File path case
        image = cv2.imread(image_input)
        if image is None:
            print(f"Cannot load image: {image_input}")
            return None
    else:
        # numpy array case (memory image)
        image = image_input
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's Haar Cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No face found.")
        return None
    
    # Process the first face
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    
    # Eye detection in face region
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
    if len(eyes) < 2:
        print("Cannot find two eyes.")
        return None
    
    # Calculate eye centers (select two eyes with similar Y coordinates)
    eye_centers = []
    for (ex, ey, ew, eh) in eyes:
        center_x = x + ex + ew // 2
        center_y = y + ey + eh // 2
        eye_centers.append((center_x, center_y, ew * eh))  # Also store area
    
    # Sort by area in descending order to select the largest two eyes
    eye_centers.sort(key=lambda x: x[2], reverse=True)
    
    if len(eye_centers) >= 2:
        # Select the largest two eyes
        left_eye = eye_centers[0]
        right_eye = eye_centers[1]
        
        # Distinguish left and right eyes based on X coordinate
        if left_eye[0] > right_eye[0]:
            left_eye, right_eye = right_eye, left_eye
        
        # Calculate angle between the two eyes
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(delta_y, delta_x))
        
        return angle
    
    return None

# Usage example
if __name__ == "__main__":
    # Enter the image file path here
    image_path = "mask_proper.jpeg"  # Change to actual image file path
    
    print("=== Face Rotation Angle Detection Using OpenCV Only ===")
    
    # Basic method
    print("\n1. Basic method:")
    angle = calculate_face_rotation(image_path)
    if angle is not None:
        print(f"Face rotation angle: {angle:.2f} degrees")
        if abs(angle) < 5:
            print("Face is almost horizontal.")
        elif angle > 0:
            print("Face is tilted clockwise.")
        else:
            print("Face is tilted counterclockwise.")
    