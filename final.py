from face_detect import find_faces_confidence_score_from_image
from face_rotation import calculate_face_rotation
import cv2
import numpy as np

def rotate_image(image, angle):
    """
    Rotates an image by the given angle.
    
    Args:
        image: Input image (numpy array)
        angle: Rotation angle (0, 90, 180, 270)
    
    Returns:
        numpy array: Rotated image
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Angle must be one of 0, 90, 180, 270.")

if __name__ == "__main__":
    image_path = "./content/image2_right.jpeg"
    
    # Load original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Rotate image by 0, 90, 180, 270 degrees and store in memory
    rotated_images = []
    angles = [0, 90, 180, 270]
    
    for angle in angles:
        rotated_image = rotate_image(original_image, angle)
        rotated_images.append(rotated_image)
        print(f"Image rotated by {angle} degrees completed")
    
    # Perform face detection on each rotated image
    scores = []
    for i, rotated_image in enumerate(rotated_images):
        angle = angles[i]
        score = find_faces_confidence_score_from_image(rotated_image)
        scores.append(score)
        print(f"Face detection score for {angle} degree rotation: {score}")
    
    # Output the highest score and corresponding angle
    max_score = max(scores)
    best_angle = angles[scores.index(max_score)]
    print(f"\nHighest score: {max_score}")
    print(f"Rotate {best_angle} degrees to face forward")


    # Calculate face rotation angle from the highest scoring image
    face_rotation = calculate_face_rotation(rotated_images[scores.index(max_score)])


    # Calculate final rotation angle
    # best_angle: angle to roughly face forward (one of 0, 90, 180, 270 degrees)
    # face_rotation: fine adjustment angle from rough front to exact front
    # final angle = best_angle + face_rotation
    
    if face_rotation is not None:
        print(f"Face rotation angle: {face_rotation} degrees")
        final_rotation_angle = best_angle + face_rotation
        
        # Normalize angle to -180 ~ 180 range
        while final_rotation_angle > 180:
            final_rotation_angle -= 360
        while final_rotation_angle <= -180:
            final_rotation_angle += 360
            
        print(f"\n=== Final Result ===")
        print(f"Stage 1 rotation (rough front): {best_angle} degrees")
        print(f"Stage 2 rotation (fine adjustment): {face_rotation:.2f} degrees")
        print(f"Final rotation angle: {final_rotation_angle:.2f} degrees")
        print(f"Rotate the original image clockwise by {final_rotation_angle:.2f} degrees to get a frontal face.")
        
        # Generate and save final result image (optional)
        # Rotate original image by final angle
        height, width = original_image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -final_rotation_angle, 1.0)  # Negative for clockwise rotation
        final_image = cv2.warpAffine(original_image, rotation_matrix, (width, height))
        
        cv2.imwrite("final_corrected_image.jpg", final_image)
        print(f"Final corrected image saved as 'final_corrected_image.jpg'.")
        
    else:
        print("Are you perhaps wearing a mask?")






