import face_recognition
import cv2
import os

# Folder structure paths
INPUT_DIR = "input"
OUTPUT_DIR = "processed_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_image(image_path):
    """
    Process the image to detect facial landmarks, annotate the mouth region, and crop the mouth area.
    """
    # Load the image
    image = face_recognition.load_image_file(image_path)

    # Detect facial landmarks
    face_landmarks_list = face_recognition.face_landmarks(image)

    if not face_landmarks_list:
        print("No face detected in the image.")
        return None, None

    # Annotate the image and extract the mouth region
    annotated_image = image.copy()
    mouth_region = None

    for face_landmarks in face_landmarks_list:
        # Get mouth landmarks
        top_lip = face_landmarks.get("top_lip", [])
        bottom_lip = face_landmarks.get("bottom_lip", [])
        mouth_points = top_lip + bottom_lip

        if mouth_points:
            # Annotate mouth landmarks
            for point in mouth_points:
                cv2.circle(annotated_image, point, 2, (0, 255, 0), -1)

            # Compute bounding box for the mouth region
            x_min = min([p[0] for p in mouth_points])
            y_min = min([p[1] for p in mouth_points])
            x_max = max([p[0] for p in mouth_points])
            y_max = max([p[1] for p in mouth_points])

            # Crop the mouth region
            mouth_region = image[y_min:y_max, x_min:x_max]

    # Save the annotated image
    annotated_image_path = os.path.join(OUTPUT_DIR, "processed_image.jpg")
    cv2.imwrite(annotated_image_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"Annotated image saved at: {annotated_image_path}")

    # Save the cropped mouth region
    if mouth_region is not None:
        mouth_image_path = os.path.join(OUTPUT_DIR, "mouth_region.jpg")
        cv2.imwrite(mouth_image_path, cv2.cvtColor(mouth_region, cv2.COLOR_RGB2BGR))
        print(f"Mouth region saved at: {mouth_image_path}")

    return annotated_image_path, mouth_image_path


def main():
    """
    Main function to process the input image, annotate facial landmarks, and crop the mouth region.
    """
    # Input image path
    input_image_path = os.path.join(INPUT_DIR, "input.jpg")

    if not os.path.exists(input_image_path):
        print(f"Error: {input_image_path} not found. Please place 'input.jpg' in the 'input/' folder.")
        return

    print(f"Processing input image: {input_image_path}")
    annotated_image_path, mouth_image_path = process_image(input_image_path)

    if annotated_image_path and mouth_image_path:
        print(f"Processing complete. Annotated image: {annotated_image_path}, Mouth region: {mouth_image_path}")


if __name__ == "__main__":
    main()
