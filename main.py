import os
import cv2
import face_recognition
from dotenv import load_dotenv
from gpu_pipeline import initialize_pipeline, process_with_pipeline
from PIL import Image
import numpy as np

# Load environment variables
load_dotenv()

# Folder paths
INPUT_DIR = "input"
OUTPUT_DIR = "processed_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detect mouth region
def detect_mouth_region(image_path):
    image = face_recognition.load_image_file(image_path)
    original_image = cv2.imread(image_path)
    face_landmarks_list = face_recognition.face_landmarks(image)

    if not face_landmarks_list:
        print("No face detected in the image.")
        return None, None

    for face_landmarks in face_landmarks_list:
        top_lip = face_landmarks.get("top_lip", [])
        bottom_lip = face_landmarks.get("bottom_lip", [])
        mouth_points = top_lip + bottom_lip

        if mouth_points:
            x_min = min([p[0] for p in mouth_points])
            y_min = min([p[1] for p in mouth_points])
            x_max = max([p[0] for p in mouth_points])
            y_max = max([p[1] for p in mouth_points])

            mouth_region = original_image[y_min:y_max, x_min:x_max]
            return mouth_region, (x_min, y_min, x_max, y_max)

    return None, None

# Blend the enhanced region back into the image
def blend_regions(original, enhanced, bbox, alpha=0.8):
    x_min, y_min, x_max, y_max = bbox
    original_region = original[y_min:y_max, x_min:x_max]
    blended = cv2.addWeighted(original_region, alpha, enhanced, 1 - alpha, 0.5)
    original[y_min:y_max, x_min:x_max] = blended
    return original

# Main workflow
def main():
    input_image_path = os.path.join(INPUT_DIR, "input.jpg")
    if not os.path.exists(input_image_path):
        print(f"Error: {input_image_path} not found. Please place 'input.jpg' in the 'input/' folder.")
        return

    print(f"Processing input image: {input_image_path}")
    mouth_region, bounding_box = detect_mouth_region(input_image_path)

    if mouth_region is None:
        print("Failed to detect mouth region.")
        return

    # Initialize the pipeline
    pipe = initialize_pipeline()

    # Preprocess the mouth region
    mouth_image = Image.fromarray(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB))

    # Enhance the mouth region using the AI model
    prompt = "Make the teeth evenly aligned and subtly whitened, blending perfectly with the surrounding skin"
    enhanced_mouth = process_with_pipeline(pipe, mouth_image, prompt, num_steps=50, guidance_scale=1.8)

    if enhanced_mouth:
        # Convert enhanced image back to OpenCV format
        enhanced_mouth = cv2.cvtColor(np.array(enhanced_mouth), cv2.COLOR_RGB2BGR)

        # Resize enhanced mouth to fit the original bounding box
        x_min, y_min, x_max, y_max = bounding_box
        enhanced_mouth_resized = cv2.resize(enhanced_mouth, (x_max - x_min, y_max - y_min))

        # Blend with the original image
        original_image = cv2.imread(input_image_path)
        blended_image = blend_regions(original_image, enhanced_mouth_resized, bounding_box)

        # Save results
        enhanced_mouth_path = os.path.join(OUTPUT_DIR, "enhanced_mouth.jpg")
        final_image_path = os.path.join(OUTPUT_DIR, "final_image.jpg")
        cv2.imwrite(enhanced_mouth_path, enhanced_mouth_resized)
        cv2.imwrite(final_image_path, blended_image)
        print(f"Enhanced mouth region saved at: {enhanced_mouth_path}")
        print(f"Final image saved at: {final_image_path}")
    else:
        print("Failed to enhance the mouth region.")

if __name__ == "__main__":
    main()
