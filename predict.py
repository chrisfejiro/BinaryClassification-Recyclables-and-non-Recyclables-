"""
predict.py – Batch prediction for images in 'input_images' folder
Automatically detects images and predicts recyclable vs non-recyclable.

Usage:
    python predict.py

The script will:
    1. Load the trained model from models/best_model.h5
    2. Process all images in the input_images/ folder
    3. Display predictions with confidence scores
    4. Show each image with its classification result

Confidence Interpretation:
    - >90%: Very high confidence (model is very certain)
    - 75-90%: High confidence (reliable prediction)
    - 60-75%: Moderate confidence (generally accurate)
    - <60%: Low confidence (consider manual verification)
    
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Configuration ---
MODEL_PATH = "models/best_model.h5"
INPUT_FOLDER = "input_images"
TARGET_SIZE = (224, 224)


def preprocess_image(img_path, target_size=TARGET_SIZE):
    """Load and preprocess a single image."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict_image(model, img_path):
    """Predict single image using the given model."""
    img, img_array = preprocess_image(img_path)
    confidence = float(model.predict(img_array)[0][0])
    label = "RECYCLABLE" if confidence >= 0.5 else "NON-RECYCLABLE"
    return img, label, confidence


def main():
    # Check model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at '{MODEL_PATH}'.")
        return

    # Load model
    print(f"Loading model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!\n")

    # Check input folder
    if not os.path.exists(INPUT_FOLDER):
        print(f"ERROR: Input folder '{INPUT_FOLDER}' not found.")
        return

    # List image files
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not image_files:
        print(f"No valid images found in '{INPUT_FOLDER}'.")
        return

    # Predict all images
    for img_file in image_files:
        img_path = os.path.join(INPUT_FOLDER, img_file)
        img, label, confidence = predict_image(model, img_path)
        print(f"{img_file}: {label} ({confidence:.4f})")
        plt.imshow(img)
        plt.title(f"{label} ({confidence:.2f})")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
