# pip install numpy==1.26.4
# pip install imgaug

import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia

# Set seed for reproducibility
ia.seed(42)

# Input/output directories
input_folder = "data/not_labubu_cleaned"
output_folder = "data/not_labubu_augmented"
os.makedirs(output_folder, exist_ok=True)

# Define augmentation pipeline
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),                       # Horizontal flip
    iaa.Flipud(0.2),                       # Vertical flip
    iaa.Affine(
        rotate=(-25, 25),                 # Random rotation
        scale=(0.9, 1.1),                 # Slight zoom
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Shift
    ),
    iaa.AddToBrightness((-20, 20)),       # Brightness variation
    iaa.AddToHueAndSaturation((-15, 15)), # Hue and saturation variation
    iaa.GaussianBlur(sigma=(0.0, 1.0)),   # Optional blur
    iaa.LinearContrast((0.75, 1.25)),     # Contrast change
])

# Number of augmentations per image
N_AUG = 3

# Process each image
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping unreadable file: {filename}")
        continue

    # Convert BGR to RGB for imgaug
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply augmentations
    images_aug = augmenters(images=[image_rgb] * N_AUG)

    base_name = os.path.splitext(filename)[0]
    for i, aug_img in enumerate(images_aug):
        # Convert RGB back to BGR for saving with OpenCV
        aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        out_filename = f"{base_name}_aug{i+1}.jpg"
        out_path = os.path.join(output_folder, out_filename)
        cv2.imwrite(out_path, aug_bgr)

print("Image augmentation complete.")
