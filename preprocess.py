import numpy as np
from PIL import Image
import os
import argparse

def process_lab_image(image_path, output_name):
    # 1. Load the raw lab image
    print(f"Loading {image_path}...")
    img = Image.open(image_path)
    
    # 2. Convert to Grayscale (Intensity only, no RGB)
    img = img.convert("L")
    
    # 3. Crop to a Square (Assuming the beam is roughly in the center)
    # This prevents the beam from getting "squished" when resizing
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    img = img.crop((left, top, right, bottom))
    
    # 4. Resize to exactly 256x256
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    
    # 5. Convert to Numpy Array and Normalize to 0.0 - 1.0
    # Your raw pixels are 0-255. The AI expects small floats.
    matrix = np.array(img, dtype=np.float64)
    matrix = matrix / 255.0 
    
    # 6. Save as .npy
    save_path = f"{output_name}.npy"
    np.save(save_path, matrix)
    print(f"Success! Saved formatted matrix to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert lab images to AI-ready .npy matrices.")
    parser.add_argument("input_image", help="Path to your raw lab image (e.g., reading_80C.png)")
    parser.add_argument("output_name", help="Name for the output .npy file (without extension)")
    args = parser.parse_args()
    
    process_lab_image(args.input_image, args.output_name)