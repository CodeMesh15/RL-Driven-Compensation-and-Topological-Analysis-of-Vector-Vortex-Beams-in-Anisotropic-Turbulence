import numpy as np
import cv2
import os
import argparse
from PIL import Image

def process_file(filepath):
    """Detects file type and returns a cleanly formatted, precision-preserved 2D array."""
    print(f"Processing {os.path.basename(filepath)}...")
    
    # 1. VIDEO FILES (.avi, .mp4, .mov)
    if filepath.lower().endswith(('.avi', '.mp4', '.mov')):
        cap = cv2.VideoCapture(filepath)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Could not read any frames from video: {filepath}")
            
        # Time-averaging: Collapse the video into one incredibly clean image
        img_array = np.mean(frames, axis=0) 
        img = Image.fromarray(img_array.astype(np.uint8))
        print(f"  -> Averaged {len(frames)} video frames.")
        
    # 2. SCIENTIFIC TIFF FILES (.tif, .tiff)
    elif filepath.lower().endswith(('.tif', '.tiff')):
        # IMREAD_UNCHANGED prevents OpenCV from crushing 16-bit data down to 8-bit
        img_array = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        
        if img_array is None:
            raise ValueError(f"Could not read TIFF file: {filepath}")
            
        # If the camera saved it as RGB by mistake, convert to grayscale
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            
        img = Image.fromarray(img_array)
        print("  -> Loaded as precision 16-bit TIFF.")

    # 3. STANDARD IMAGES (.png, .jpg)
    else:
        # Pillow handles standard 8-bit images perfectly fine
        img = Image.open(filepath).convert("L")
        print("  -> Loaded as standard 8-bit image.")
    
    # --- Standardize Formatting (Crop & Resize) ---
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    
    # Crop to a perfect square to prevent warping the vortex
    img = img.crop((left, top, right, bottom))
    
    # Resize to the exact 256x256 shape the Neural Network expects
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    matrix = np.array(img, dtype=np.float64)
    
    # --- Smart Normalization ---
    # If the max value exceeds 255, we know it's a 16-bit image and scale accordingly
    if matrix.max() > 255.0:
        matrix = matrix / 65535.0
    else:
        matrix = matrix / 255.0
        
    return matrix

def calculate_stokes(Ih_path, Iv_path, Id_path, Ia_path, Ir_path, Il_path, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all 6 files (handles .avi, .tif, and .png automatically)
    Ih = process_file(Ih_path)
    Iv = process_file(Iv_path)
    Id = process_file(Id_path)
    Ia = process_file(Ia_path)
    Ir = process_file(Ir_path)
    Il = process_file(Il_path)
    
    # Stokes Math
    S0 = Ih + Iv
    # Adding 1e-10 prevents division by zero in pure dark spots
    S1 = (Ih - Iv) / (S0 + 1e-10)
    S2 = (Id - Ia) / (S0 + 1e-10)
    S3 = (Ir - Il) / (S0 + 1e-10)
    
    # Save the resulting matrices
    np.save(os.path.join(output_dir, f"{prefix}_S0.npy"), S0)
    np.save(os.path.join(output_dir, f"{prefix}_S1.npy"), S1)
    np.save(os.path.join(output_dir, f"{prefix}_S2.npy"), S2)
    np.save(os.path.join(output_dir, f"{prefix}_S3.npy"), S3)
    
    print(f"\nSuccess! Saved S0, S1, S2, and S3 to {output_dir}")
    print(f"S1 Range Check: [{np.min(S1):.2f}, {np.max(S1):.2f}] (Should be between -1.0 and 1.0)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 6 intensity files (.tif, .png, or .avi) into Stokes matrices.")
    parser.add_argument("--Ih", required=True, help="Path to Horizontal file")
    parser.add_argument("--Iv", required=True, help="Path to Vertical file")
    parser.add_argument("--Id", required=True, help="Path to Diagonal file")
    parser.add_argument("--Ia", required=True, help="Path to Anti-Diagonal file")
    parser.add_argument("--Ir", required=True, help="Path to Right-Circular file")
    parser.add_argument("--Il", required=True, help="Path to Left-Circular file")
    parser.add_argument("--out_dir", default="./stokes_matrices", help="Output directory")
    parser.add_argument("--prefix", default="reading", help="Prefix for saved files (e.g., '80C')")
    args = parser.parse_args()
    
    calculate_stokes(args.Ih, args.Iv, args.Id, args.Ia, args.Ir, args.Il, args.out_dir, args.prefix)
