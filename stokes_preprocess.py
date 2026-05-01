import numpy as np
import cv2
import os
import argparse
from PIL import Image

def process_file(filepath):
    """Detects if a file is an image or video, and returns a clean 2D numpy array."""
    print(f"Processing {os.path.basename(filepath)}...")
    
    # 1. Check if it is a video file
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
            
        # TIME-AVERAGING: Collapse the video into one incredibly clean, noiseless image
        img_array = np.mean(frames, axis=0).astype(np.uint8)
        img = Image.fromarray(img_array)
        print(f"  -> Averaged {len(frames)} video frames into one clean image.")
        
    # 2. Otherwise, treat it as a static image
    else:
        img = Image.open(filepath).convert("L")
    
    # 3. Standardize the formatting (Crop & Resize)
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    img = img.crop((left, top, right, bottom))
    
    # Resize to 256x256
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    
    # 4. Normalize to 0.0 -> 1.0
    matrix = np.array(img, dtype=np.float64) / 255.0
    return matrix

def calculate_stokes(Ih_path, Iv_path, Id_path, Ia_path, Ir_path, Il_path, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all 6 files (handles both .avi and .png automatically)
    Ih = process_file(Ih_path)
    Iv = process_file(Iv_path)
    Id = process_file(Id_path)
    Ia = process_file(Ia_path)
    Ir = process_file(Ir_path)
    Il = process_file(Il_path)
    
    # Stokes Math
    S0 = Ih + Iv
    S1 = (Ih - Iv) / (S0 + 1e-10)
    S2 = (Id - Ia) / (S0 + 1e-10)
    S3 = (Ir - Il) / (S0 + 1e-10)
    
    # Save the resulting matrices
    np.save(os.path.join(output_dir, f"{prefix}_S0.npy"), S0)
    np.save(os.path.join(output_dir, f"{prefix}_S1.npy"), S1)
    np.save(os.path.join(output_dir, f"{prefix}_S2.npy"), S2)
    np.save(os.path.join(output_dir, f"{prefix}_S3.npy"), S3)
    
    print(f"\n✅ Success! Saved S0, S1, S2, and S3 to {output_dir}")
    print(f"S1 Range: [{np.min(S1):.2f}, {np.max(S1):.2f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 6 intensity files (.png or .avi) into Stokes matrices.")
    parser.add_argument("--Ih", required=True, help="Path to Horizontal file")
    parser.add_argument("--Iv", required=True, help="Path to Vertical file")
    parser.add_argument("--Id", required=True, help="Path to Diagonal file")
    parser.add_argument("--Ia", required=True, help="Path to Anti-Diagonal file")
    parser.add_argument("--Ir", required=True, help="Path to Right-Circular file")
    parser.add_argument("--Il", required=True, help="Path to Left-Circular file")
    parser.add_argument("--out_dir", default="./stokes_data", help="Output directory")
    parser.add_argument("--prefix", default="reading", help="Prefix for saved files")
    args = parser.parse_args()
    
    calculate_stokes(args.Ih, args.Iv, args.Id, args.Ia, args.Ir, args.Il, args.out_dir, args.prefix)