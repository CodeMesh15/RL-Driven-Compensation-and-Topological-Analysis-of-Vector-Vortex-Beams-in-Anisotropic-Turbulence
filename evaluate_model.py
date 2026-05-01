import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from stable_baselines3 import PPO

from env import TurbulenceEnv
import turbulence_engine


def load_matrix(path):
    matrix = np.load(path)
    if matrix.shape != (256, 256):
        raise ValueError(f"Expected a 256x256 matrix, got {matrix.shape}")
    return matrix.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained turbulence compensator.")
    parser.add_argument("--model", default="ppo_turbulence_final.zip")
    parser.add_argument("--baseline", help="Optional clean/reference 256x256 .npy matrix.")
    parser.add_argument("--distorted", help="Optional distorted/test 256x256 .npy matrix.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="Display before/after visual comparison.")
    parser.add_argument("--save_dir", default="./results", help="Directory to save extracted matrices and coefficients.")
    args = parser.parse_args()

    # Create results folder if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    baseline = load_matrix(args.baseline) if args.baseline else None
    env = TurbulenceEnv(baseline_matrix=baseline)
    model = PPO.load(args.model, env=env, device="cpu")

    scores_before = []
    scores_after = []

    for i in range(args.episodes):
        if args.distorted:
            distorted = load_matrix(args.distorted)
            env.current_state = distorted
            obs = env._get_obs(distorted)
        else:
            obs, _ = env.reset()
            distorted = env.current_state

        # 1. Ask the AI for the Zernike fix
        action, _ = model.predict(obs, deterministic=True)
        
        # 2. Apply the fix via the Rust engine
        corrected = turbulence_engine.apply_turbulence(distorted, action.tolist())

        # 3. Calculate SSIM
        before = ssim(env.baseline, distorted, data_range=1.0)
        after = ssim(env.baseline, corrected, data_range=1.0)
        scores_before.append(before)
        scores_after.append(after)

        # --- NEW: Save the Outputs for Topological Analysis ---
        base_name = f"episode_{i}"
        if args.distorted:
            base_name = os.path.basename(args.distorted).replace(".npy", "")

        np.save(os.path.join(args.save_dir, f"{base_name}_corrected.npy"), corrected)
        np.savetxt(os.path.join(args.save_dir, f"{base_name}_zernikes.csv"), action, delimiter=",")

        print(f"--- Episode {i+1} ---")
        print(f"SSIM Improved: {before:.4f} -> {after:.4f}")
        print(f"Extracted Zernikes saved to {args.save_dir}")

        # --- NEW: Plotting for your Thesis ---
        if args.plot:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(env.baseline, cmap='inferno')
            axs[0].set_title("Cold Baseline")
            axs[1].imshow(distorted, cmap='inferno')
            axs[1].set_title(f"Distorted (SSIM: {before:.2f})")
            axs[2].imshow(corrected, cmap='inferno')
            axs[2].set_title(f"AI Corrected (SSIM: {after:.2f})")
            plt.show()

    print("\n--- FINAL METRICS ---")
    print(f"Mean SSIM before correction: {np.mean(scores_before):.4f}")
    print(f"Mean SSIM after correction:  {np.mean(scores_after):.4f}")
    print(f"Mean improvement:            {np.mean(scores_after) - np.mean(scores_before):.4f}")


if __name__ == "__main__":
    main()