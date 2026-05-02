import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import numpy as np
from skimage.metrics import structural_similarity as ssim
from stable_baselines3 import PPO

from env import TurbulenceEnv
from stokes_preprocess import calculate_stokes
import fourier_engine


CHANNELS = ("Ih", "Iv", "Id", "Ia", "Ir", "Il")


def find_channel_file(folder, prefix, channel):
    candidates = sorted(folder.glob(f"{prefix}_{channel}.*"))
    if not candidates:
        candidates = sorted(folder.glob(f"*_{channel}.*"))
    if not candidates:
        raise FileNotFoundError(f"Missing {channel} file in {folder}")
    return candidates[0]


def has_all_channels(folder, prefix):
    return all(
        list(folder.glob(f"{prefix}_{channel}.*")) or list(folder.glob(f"*_{channel}.*"))
        for channel in CHANNELS
    )


def preprocess_folder(folder, output_root, prefix):
    output_dir = output_root / prefix
    s1_path = output_dir / f"{prefix}_S1.npy"
    if s1_path.exists():
        return s1_path

    files = [find_channel_file(folder, prefix, channel) for channel in CHANNELS]
    calculate_stokes(*(str(path) for path in files), str(output_dir), prefix)
    return s1_path


def load_matrix(path):
    matrix = np.load(path)
    if matrix.shape != (256, 256):
        raise ValueError(f"Expected 256x256 matrix at {path}, got {matrix.shape}")
    return matrix.astype(np.float64)


def metrics(reference, candidate):
    mse = float(np.mean((reference - candidate) ** 2))
    rmse = float(np.sqrt(mse))
    reference_flat = reference.ravel()
    candidate_flat = candidate.ravel()
    if np.std(reference_flat) == 0 or np.std(candidate_flat) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(reference_flat, candidate_flat)[0, 1])
    return {
        "ssim": float(ssim(reference, candidate, data_range=2.0)),
        "mse": mse,
        "rmse": rmse,
        "correlation": correlation,
    }


def zernike_coeffs_to_phase_mask(coeffs, shape):
    height, width = shape
    y = (np.arange(height, dtype=np.float64) / height) * 2.0 - 1.0
    x = (np.arange(width, dtype=np.float64) / width) * 2.0 - 1.0
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    mask = r <= 1.0

    basis = np.zeros((14, height, width), dtype=np.float64)
    basis[0] = 1.0
    basis[1] = 2.0 * r * np.cos(theta)
    basis[2] = 2.0 * r * np.sin(theta)
    basis[3] = (3.0 * r**2 - 1.0) * np.sqrt(3.0)
    basis[4] = r**2 * np.sqrt(6.0) * np.cos(2.0 * theta)
    basis[5] = r**2 * np.sqrt(6.0) * np.sin(2.0 * theta)
    basis[6] = (3.0 * r**3 - 2.0 * r) * np.sqrt(8.0) * np.cos(theta)
    basis[7] = (3.0 * r**3 - 2.0 * r) * np.sqrt(8.0) * np.sin(theta)
    basis[8] = r**3 * np.sqrt(8.0) * np.cos(3.0 * theta)
    basis[9] = r**3 * np.sqrt(8.0) * np.sin(3.0 * theta)
    basis[10] = (6.0 * r**4 - 6.0 * r**2 + 1.0) * np.sqrt(5.0)
    basis[11] = (4.0 * r**4 - 3.0 * r**2) * np.sqrt(10.0) * np.cos(2.0 * theta)
    basis[12] = (4.0 * r**4 - 3.0 * r**2) * np.sqrt(10.0) * np.sin(2.0 * theta)
    basis[13] = r**4 * np.sqrt(10.0) * np.cos(4.0 * theta)
    basis *= mask

    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if coeffs.shape[0] != 14:
        raise ValueError(f"Expected 14 Zernike coefficients, got {coeffs.shape[0]}")
    return np.tensordot(coeffs, basis, axes=(0, 0))


def generate_zernike_mask(coeffs, shape=(256, 256)):
    return zernike_coeffs_to_phase_mask(coeffs, shape)


def iterative_correct(model, env, distorted, baseline, steps):
    current = distorted.copy()
    best = current.copy()
    best_step = 0
    action_history = []
    best_metrics = metrics(baseline, current)

    for step in range(1, steps + 1):
        obs = env._get_obs(current)
        action, _ = model.predict(obs, deterministic=True)
        action_phase_mask = zernike_coeffs_to_phase_mask(action, current.shape)
        current = fourier_engine.apply_fourier_phase_mask(current, action_phase_mask)
        action_history.append(action.reshape(-1))

        step_metrics = metrics(baseline, current)
        if step_metrics["ssim"] > best_metrics["ssim"]:
            best = current.copy()
            best_step = step
            best_metrics = step_metrics

    return best, best_step, best_metrics, np.asarray(action_history)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model on all lab temperature folders.")
    parser.add_argument("--raw_dir", default="raw_lab_data")
    parser.add_argument("--output_dir", default="stokes_matrices")
    parser.add_argument("--results_dir", default="results/lab_dataset")
    parser.add_argument("--model", default="ppo_turbulence_compensator.zip")
    parser.add_argument("--ground_truth_folder", default="Only beams")
    parser.add_argument("--ground_truth_prefix", default="clean")
    parser.add_argument("--steps", type=int, default=20, help="Iterative correction steps per temperature.")
    parser.add_argument(
        "--aligned_dir",
        help="Optional folder containing pre-aligned *_aligned_S1.npy distorted inputs.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    aligned_dir = Path(args.aligned_dir) if args.aligned_dir else None
    results_dir.mkdir(parents=True, exist_ok=True)

    gt_folder = raw_dir / args.ground_truth_folder
    baseline_s1_path = preprocess_folder(gt_folder, output_dir, args.ground_truth_prefix)
    baseline = load_matrix(baseline_s1_path)

    env = TurbulenceEnv(baseline_matrix=baseline)
    model = PPO.load(args.model, env=env, device="cpu")

    rows = []
    for folder in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
        if folder.name == args.ground_truth_folder:
            continue
        prefix = folder.name
        if not has_all_channels(folder, prefix):
            print(f"Skipping {folder.name}: missing one or more polarization channels.")
            continue

        if aligned_dir:
            distorted_s1_path = aligned_dir / f"{prefix}_aligned_S1.npy"
            if not distorted_s1_path.exists():
                print(f"Skipping {folder.name}: aligned input not found at {distorted_s1_path}.")
                continue
        else:
            distorted_s1_path = preprocess_folder(folder, output_dir, prefix)
        distorted = load_matrix(distorted_s1_path)
        before_metrics = metrics(baseline, distorted)
        corrected, best_step, after_metrics, action_history = iterative_correct(
            model=model,
            env=env,
            distorted=distorted,
            baseline=baseline,
            steps=args.steps,
        )
        improvement = after_metrics["ssim"] - before_metrics["ssim"]

        np.save(results_dir / f"{prefix}_corrected_S1.npy", corrected)
        np.savetxt(results_dir / f"{prefix}_zernike_history.csv", action_history, delimiter=",")

        rows.append(
            {
                "temperature": prefix,
                "best_step": best_step,
                "ssim_before": before_metrics["ssim"],
                "ssim_after": after_metrics["ssim"],
                "improvement": improvement,
                "rmse_before": before_metrics["rmse"],
                "rmse_after": after_metrics["rmse"],
                "correlation_before": before_metrics["correlation"],
                "correlation_after": after_metrics["correlation"],
                "distorted_s1": str(distorted_s1_path),
                "corrected_s1": str(results_dir / f"{prefix}_corrected_S1.npy"),
                "zernikes": str(results_dir / f"{prefix}_zernike_history.csv"),
            }
        )

        print(
            f"{prefix}: best step {best_step}/{args.steps}, "
            f"SSIM {before_metrics['ssim']:.4f} -> {after_metrics['ssim']:.4f} "
            f"(delta {improvement:+.4f})"
        )

    summary_path = results_dir / "summary.csv"
    if rows:
        with summary_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved summary to {summary_path}")
    else:
        print("\nNo complete temperature folders were found to evaluate.")


if __name__ == "__main__":
    main()
