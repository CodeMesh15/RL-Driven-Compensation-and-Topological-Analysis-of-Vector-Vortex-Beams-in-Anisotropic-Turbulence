import numpy as np
import argparse
import csv
import os
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def align_beams(baseline_path, distorted_path, output_dir=None, label=None, plot=False):
    baseline = np.load(baseline_path)
    distorted = np.load(distorted_path)
    
    # Calculate the exact pixel shift 
    shift_vector, error, diffphase = phase_cross_correlation(
        baseline,
        distorted,
        upsample_factor=20,
    )

    # Physically shift the distorted matrix to match the baseline
    aligned_distorted = shift(distorted, shift_vector, mode="nearest")
    before = ssim(baseline, distorted, data_range=2.0)
    after = ssim(baseline, aligned_distorted, data_range=2.0)

    if output_dir and label:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"{label}_aligned_S1.npy", aligned_distorted)

    if plot and output_dir and label:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(baseline, cmap="inferno")
        axs[0].set_title("Master Baseline")
        axs[1].imshow(distorted, cmap="inferno")
        axs[1].set_title(f"Original {label}")
        axs[2].imshow(aligned_distorted, cmap="inferno")
        axs[2].set_title(f"Aligned {label}")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{label}_alignment.png", dpi=160)
        plt.close(fig)

    return {
        "label": label or Path(distorted_path).stem,
        "shift_y": float(shift_vector[0]),
        "shift_x": float(shift_vector[1]),
        "registration_error": float(error),
        "diffphase": float(diffphase),
        "ssim_before": float(before),
        "ssim_after_alignment": float(after),
        "alignment_gain": float(after - before),
    }


def main():
    parser = argparse.ArgumentParser(description="Check beam drift against the clean S1 baseline.")
    parser.add_argument("--baseline", default="stokes_matrices/clean/clean_S1.npy")
    parser.add_argument("--stokes_dir", default="stokes_matrices")
    parser.add_argument("--output_dir", default="results/alignment")
    parser.add_argument("--plot", action="store_true", help="Save before/after alignment plots.")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    stokes_dir = Path(args.stokes_dir)
    output_dir = Path(args.output_dir)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    rows = []
    for folder in sorted(path for path in stokes_dir.iterdir() if path.is_dir()):
        if folder.name == "clean":
            continue
        s1_path = folder / f"{folder.name}_S1.npy"
        if not s1_path.exists():
            print(f"Skipping {folder.name}: {s1_path} not found.")
            continue

        result = align_beams(
            baseline_path=baseline_path,
            distorted_path=s1_path,
            output_dir=output_dir,
            label=folder.name,
            plot=args.plot,
        )
        rows.append(result)
        wandered = abs(result["shift_y"]) >= 1 or abs(result["shift_x"]) >= 1
        verdict = "DRIFT" if wandered else "OK"
        print(
            f"{result['label']}: {verdict}, "
            f"shift_x={result['shift_x']:+.2f}px, shift_y={result['shift_y']:+.2f}px, "
            f"SSIM {result['ssim_before']:.4f} -> {result['ssim_after_alignment']:.4f} "
            f"(delta {result['alignment_gain']:+.4f})"
        )

    if rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "alignment_summary.csv"
        with summary_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved alignment summary to {summary_path}")
    else:
        print("No S1 matrices found to align.")
    

if __name__ == "__main__":
    main()
