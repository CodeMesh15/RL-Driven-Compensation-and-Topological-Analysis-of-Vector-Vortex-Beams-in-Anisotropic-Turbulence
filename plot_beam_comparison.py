from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main():
    datasets = [
        (
            "Distorted by 60°C Turbulence",
            Path("results/alignment/60C_aligned_S1.npy"),
        ),
        (
            "AI Phase Mask Correction",
            Path("results/topological_ready/60C_corrected_S1.npy"),
        ),
        (
            "Ground Truth (Baseline Beam)",
            Path("stokes_matrices/only_beams/clean_S1.npy"),
        ),
    ]

    arrays = []
    for title, path in datasets:
        if not path.exists():
            raise FileNotFoundError(f"Missing input for {title}: {path}")
        arrays.append(np.load(path))

    vmin = min(float(np.min(array)) for array in arrays)
    vmax = max(float(np.max(array)) for array in arrays)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    image = None
    for ax, (title, _), array in zip(axes, datasets, arrays):
        image = ax.imshow(array, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    fig.colorbar(image, ax=axes, location="right", shrink=0.85, label="$S_1$")

    output_path = Path("results/beam_visual_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved beam comparison figure to {output_path}")


if __name__ == "__main__":
    main()
