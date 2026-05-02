import os
from pathlib import Path

import numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

import fourier_engine
from evaluate_lab_dataset import generate_zernike_mask
from stokes_preprocess import calculate_stokes


TEMP_PREFIX = "60C"
CHANNELS = ("S0", "S1", "S2", "S3")


def ensure_control_stokes():
    control_dir = Path("stokes_matrices/only_beams")
    expected = [control_dir / f"clean_{channel}.npy" for channel in CHANNELS]
    if all(path.exists() for path in expected):
        print("Control Stokes set already exists in stokes_matrices/only_beams/.")
        return

    raw_dir = Path("raw_lab_data/Only beams")
    raw_paths = {
        "Ih": raw_dir / "clean_Ih.png",
        "Iv": raw_dir / "clean_Iv.png",
        "Id": raw_dir / "clean_Id.png",
        "Ia": raw_dir / "clean_Ia.png",
        "Ir": raw_dir / "clean_Ir.png",
        "Il": raw_dir / "clean_Il.png",
    }
    missing = [str(path) for path in raw_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot generate Control Stokes matrices. Missing raw files:\n  "
            + "\n  ".join(missing)
        )

    control_dir.mkdir(parents=True, exist_ok=True)
    calculate_stokes(
        str(raw_paths["Ih"]),
        str(raw_paths["Iv"]),
        str(raw_paths["Id"]),
        str(raw_paths["Ia"]),
        str(raw_paths["Ir"]),
        str(raw_paths["Il"]),
        str(control_dir),
        "clean",
    )
    print("Generated Control Stokes set in stokes_matrices/only_beams/.")


def align_full_stokes_set():
    baseline_path = Path("stokes_matrices/only_beams/clean_S1.npy")
    distorted_path = Path("stokes_matrices/60C/60C_S1.npy")
    alignment_dir = Path("results/alignment")
    alignment_dir.mkdir(parents=True, exist_ok=True)

    baseline = np.load(baseline_path)
    distorted = np.load(distorted_path)
    shift_vector, error, diffphase = phase_cross_correlation(
        baseline,
        distorted,
        upsample_factor=20,
    )

    print(
        "Detected 60C drift: "
        f"x={shift_vector[1]:+.2f}px, y={shift_vector[0]:+.2f}px "
        f"(registration error={error:.6f}, diffphase={diffphase:.6f})"
    )

    for channel in CHANNELS:
        input_path = Path(f"stokes_matrices/60C/60C_{channel}.npy")
        output_path = alignment_dir / f"60C_aligned_{channel}.npy"
        matrix = np.load(input_path)
        aligned = shift(matrix, shift_vector, mode="nearest")
        np.save(output_path, aligned)
        print(f"Saved aligned {channel}: {output_path}")

    print("Task 1 complete: full 60C Stokes set spatially aligned.")


def correct_full_stokes_set():
    zernike_history_path = Path("results/lab_dataset_aligned/60C_zernike_history.csv")
    if not zernike_history_path.exists():
        raise FileNotFoundError(f"Missing AI Zernike history: {zernike_history_path}")

    zernike_history = np.loadtxt(zernike_history_path, delimiter=",")
    if zernike_history.ndim == 1:
        final_coeffs = zernike_history
    else:
        final_coeffs = zernike_history[-1]

    phase_mask = generate_zernike_mask(final_coeffs, shape=(256, 256))
    alignment_dir = Path("results/alignment")
    output_dir = Path("results/topological_ready")
    output_dir.mkdir(parents=True, exist_ok=True)

    for channel in CHANNELS:
        input_path = alignment_dir / f"60C_aligned_{channel}.npy"
        output_path = output_dir / f"60C_corrected_{channel}.npy"
        matrix = np.load(input_path)
        corrected = fourier_engine.apply_fourier_phase_mask(matrix, phase_mask)
        np.save(output_path, corrected)
        print(f"Saved corrected {channel}: {output_path}")

    print("Task 2 complete: full 60C Stokes set AI-corrected for topology.")


def main():
    os.makedirs("results", exist_ok=True)
    print("Preparing thesis topological data for 60C...")
    ensure_control_stokes()
    align_full_stokes_set()
    correct_full_stokes_set()
    print("\nThesis data preparation complete.")
    print("You can now run: python topological_analysis.py")


if __name__ == "__main__":
    main()
