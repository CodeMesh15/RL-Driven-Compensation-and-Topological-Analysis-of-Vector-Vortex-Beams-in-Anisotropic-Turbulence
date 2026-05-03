import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from skimage.metrics import structural_similarity as ssim

from evaluate_lab_dataset import generate_zernike_mask
from reconstruct_full_beam import reconstruct_stokes_parameters


CHANNELS = ("S0", "S1", "S2", "S3")


def load_stokes(folder, prefix, suffix):
    folder = Path(folder)
    stem = f"{prefix}_{suffix}" if suffix else prefix
    return {
        channel: np.load(folder / f"{stem}_{channel}.npy")
        for channel in CHANNELS
    }


def stokes_to_fields(stokes):
    S0 = stokes["S0"]
    S1 = stokes["S1"]
    S2 = stokes["S2"]
    S3 = stokes["S3"]

    Ex_amp = np.sqrt(np.clip((S0 + S1) / 2.0, 0, None))
    Ey_amp = np.sqrt(np.clip((S0 - S1) / 2.0, 0, None))
    delta = np.arctan2(S3, S2)

    Ex = Ex_amp.astype(np.complex128)
    Ey = Ey_amp * np.exp(1j * delta)
    return np.fft.fftshift(np.fft.fft2(Ex)), np.fft.fftshift(np.fft.fft2(Ey))


def fields_to_stokes(Ex_pupil, Ey_pupil, phase_mask):
    correction = np.exp(1j * phase_mask)
    Ex_corr = np.fft.ifft2(np.fft.ifftshift(Ex_pupil * correction))
    Ey_corr = np.fft.ifft2(np.fft.ifftshift(Ey_pupil * correction))

    return {
        "S0": np.abs(Ex_corr) ** 2 + np.abs(Ey_corr) ** 2,
        "S1": np.abs(Ex_corr) ** 2 - np.abs(Ey_corr) ** 2,
        "S2": 2 * np.real(Ex_corr * np.conj(Ey_corr)),
        "S3": 2 * np.imag(Ex_corr * np.conj(Ey_corr)),
    }


def sanitize_for_metric(stokes, reference_mask=None):
    S0 = stokes["S0"]
    max_s0 = np.max(np.abs(S0)) + 1e-10
    mask = reference_mask if reference_mask is not None else S0 > 0.05 * np.max(S0)

    cleaned = {
        "S0": np.where(mask, S0 / max_s0, 0.0),
        "S1": np.where(mask, np.clip(stokes["S1"], -1.0, 1.0), 0.0),
        "S2": np.where(mask, np.clip(stokes["S2"], -1.0, 1.0), 0.0),
        "S3": np.where(mask, np.clip(stokes["S3"], -1.0, 1.0), 0.0),
    }
    return cleaned


def full_stokes_score(reference, candidate):
    reference_mask = reference["S0"] > 0.05 * np.max(reference["S0"])
    reference = sanitize_for_metric(reference, reference_mask)
    candidate = sanitize_for_metric(candidate, reference_mask)

    ssim_scores = [
        ssim(reference[channel], candidate[channel], data_range=2.0)
        for channel in CHANNELS
    ]
    rmse_scores = [
        np.sqrt(np.mean((reference[channel] - candidate[channel]) ** 2))
        for channel in CHANNELS
    ]
    return float(np.mean(ssim_scores) - 0.15 * np.mean(rmse_scores))


def main():
    parser = argparse.ArgumentParser(description="Directly optimize 14 Zernike coefficients for aligned 60C.")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--bound", type=float, default=1.5)
    parser.add_argument("--output_dir", default="results/optimized_60C")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_stokes("stokes_matrices/only_beams", "clean", "")
    aligned = load_stokes("results/alignment", "60C", "aligned")
    Ex_pupil, Ey_pupil = stokes_to_fields(aligned)

    def objective(coeffs):
        phase_mask = generate_zernike_mask(coeffs, shape=aligned["S0"].shape)
        corrected = fields_to_stokes(Ex_pupil, Ey_pupil, phase_mask)
        return -full_stokes_score(reference, corrected)

    print("Starting direct Zernike optimization for 60C...")
    result = differential_evolution(
        objective,
        bounds=[(-args.bound, args.bound)] * 14,
        maxiter=args.iterations,
        popsize=args.popsize,
        polish=True,
        seed=42,
        updating="immediate",
        workers=1,
        disp=True,
    )

    coeffs = result.x
    np.savetxt(output_dir / "60C_optimized_zernikes.csv", coeffs[None, :], delimiter=",")
    print(f"Best optimization score: {-result.fun:.6f}")
    print(f"Saved optimized coefficients to {output_dir / '60C_optimized_zernikes.csv'}")

    reconstruct_stokes_parameters(
        zernike_history_path=output_dir / "60C_optimized_zernikes.csv",
        output_dir=output_dir / "topological_ready",
    )

    corrected = load_stokes(output_dir / "topological_ready", "60C", "corrected")
    rows = []
    for channel in CHANNELS:
        before_rmse = float(np.sqrt(np.mean((reference[channel] - aligned[channel]) ** 2)))
        after_rmse = float(np.sqrt(np.mean((reference[channel] - corrected[channel]) ** 2)))
        before_ssim = float(ssim(reference[channel], aligned[channel], data_range=2.0))
        after_ssim = float(ssim(reference[channel], corrected[channel], data_range=2.0))
        rows.append(
            {
                "channel": channel,
                "ssim_before": before_ssim,
                "ssim_after": after_ssim,
                "rmse_before": before_rmse,
                "rmse_after": after_rmse,
            }
        )

    with (output_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics to {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
