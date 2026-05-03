from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim

from evaluate_lab_dataset import generate_zernike_mask
from optimize_60c_zernikes import fields_to_stokes, stokes_to_fields
from reconstruct_full_beam import reconstruct_stokes_parameters


def load_stokes(folder, stem):
    folder = Path(folder)
    return {
        channel: np.load(folder / f"{stem}_{channel}.npy")
        for channel in ("S0", "S1", "S2", "S3")
    }


def main():
    output_dir = Path("results/refined_60C_s1")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_stokes("stokes_matrices/only_beams", "clean")
    aligned = load_stokes("results/alignment", "60C_aligned")
    Ex_pupil, Ey_pupil = stokes_to_fields(aligned)

    history = np.loadtxt("results/lab_dataset_aligned_specialist_60C/60C_zernike_history.csv", delimiter=",")
    start = history[-1] if history.ndim > 1 else history

    def objective(coeffs):
        phase_mask = generate_zernike_mask(coeffs, shape=aligned["S0"].shape)
        corrected = fields_to_stokes(Ex_pupil, Ey_pupil, phase_mask)
        s1_ssim = ssim(reference["S1"], corrected["S1"], data_range=2.0)
        s1_rmse = np.sqrt(np.mean((reference["S1"] - corrected["S1"]) ** 2))
        return -(s1_ssim - 0.03 * s1_rmse)

    print("Refining specialist 60C coefficients for S1 visual recovery...")
    result = minimize(
        objective,
        start,
        method="Powell",
        bounds=[(-2.0, 2.0)] * 14,
        options={"maxiter": 80, "xtol": 1e-3, "ftol": 1e-4, "disp": True},
    )

    coeffs = result.x
    np.savetxt(output_dir / "60C_refined_zernikes.csv", coeffs[None, :], delimiter=",")
    reconstruct_stokes_parameters(
        zernike_history_path=output_dir / "60C_refined_zernikes.csv",
        output_dir=output_dir / "topological_ready",
    )

    corrected = load_stokes(output_dir / "topological_ready", "60C_corrected")
    before_ssim = ssim(reference["S1"], aligned["S1"], data_range=2.0)
    after_ssim = ssim(reference["S1"], corrected["S1"], data_range=2.0)
    before_rmse = np.sqrt(np.mean((reference["S1"] - aligned["S1"]) ** 2))
    after_rmse = np.sqrt(np.mean((reference["S1"] - corrected["S1"]) ** 2))
    print(f"S1 SSIM: {before_ssim:.4f} -> {after_ssim:.4f}")
    print(f"S1 RMSE: {before_rmse:.4f} -> {after_rmse:.4f}")


if __name__ == "__main__":
    main()
