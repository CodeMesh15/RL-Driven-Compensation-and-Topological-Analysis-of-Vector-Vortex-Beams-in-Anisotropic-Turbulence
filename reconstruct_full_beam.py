from pathlib import Path

import numpy as np

from evaluate_lab_dataset import generate_zernike_mask


def reconstruct_stokes_parameters(
    aligned_dir="results/alignment",
    zernike_history_path="results/lab_dataset_aligned/60C_zernike_history.csv",
    output_dir="results/topological_ready",
    prefix="60C",
):
    aligned_dir = Path(aligned_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    S0 = np.load(aligned_dir / f"{prefix}_aligned_S0.npy")
    S1 = np.load(aligned_dir / f"{prefix}_aligned_S1.npy")
    S2 = np.load(aligned_dir / f"{prefix}_aligned_S2.npy")
    S3 = np.load(aligned_dir / f"{prefix}_aligned_S3.npy")

    Ex_amp = np.sqrt(np.clip((S0 + S1) / 2.0, 0, None))
    Ey_amp = np.sqrt(np.clip((S0 - S1) / 2.0, 0, None))

    delta = np.arctan2(S3, S2)

    Ex = Ex_amp * np.exp(1j * 0)
    Ey = Ey_amp * np.exp(1j * delta)

    Ex_pupil = np.fft.fftshift(np.fft.fft2(Ex))
    Ey_pupil = np.fft.fftshift(np.fft.fft2(Ey))

    zernike_history = np.loadtxt(zernike_history_path, delimiter=",")
    if zernike_history.ndim == 1:
        coeffs = zernike_history
    else:
        coeffs = zernike_history[-1]
    phase_mask = generate_zernike_mask(coeffs, shape=S0.shape)

    Ex_pupil_corr = Ex_pupil * np.exp(1j * phase_mask)
    Ey_pupil_corr = Ey_pupil * np.exp(1j * phase_mask)

    Ex_corr = np.fft.ifft2(np.fft.ifftshift(Ex_pupil_corr))
    Ey_corr = np.fft.ifft2(np.fft.ifftshift(Ey_pupil_corr))

    S0_new = np.abs(Ex_corr) ** 2 + np.abs(Ey_corr) ** 2
    S1_new = np.abs(Ex_corr) ** 2 - np.abs(Ey_corr) ** 2
    S2_new = 2 * np.real(Ex_corr * np.conj(Ey_corr))
    S3_new = 2 * np.imag(Ex_corr * np.conj(Ey_corr))

    outputs = {
        "S0": S0_new,
        "S1": S1_new,
        "S2": S2_new,
        "S3": S3_new,
    }
    for channel, matrix in outputs.items():
        path = output_dir / f"{prefix}_corrected_{channel}.npy"
        np.save(path, matrix)
        print(f"Saved {path}")

    print("Vector-field Stokes reconstruction complete.")


if __name__ == "__main__":
    reconstruct_stokes_parameters()
