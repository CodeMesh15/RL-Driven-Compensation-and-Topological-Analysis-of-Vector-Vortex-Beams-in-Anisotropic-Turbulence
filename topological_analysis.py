import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, map_coordinates


def calculate_concurrence(S0, S1, S2, S3):
    """
    Calculates the Global Concurrence (Classical Entanglement) of the beam.
    Uses the spatial integration of Stokes parameters to measure non-separability.
    """
    threshold = 0.05 * np.max(S0)
    mask = S0 > threshold

    s0_m = S0[mask]
    s1_m = S1[mask]
    s2_m = S2[mask]
    s3_m = S3[mask]

    p_mag = np.sqrt(s1_m**2 + s2_m**2 + s3_m**2)
    safe_p = np.where(p_mag == 0, 1e-10, p_mag)
    overshoot = p_mag > s0_m

    s1_fixed = np.where(overshoot, s1_m * (s0_m / safe_p), s1_m)
    s2_fixed = np.where(overshoot, s2_m * (s0_m / safe_p), s2_m)
    s3_fixed = np.where(overshoot, s3_m * (s0_m / safe_p), s3_m)

    sum_S0 = np.sum(s0_m)
    sum_S1 = np.sum(s1_fixed)
    sum_S2 = np.sum(s2_fixed)
    sum_S3 = np.sum(s3_fixed)

    s1_global = sum_S1 / sum_S0
    s2_global = sum_S2 / sum_S0
    s3_global = sum_S3 / sum_S0

    DoP = np.sqrt(s1_global**2 + s2_global**2 + s3_global**2)
    DoP = np.clip(DoP, 0.0, 1.0)
    return np.sqrt(1 - DoP**2)


def plot_pb_phase(S1, S2, title, save_path=None):
    """
    Extracts the Pancharatnam-Berry (PB) phase map to reveal V-points and C-points.
    """
    pb_phase = 0.5 * np.arctan2(S2, S1 + 1e-10)

    plt.figure(figsize=(6, 5))
    plt.imshow(pb_phase, cmap="hsv", extent=[-1, 1, -1, 1])
    plt.colorbar(label="PB Phase (Orientation Angle)")
    plt.title(f"Topological Phase Map: {title}")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def extract_oam_spectrum(S0, S1, S2, radius=60, title="Extracted OAM Spectrum", save_path=None):
    """
    Extracts the relative OAM spectrum by taking the azimuthal Fourier transform
    of the complex Stokes field (S1 + iS2).
    """
    cy, cx = center_of_mass(S0)
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)

    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    ring_S1 = map_coordinates(S1, [y, x], order=1)
    ring_S2 = map_coordinates(S2, [y, x], order=1)
    complex_field = ring_S1 + 1j * ring_S2

    oam_spectrum = np.abs(np.fft.fftshift(np.fft.fft(complex_field)))
    max_value = np.max(oam_spectrum)
    if max_value > 0:
        oam_spectrum = oam_spectrum / max_value

    l_modes = np.arange(-128, 128)

    plt.figure(figsize=(8, 4))
    plt.bar(l_modes, oam_spectrum, color="purple")
    plt.xlim(-8, 8)
    plt.xlabel("Relative OAM Mode (Delta l)")
    plt.ylabel("Normalized Intensity")
    plt.title(title)
    plt.xticks(np.arange(-8, 9, 2))
    plt.grid(axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_stokes_set(folder, prefix):
    folder = Path(folder)
    paths = {
        "S0": folder / f"{prefix}_S0.npy",
        "S1": folder / f"{prefix}_S1.npy",
        "S2": folder / f"{prefix}_S2.npy",
        "S3": folder / f"{prefix}_S3.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Stokes matrices:\n  " + "\n  ".join(missing))

    return tuple(np.load(paths[key]) for key in ("S0", "S1", "S2", "S3"))


if __name__ == "__main__":
    plot_dir = Path("results/topological_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        ("Control", Path("stokes_matrices/only_beams"), "clean"),
        ("Disease", Path("results/alignment"), "60C_aligned"),
        ("Cure", Path("results/topological_ready"), "60C_corrected"),
    ]

    for stage_name, folder, prefix in stages:
        print(f"\n--- {stage_name}: {folder} / {prefix} ---")
        try:
            S0, S1, S2, S3 = load_stokes_set(folder, prefix)
        except FileNotFoundError as error:
            print(error)
            print(f"Skipping {stage_name} until the full S0/S1/S2/S3 set exists.")
            continue

        concurrence = calculate_concurrence(S0, S1, S2, S3)
        print(f"{stage_name} Concurrence: {concurrence:.4f}")

        plot_pb_phase(
            S1,
            S2,
            title=f"{stage_name} PB Phase",
            save_path=plot_dir / f"{stage_name}_PB_Phase.png",
        )
        extract_oam_spectrum(
            S0,
            S1,
            S2,
            title=f"{stage_name} OAM Spectrum",
            save_path=plot_dir / f"{stage_name}_OAM_Spectrum.png",
        )

    print(f"\nTopological plots saved to {plot_dir}")
