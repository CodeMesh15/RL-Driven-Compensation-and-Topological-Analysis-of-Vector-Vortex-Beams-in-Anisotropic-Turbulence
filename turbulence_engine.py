import functools

import numpy as np


@functools.lru_cache(maxsize=8)
def _zernike_basis(height, width):
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
    return basis


def apply_turbulence(base_matrix, zernike_coeffs):
    base = np.asarray(base_matrix, dtype=np.float64)
    height, width = base.shape
    coeffs = np.asarray(zernike_coeffs, dtype=np.float64)
    phase_shift = np.tensordot(coeffs, _zernike_basis(height, width), axes=(0, 0))
    return base * np.cos(phase_shift)
