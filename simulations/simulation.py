import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def make_grid(N, L):
    """Creates the spatial coordinate grid."""
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    return X, Y, rho, theta

def create_vector_vortex_beam(N, L, w0=0.3, charge=1):
    """
    Generates a Vector Vortex Beam (Radial Polarization).
    Mathematically, this is a superposition of two scalar vortex beams 
    with opposite topological charges and orthogonal polarizations.
    """
    X, Y, rho, theta = make_grid(N, L)
    
    # Base Amplitude (LG01 approximation)
    amplitude = (rho / w0) * np.exp(-(rho**2) / (w0**2))
    
    # Helical phase for the underlying scalar components
    phase_plus = np.exp(1j * charge * theta)
    phase_minus = np.exp(-1j * charge * theta)
    
    # Superposition to create Radial Polarization
    Ex = amplitude * (phase_plus + phase_minus) / 2
    Ey = amplitude * (phase_plus - phase_minus) / (2 * 1j)
    
    return Ex, Ey

def create_aniso_phase_screen(N, L, r0=0.1, alpha=3.67, mu_x=2.0, mu_y=1.0):
    """
    Generates Anisotropic Non-Kolmogorov Turbulence Phase Screen.
    mu_x and mu_y define the elliptical stretch (the thermal plumes).
    """
    df = 1.0 / L
    f = np.arange(-N/2, N/2) * df
    Fx, Fy = np.meshgrid(f, f)
    Fx[Fx==0] = 1e-10; Fy[Fy==0] = 1e-10 # Singularity fix
    
    # Anisotropic Power Spectrum (Symmetry Breaking)
    K_sq = (Fx * mu_x)**2 + (Fy * mu_y)**2
    PSD = 0.023 * (r0**(-5/3)) * (K_sq + (1/100)**2)**(-alpha/2)
    
    noise = np.random.normal(0, 1, (N, N)) + 1j * np.random.normal(0, 1, (N, N))
    screen = np.real(ifft2(ifftshift(noise * np.sqrt(PSD)))) * N**2
    return screen

def propagate_vector_beam(Ex, Ey, phase_screen, N, L, distance=200, wavelength=633e-9):
    """Propagates both electric field components through the turbulence."""
    # Apply Turbulence Phase
    Ex_distorted = Ex * np.exp(1j * phase_screen)
    Ey_distorted = Ey * np.exp(1j * phase_screen)
    
    # Angular Spectrum Propagation
    k = 2 * np.pi / wavelength
    df = 1.0 / L
    f = np.arange(-N/2, N/2) * df
    Fx, Fy = np.meshgrid(f, f)
    
    H = np.exp(-1j * (Fx**2 + Fy**2) * (np.pi * wavelength * distance))
    
    Ex_final = ifft2(ifftshift(fftshift(fft2(Ex_distorted)) * H))
    Ey_final = ifft2(ifftshift(fftshift(fft2(Ey_distorted)) * H))
    
    return Ex_final, Ey_final


# VIRTUAL POLARIMETRY

def get_stokes_parameters(Ex, Ey):
    """Converts complex vector fields into measurable Stokes Parameters."""
    Ex_star = np.conj(Ex)
    Ey_star = np.conj(Ey)
    
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(Ex * Ey_star)
    S3 = 2 * np.imag(Ex * Ey_star)
    
    # Normalize 
    norm = np.max(S0) + 1e-9
    return S0/norm, S1/norm, S2/norm, S3/norm

# ==========================================
# 3. RUN SIMULATION & MAP TO IMAGES
# ==========================================
if __name__ == "__main__":
    N = 128
    L = 1.0
    
    # 1. Generate Perfect VVB
    Ex, Ey = create_vector_vortex_beam(N, L, charge=1)
    scalar_phase = np.angle(Ex + 1j*Ey) # For visualization
    
    # 2. Generate Anisotropic Turbulence (The "Hot Water" plumes)
    turbulence = create_aniso_phase_screen(N, L, alpha=3.1, mu_x=3.0, mu_y=1.0)
    
    # 3. Propagate Uncorrected Beam (To show the damage)
    Ex_dist, Ey_dist = propagate_vector_beam(Ex, Ey, turbulence, N, L)
    S0_dist, S1_dist, S2_dist, S3_dist = get_stokes_parameters(Ex_dist, Ey_dist)
    
    # 4. Propagate Corrected Beam (Ideal Pre-Compensation / The "Shield")
    # We perfectly subtract the turbulence to simulate an ideal AI/Optical correction
    ideal_correction_phase = -turbulence
    Ex_corr, Ey_corr = propagate_vector_beam(Ex, Ey, turbulence + ideal_correction_phase, N, L)
    S0_corr, S1_corr, S2_corr, S3_corr = get_stokes_parameters(Ex_corr, Ey_corr)

    # --- VISUALIZATION MAPPED TO THE 5 IMAGES ---
    
    plt.figure(figsize=(15, 8))
    
    # IMAGE 1: The Physics Engine Output
    plt.subplot(2, 3, 1)
    plt.title("Image 1: Original Vortex Phase")
    plt.imshow(scalar_phase, cmap='twilight')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Image 1: Turbulence Phase Screen")
    plt.imshow(turbulence, cmap='jet')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("Image 1: Distorted Intensity")
    plt.imshow(S0_dist, cmap='gray')
    plt.axis('off')
    
    # IMAGES 2 & 4: The Stokes Parameters (The Clovers)
    plt.subplot(2, 3, 4)
    plt.title("Image 2 & 4: S1 Distorted Clover")
    plt.imshow(S1_dist, cmap='bwr', vmin=-1, vmax=1)
    plt.axis('off')
    
    # IMAGE 3: The Correction Phase
    plt.subplot(2, 3, 5)
    plt.title("Image 3: AI Correction Phase")
    plt.imshow(ideal_correction_phase, cmap='jet')
    plt.axis('off')
    
    # IMAGE 5: The Corrected Beam
    plt.subplot(2, 3, 6)
    plt.title("Image 5: Corrected Beam (S0)")
    plt.imshow(S0_corr, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
