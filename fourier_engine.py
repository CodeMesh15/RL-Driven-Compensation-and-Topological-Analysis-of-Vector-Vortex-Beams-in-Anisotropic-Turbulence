import numpy as np

def apply_fourier_phase_mask(intensity_matrix, zernike_phase_mask):
    """
    Applies a Zernike phase mask using proper Pupil/Focal plane wave propagation.
    """
    # 1. Estimate Electric Field Amplitude 
    E_focal = np.sqrt(np.abs(intensity_matrix)) 
    
    # 2. Propagate to Pupil Plane (2D Fast Fourier Transform)
    E_pupil = np.fft.fftshift(np.fft.fft2(E_focal))
    
    # 3. Apply the Zernike Phase Mask
    E_pupil_corrected = E_pupil * np.exp(1j * zernike_phase_mask)
    
    # 4. Propagate back to Camera Focal Plane (Inverse 2D FFT)
    E_focal_corrected = np.fft.ifft2(np.fft.ifftshift(E_pupil_corrected))
    
    # 5. Calculate Final Intensity (Amplitude Squared)
    corrected_intensity = np.abs(E_focal_corrected)**2
    
    # Normalize back to match original Stokes S1 scale
    max_val = np.max(np.abs(intensity_matrix))
    if max_val > 0:
        corrected_intensity = (corrected_intensity / np.max(corrected_intensity)) * max_val
        
    return corrected_intensity