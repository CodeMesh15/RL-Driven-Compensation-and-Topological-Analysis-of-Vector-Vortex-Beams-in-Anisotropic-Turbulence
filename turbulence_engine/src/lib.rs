use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Calculates a specific Zernike polynomial value for a given (r, theta)
fn zernike_mode(index: usize, r: f64, theta: f64) -> f64 {
    if r > 1.0 { return 0.0; }

    match index {
        0  => 1.0,                                      // Piston
        1  => 2.0 * r * theta.cos(),                    // Tilt X
        2  => 2.0 * r * theta.sin(),                    // Tilt Y
        3  => (3.0 * r.powi(2) - 1.0) * 3.0f64.sqrt(),  // Defocus
        4  => (r.powi(2)) * 6.0f64.sqrt() * (2.0 * theta).cos(), // Astigmatism (Oblique)
        5  => (r.powi(2)) * 6.0f64.sqrt() * (2.0 * theta).sin(), // Astigmatism (Vertical)
        6  => (3.0 * r.powi(3) - 2.0 * r) * 8.0f64.sqrt() * theta.cos(), // Coma (Horizontal)
        7  => (3.0 * r.powi(3) - 2.0 * r) * 8.0f64.sqrt() * theta.sin(), // Coma (Vertical)
        8  => r.powi(3) * 8.0f64.sqrt() * (3.0 * theta).cos(), // Trefoil (Oblique)
        9  => r.powi(3) * 8.0f64.sqrt() * (3.0 * theta).sin(), // Trefoil (Vertical)
        10 => (6.0 * r.powi(4) - 6.0 * r.powi(2) + 1.0) * 5.0f64.sqrt(), // Primary Spherical
        11 => (4.0 * r.powi(4) - 3.0 * r.powi(2)) * 10.0f64.sqrt() * (2.0 * theta).cos(), // 2nd Astig (Oblique)
        12 => (4.0 * r.powi(4) - 3.0 * r.powi(2)) * 10.0f64.sqrt() * (2.0 * theta).sin(), // 2nd Astig (Vertical)
        13 => r.powi(4) * 10.0f64.sqrt() * (4.0 * theta).cos(), // Tetrafoil (Oblique)
        _  => 0.0,
    }
}

#[pyfunction]
fn apply_turbulence<'py>(
    py: Python<'py>,
    base_matrix: PyReadonlyArray2<'py, f64>,    // Added exact lifetime marker
    zernike_coeffs: Vec<f64>,
) -> &'py PyArray2<f64> {
    
    let base_view = base_matrix.as_array();
    let shape = base_view.raw_dim();
    let height = shape[0];
    let width = shape[1];
    
    // THE FIX: Use numpy's built-in ndarray so the versions match perfectly
    let mut output_matrix = numpy::ndarray::Array2::<f64>::zeros(shape);

    // Loop through every pixel in the matrix
    for ((i, j), &pixel_value) in base_view.indexed_iter() {
        
        let x = (j as f64 / width as f64) * 2.0 - 1.0;
        let y = (i as f64 / height as f64) * 2.0 - 1.0;
        
        let r = (x * x + y * y).sqrt();
        let theta = y.atan2(x);

        let mut total_phase_shift = 0.0;
        for (idx, &coeff) in zernike_coeffs.iter().enumerate() {
            total_phase_shift += coeff * zernike_mode(idx, r, theta);
        }

        let distorted_intensity = pixel_value * total_phase_shift.cos();
        
        output_matrix[[i, j]] = distorted_intensity;
    }

    output_matrix.into_pyarray(py)
}

#[pymodule]
fn turbulence_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_turbulence, m)?)?;
    Ok(())
}