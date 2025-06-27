//use ndarray::{ArrayD, Array1, Array2, Ix1, Ix2, Axis};
use ndarray::{ArrayD, Array1, Ix1, Ix2};
/// Concatenates two 1D tensors into one 1D tensor
pub fn concat_1d(a: &ArrayD<f64>, b: &ArrayD<f64>) -> ArrayD<f64> {
    let a = a.clone().into_dimensionality::<Ix1>().unwrap();
    let b = b.clone().into_dimensionality::<Ix1>().unwrap();
    let mut combined = Vec::new();
    combined.extend(a.iter());
    combined.extend(b.iter());
    Array1::from(combined).into_dyn()
}

/// Dot product between 1D and 2D (weights.dot(input))
pub fn dot_2d_1d(weights: &ArrayD<f64>, input: &ArrayD<f64>) -> ArrayD<f64> {
    let input = input.clone().into_dimensionality::<Ix1>().unwrap();
    let weights = weights.clone().into_dimensionality::<Ix2>().unwrap();
    weights.dot(&input).into_dyn()
}
