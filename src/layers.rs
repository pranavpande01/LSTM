use ndarray::{ArrayD, Ix1, Ix2};
//use ndarray::{ArrayD, Ix1, Ix2, Array1, Array2};

pub fn fully_connected(input: ArrayD<f64>, weights: ArrayD<f64>, bias: ArrayD<f64>) -> ArrayD<f64> {
    let input = input.into_dimensionality::<Ix1>().unwrap();
    let weights = weights.into_dimensionality::<Ix2>().unwrap();
    let bias = bias.into_dimensionality::<Ix1>().unwrap();

    let result = weights.dot(&input) + &bias;
    result.into_dyn()
}
