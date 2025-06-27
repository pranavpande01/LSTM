use crate::operations;
use ndarray::{Array1, ArrayD};

pub fn fully_connected(tensor: Vec<f64>, weights: Vec<Vec<f64>>) -> ArrayD<f64> {
    let mut output = Vec::new();
    for i in 0..weights.len() {
        output.push(operations::dot_product(&tensor, &weights[i]));
    }
    Array1::from(output).into_dyn()
}
