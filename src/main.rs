mod activation;
mod layers;
mod operations;

use ndarray::{Array1, Array2, ArrayD};

fn main() {
    // === Prepare inputs ===
    let tensor: ArrayD<f64> = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn(); // 1D input

    let weights: Array2<f64> = Array2::from_shape_vec(
        (3, 3), // 3 output neurons, 3 input features
        vec![
            0.5, 0.5, 0.5,   // neuron 1
            1.0, 0.0, -1.0,  // neuron 2
            10.0, 2.0, -11.0 // neuron 3
        ]
    ).unwrap();
    let weights: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<ndarray::IxDynImpl>> = weights.into_dyn();

    let bias: ArrayD<f64> = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();

    // === Apply fully connected layer ===
    let result = layers::fully_connected(tensor, weights, bias);

    println!("Fully connected result: {:?}", activation::sigmoid(result));
}
