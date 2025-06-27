use crate::activation::sigmoid;

mod activation;
mod layers;
mod operations;



fn main() {
    let tensor = vec![1.0, 2.0, 3.0];
    let weights = vec![
        vec![0.5, 0.5, 0.5],   // neuron 1
        vec![1.0, 0.0, -1.0],  // neuron 2
    ];
    
    let result = layers::fully_connected(tensor, weights);
    println!("{:?}", result);  // Output: [3.0, -2.0]
}
