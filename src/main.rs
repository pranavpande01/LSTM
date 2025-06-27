use crate::activation::sigmoid;

mod activation;
mod layers;
mod operations;



fn main() {
    let tensor: Vec<f64> = vec![1.0, 2.0, 3.0];
    let weights: Vec<Vec<f64>> = vec![
        vec![0.5, 0.5, 0.5],   
        vec![1.0, 0.0, -1.0], 
        vec![10.0, 02.0, -11.0],  
        
    ];
    
    let result = layers::fully_connected(tensor, weights);
    println!("{:?}", result);
}
