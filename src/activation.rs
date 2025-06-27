use ndarray::ArrayD;

pub fn sigmoid(mut tensor: ArrayD<f64>) -> ArrayD<f64> {
    tensor.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
    tensor
}
