use rand_distr::{Distribution, Normal};
use anyhow::Result;

pub const EPS: f64 = 1e-12;

pub fn relu(x: f64) -> f64 {
    if x >= 0.0 { x } else { 0. }
}

pub fn relu_deriv(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let mut logits: Vec<f64> = logits.iter().copied().collect();
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.;
    for logit in logits.iter_mut() {
        *logit -= max;
        *logit = f64::exp(*logit);
        sum += *logit;
    }
    for logit in logits.iter_mut() {
        *logit /= sum;
    }
    logits
}

pub fn cross_entropy_loss(prob: &[f64], label: &[f64]) -> f64 {
    let label_index = label
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap();
    -(prob[label_index] + EPS).ln()
}

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.; rows * cols],
        }
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        Self {
            rows,
            cols,
            data,
        }
    }

    pub fn from_row_vec(data: Vec<f64>) -> Self {
        let rows = data.len();
        let cols = 1;
        Self {
            rows,
            cols,
            data,
        }
    }

    pub fn from_col_vec(data: Vec<f64>) -> Self {
        let rows = 1;
        let cols = data.len();
        Self {
            rows,
            cols,
            data,
        }
    }

    pub fn at(&self, row: usize, col: usize) -> &f64 {
        &self.data[row * self.cols + col]
    }

    pub fn at_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        &mut self.data[row * self.cols + col]
    }

    pub fn row(&self, row: usize) -> &[f64] {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [f64] {
        &mut self.data[row * self.cols..(row + 1) * self.cols]
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn mul_matmat(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows); // - |
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    *result.at_mut(i, j) += self.at(i, k) * other.at(k, j);
                }
            }
        }
        result
    }

    pub fn mul_matcol(&self, other: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, other.len());
        let mut result = vec![0.; self.rows];
        for i in 0..self.rows {
            for k in 0..self.cols {
                result[i] += self.at(i, k) * other[k];
            }
        }
        result
    }

    pub fn mul_rowmat(this: &[f64], other: &Matrix) -> Vec<f64> {
        assert_eq!(this.len(), other.rows);
        let mut result = vec![0.; other.cols];
        for i in 0..other.cols {
            for k in 0..this.len() {
                result[i] += this[k] * other.at(k, i);
            }
        }
        result
    }

    pub fn mul_rowcol(this: &[f64], other: &[f64]) -> f64 {
        assert_eq!(this.len(), other.len());
        this.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn mul_colrow(this: &[f64], other: &[f64]) -> Matrix {
        let mut result = Matrix::new(this.len(), other.len());
        for i in 0..this.len() {
            for j in 0..other.len() {
                *result.at_mut(i, j) += this[i] * other[j];
            }
        }
        result
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                *result.at_mut(i, j) += self.at(i, j) + other.at(i, j);
            }
        }

        result
    }

    pub fn mul_scalar(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                *result.at_mut(i, j) += self.at(i, j) * scalar;
            }
        }
        result
    }

}




pub struct FNNLayer {
    pub n_in: usize,
    pub n_out: usize,
    pub weights: Matrix,  // shape: (n_out, n_in)
    pub biases: Matrix,   // shape: (n_out)
}

impl FNNLayer {
    pub fn new(n_in: usize, n_out: usize) -> Result<Self> {
        // Kaiming He initialization: W ~ N(0, sqrt(2/n_in))
        let norm_distr = Normal::new(0., (2.0_f64 / n_in as f64).sqrt())?;
        let weights = norm_distr.sample_iter(rand::rng())
            .take(n_out * n_in)
            .collect();
        let weights = Matrix::from_data(n_out, n_in, weights);

        Ok(Self {
            n_in,
            n_out,
            weights,
            biases: Matrix::from_data(n_out, 1, vec![0.; n_out]),
        })
    }

    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        // z = Wx + b
        let mut z =self.weights.mul_matcol(x);
        z.iter_mut().enumerate()
            .for_each(|(i, z)| *z += self.biases.at(i, 0));
        z
    }

}


pub struct FNN {
    pub layers: Vec<FNNLayer>,
}

#[derive(Default)]
pub struct LayerCache {
    pub z: Vec<f64>,    // shape: (n_out)
    pub a: Vec<f64>,    // shape: (n_out)
}

pub struct LayerGrad {
    pub dw: Matrix,     // shape: (n_out, n_in)
    pub db: Vec<f64>,   // shape: (n_out)
}

impl LayerGrad {
    pub fn zero(n_out: usize, n_in: usize) -> Self {
        Self { dw: Matrix::new(n_out, n_in), db: vec![0.0; n_out] }
    }

    pub fn accumulate(&mut self, other: &LayerGrad) {
        for (a, b) in self.dw.data.iter_mut().zip(other.dw.data.iter()) {
            *a += b;
        }
        for (a, b) in self.db.iter_mut().zip(other.db.iter()) {
            *a += b;
        }
    }

    pub fn scale(&mut self, factor: f64) {
        for v in self.dw.data.iter_mut() { *v *= factor; }
        for v in self.db.iter_mut() { *v *= factor; }
    }
}

impl FNN {
    pub fn new(layer_dim: &[usize]) -> Result<Self> {
        assert!(layer_dim.len() >= 2);
        let layer_size = layer_dim.len() - 1;
        let mut layers = Vec::new();
        for i in 0..layer_size {
            let n_in = layer_dim[i];
            let n_out = layer_dim[i + 1];
            layers.push(FNNLayer::new(n_in, n_out)?);
        }

        Ok(Self {
            layers
        })
    }

    pub fn forward(&self, input: &[f64]) -> Vec<LayerCache> {
        assert_eq!(self.layers[0].n_in, input.len());
        let mut caches: Vec<LayerCache> = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            if i == 0 {
                let z = layer.forward(input);
                let a = z.iter().copied().map(relu).collect();
                caches.push(LayerCache { z, a });
            } else if i < self.layers.len() - 1 {
                let z = layer.forward(&caches[i - 1].a);
                let a = z.iter().copied().map(relu).collect();
                caches.push(LayerCache { z, a });
            } else {
                let z = layer.forward(&caches[i - 1].a);
                let a = softmax(&z);
                caches.push(LayerCache { z, a });
            }
        }
        caches
    }

    pub fn backward(&self, input: &[f64], mut caches: Vec<LayerCache>, label: &[f64]) -> Vec<LayerGrad> {
        assert_eq!(caches.len(), self.layers.len());
        let n = self.layers.len();
        let mut layer_grads: Vec<LayerGrad> = Vec::new();
        for i in 0..n {
            let n_out = self.layers[i].n_out;
            let n_in = self.layers[i].n_in;
            layer_grads.push(LayerGrad { dw: Matrix::new(n_out, n_in), db: vec![0.; n_out] });
        }

        // output layer: softmax + cross-entropy combined gradient = p - y
        let last_cache = caches.pop().unwrap();
        let mut delta: Vec<f64> = last_cache.a.iter().zip(label).map(|(p, y)| p - y).collect();

        for l in (0..n).rev() {
            let layer = &self.layers[l];
            let a_prev: &[f64] = if l == 0 { input } else { &caches[l - 1].a };

            // dW[i][j] = outer product delta * a
            let dw = Matrix::mul_colrow(&delta, a_prev);

            layer_grads[l] = LayerGrad { dw, db: delta.clone() };

            if l > 0 {
                // da_prev = W^T · delta = delta^T · W
                let da_prev = Matrix::mul_rowmat(&delta, &layer.weights);
                delta = da_prev.iter().zip(caches[l - 1].z.iter())
                    .map(|(da, z)| da * relu_deriv(*z))
                    .collect();
            }
        }
        layer_grads
    }

    pub fn update(&mut self, grads: &[LayerGrad], learning_rate: f64) {
        for (layer, grad) in self.layers.iter_mut().zip(grads.iter()) {
            for (w, dw) in layer.weights.data.iter_mut().zip(grad.dw.data.iter()) {
                *w -= learning_rate * dw;
            }
            for (b, db) in layer.biases.data.iter_mut().zip(grad.db.iter()) {
                *b -= learning_rate * db;
            }
        }
    }

}
