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

    /// Computes A @ B^T: (m×k) @ (n×k)^T = (m×n)
    /// Used in forward pass: input_batch @ W^T
    pub fn mul_abt(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.cols, b.cols);
        let (m, k, n) = (a.rows, a.cols, b.rows);
        let mut result = Matrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f64;
                for kk in 0..k {
                    sum += a.data[i * k + kk] * b.data[j * k + kk];
                }
                result.data[i * n + j] = sum;
            }
        }
        result
    }

    /// Computes A^T @ B: (n×m)^T @ (n×k) = (m×k)
    /// Used in backward pass: delta^T @ a_prev → dW
    pub fn mul_atb(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.rows, b.rows);
        let (n_rows, m, k) = (a.rows, a.cols, b.cols);
        let mut result = Matrix::new(m, k);
        for nn in 0..n_rows {
            for i in 0..m {
                let a_val = a.data[nn * m + i];
                for j in 0..k {
                    result.data[i * k + j] += a_val * b.data[nn * k + j];
                }
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

pub struct BatchLayerCache {
    pub z: Matrix,  // shape: (batch, n_out)
    pub a: Matrix,  // shape: (batch, n_out)
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

    /// Forward pass for a whole batch.
    /// input: Matrix (batch × n_in), returns one BatchLayerCache per layer.
    pub fn forward_batch(&self, input: &Matrix) -> Vec<BatchLayerCache> {
        assert_eq!(self.layers[0].n_in, input.cols);
        let batch = input.rows;
        let mut caches: Vec<BatchLayerCache> = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let prev_a: &Matrix = if i == 0 { input } else { &caches[i - 1].a };

            // z = prev_a @ W^T + b  →  (batch × n_out)
            let mut z = Matrix::mul_abt(prev_a, &layer.weights);
            for r in 0..batch {
                for c in 0..layer.n_out {
                    z.data[r * layer.n_out + c] += layer.biases.data[c];
                }
            }

            let a = if i < self.layers.len() - 1 {
                let data: Vec<f64> = z.data.iter().copied().map(relu).collect();
                Matrix::from_data(batch, layer.n_out, data)
            } else {
                let mut a_data = vec![0.0; batch * layer.n_out];
                for r in 0..batch {
                    let s = r * layer.n_out;
                    let probs = softmax(&z.data[s..s + layer.n_out]);
                    a_data[s..s + layer.n_out].copy_from_slice(&probs);
                }
                Matrix::from_data(batch, layer.n_out, a_data)
            };

            caches.push(BatchLayerCache { z, a });
        }
        caches
    }

    /// Backward pass for a whole batch. Returns already-averaged gradients.
    /// input: (batch × n_in), labels: (batch × n_out) one-hot
    pub fn backward_batch(
        &self,
        input: &Matrix,
        caches: &[BatchLayerCache],
        labels: &Matrix,
    ) -> Vec<LayerGrad> {
        let batch = input.rows;
        let n = self.layers.len();
        let scale = 1.0 / batch as f64;

        // initial delta: p - y  (batch × n_classes)
        let last_a = &caches.last().unwrap().a;
        let delta_data: Vec<f64> = last_a.data.iter().zip(labels.data.iter())
            .map(|(p, y)| p - y)
            .collect();
        let mut delta = Matrix::from_data(batch, self.layers.last().unwrap().n_out, delta_data);

        let mut layer_grads: Vec<LayerGrad> = self.layers.iter()
            .map(|l| LayerGrad::zero(l.n_out, l.n_in))
            .collect();

        for l in (0..n).rev() {
            let layer = &self.layers[l];
            let a_prev: &Matrix = if l == 0 { input } else { &caches[l - 1].a };

            // dW = delta^T @ a_prev / batch  (n_out × n_in)
            let mut dw = Matrix::mul_atb(&delta, a_prev);
            for v in dw.data.iter_mut() { *v *= scale; }

            // db = mean(delta, axis=0)  (n_out,)
            let mut db = vec![0.0; layer.n_out];
            for r in 0..batch {
                for c in 0..layer.n_out {
                    db[c] += delta.data[r * layer.n_out + c];
                }
            }
            for v in db.iter_mut() { *v *= scale; }

            if l > 0 {
                // delta_prev = (delta @ W) * relu_deriv(z_prev)
                let delta_prev_raw = delta.mul_matmat(&layer.weights);
                let z_prev = &caches[l - 1].z;
                let data: Vec<f64> = delta_prev_raw.data.iter()
                    .zip(z_prev.data.iter())
                    .map(|(da, z)| da * relu_deriv(*z))
                    .collect();
                delta = Matrix::from_data(batch, self.layers[l - 1].n_out, data);
            }

            layer_grads[l] = LayerGrad { dw, db };
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
