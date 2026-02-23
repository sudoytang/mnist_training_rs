use rand_distr::{Distribution, Normal};
use anyhow::Result;
use ndarray::{Array1, Array2, Axis};

pub const EPS: f64 = 1e-12;

pub fn relu(x: f64) -> f64 {
    if x >= 0.0 { x } else { 0. }
}

pub fn relu_deriv(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

pub fn cross_entropy_loss(prob: &[f64], label: &[f64]) -> f64 {
    let label_index = label
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i).unwrap();
    -(prob[label_index] + EPS).ln()
}

pub struct FNNLayer {
    pub n_in: usize,
    pub n_out: usize,
    pub weights: Array2<f64>,    // (n_in, n_out) C-contiguous
    pub weights_t: Array2<f64>,  // (n_out, n_in) C-contiguous, kept in sync with weights
    pub biases: Array1<f64>,     // (n_out,)
}

impl FNNLayer {
    pub fn new(n_in: usize, n_out: usize) -> Result<Self> {
        // Kaiming He initialization: W ~ N(0, sqrt(2/n_in))
        let norm_distr = Normal::new(0., (2.0_f64 / n_in as f64).sqrt())?;
        let weights = norm_distr.sample_iter(rand::rng())
            .take(n_in * n_out)
            .collect();
        let weights = Array2::from_shape_vec((n_in, n_out), weights)?;
        // zeros() allocates C-contiguous; assign() copies data without changing layout
        let mut weights_t = Array2::<f64>::zeros((n_out, n_in));
        weights_t.assign(&weights.t());

        Ok(Self {
            n_in,
            n_out,
            weights,
            weights_t,
            biases: Array1::zeros(n_out),
        })
    }
}


pub struct FNN {
    pub layers: Vec<FNNLayer>,
}

pub struct BatchLayerCache {
    pub z: Array2<f64>,  // (batch, n_out)
    pub a: Array2<f64>,  // (batch, n_out)
}

pub struct LayerGrad {
    pub dw: Array2<f64>,  // (n_in, n_out)
    pub db: Array1<f64>,  // (n_out,)
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

    /// Forward pass for a whole batch.
    /// input: (batch × n_in), returns one BatchLayerCache per layer.
    pub fn forward_batch(&self, input: &Array2<f64>) -> Vec<BatchLayerCache> {
        assert_eq!(self.layers[0].n_in, input.ncols());
        let batch = input.nrows();
        let mut caches: Vec<BatchLayerCache> = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let prev_a: &Array2<f64> = if i == 0 { input } else { &caches[i - 1].a };

            // z = prev_a @ W + b  (batch × n_out)
            // both C-contiguous → BLAS DGEMM
            let mut z = prev_a.dot(&layer.weights);
            z += &layer.biases;  // broadcast bias to every row

            let a = if i < self.layers.len() - 1 {
                z.mapv(relu)
            } else {
                // softmax row-wise
                let mut a = Array2::zeros((batch, layer.n_out));
                for r in 0..batch {
                    let row = z.row(r);
                    let max = row.fold(f64::NEG_INFINITY, |acc, &v| acc.max(v));
                    let exp: Array1<f64> = row.mapv(|v| (v - max).exp());
                    let sum = exp.sum();
                    a.row_mut(r).assign(&(exp / sum));
                }
                a
            };

            caches.push(BatchLayerCache { z, a });
        }
        caches
    }

    /// Backward pass for a whole batch. Returns already-averaged gradients.
    /// input: (batch × n_in), labels: (batch × n_classes) one-hot
    pub fn backward_batch(
        &self,
        input: &Array2<f64>,
        caches: &[BatchLayerCache],
        labels: &Array2<f64>,
    ) -> Vec<LayerGrad> {
        let batch = input.nrows();
        let scale = 1.0 / batch as f64;
        let n = self.layers.len();

        // initial delta: p - y  (batch × n_classes)
        let mut delta = &caches.last().unwrap().a - labels;

        let mut layer_grads: Vec<Option<LayerGrad>> = (0..n).map(|_| None).collect();

        for l in (0..n).rev() {
            let layer = &self.layers[l];
            let a_prev: &Array2<f64> = if l == 0 { input } else { &caches[l - 1].a };

            // dW = a_prev^T @ delta / batch  (n_in × n_out)
            // zeros() → C-contiguous, assign() copies without changing layout → C × C → BLAS
            let mut a_prev_t = Array2::<f64>::zeros((a_prev.ncols(), a_prev.nrows()));
            a_prev_t.assign(&a_prev.t());
            let dw = a_prev_t.dot(&delta) * scale;

            // db = mean(delta, axis=0)  (n_out,)
            let db = delta.mean_axis(Axis(0)).unwrap();

            if l > 0 {
                // delta_prev = delta @ weights_t * relu_deriv(z_prev)  (batch × n_in)
                // weights_t is C-contiguous → C × C → BLAS
                let new_delta = delta.dot(&layer.weights_t) * caches[l - 1].z.mapv(relu_deriv);
                delta = new_delta;
            }

            layer_grads[l] = Some(LayerGrad { dw, db });
        }

        layer_grads.into_iter().map(|g| g.unwrap()).collect()
    }

    pub fn update(&mut self, grads: &[LayerGrad], lr: f64) {
        for (layer, grad) in self.layers.iter_mut().zip(grads.iter()) {
            layer.weights.scaled_add(-lr, &grad.dw);
            layer.weights_t.assign(&layer.weights.t());  // sync C-contiguous transpose
            layer.biases.scaled_add(-lr, &grad.db);
        }
    }
}
