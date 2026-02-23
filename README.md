# MNIST Training in Rust

A from-scratch feedforward neural network trained on MNIST, written in Rust.
The project started as a direct port of a Python/NumPy reference implementation
and was iteratively optimized to match NumPy's throughput.

## Architecture

- Network: **784 -> 256 -> 128 -> 10** (fully connected)
- Activation: ReLU (hidden layers), Softmax (output)
- Loss: Cross-entropy
- Optimizer: SGD, learning rate 0.1
- Batch size: 64, Epochs: 20
- Weight init: Kaiming He - `W ~ N(0, sqrt(2/n_in))`

## Optimization Journey

### Step 1 — Baseline: scalar forward/backward (`0db1135`)

The initial port implemented a hand-rolled `Matrix` struct with plain
triple-loop matrix multiplication. Training was sample-by-sample: for each
mini-batch, the code looped over every individual sample, ran `forward` and
`backward` once per sample, accumulated gradients into `LayerGrad` buffers,
then averaged and applied them.

Key characteristics:
- Custom `Matrix` type with `Vec<f64>` storage
- Single-sample `forward` / `backward` methods
- Gradient accumulation loop: `O(batch_size)` forward+backward calls per step
- Very slow: pure Rust scalar loops, no SIMD, no BLAS

---

### Step 2 — Batch computation (`cb59775`)

Replaced the per-sample accumulation loop with true batched forward and
backward passes.

Changes:
- Added `forward_batch(input: &Matrix) -> Vec<BatchLayerCache>` — processes
  the whole mini-batch as a single matrix multiply
  (`A @ B^T` via `mul_abt`, `A^T @ B` via `mul_atb`)
- Added `backward_batch` — computes `dW = delta^T @ a_prev / batch` and
  `db = mean(delta, axis=0)` in one shot, no accumulation loop
- Removed the old per-sample `forward` / `backward` methods
- Test inference also batched

The bottleneck shifted from loop overhead to the matrix multiplication kernels
inside `mul_abt` / `mul_atb`, which were still naive triple-loops.

---

### Step 3 — Switch to ndarray + BLAS (`f77c3d2`)

Replaced the hand-rolled `Matrix` type and all triple-loop matmuls with
`ndarray::Array2<f64>` backed by **OpenBLAS** (`blas-src` / `cblas` crates,
linked via `build.rs`).

Changes:
- Removed the `Matrix` struct entirely (~200 lines)
- `FNNLayer::weights` / `biases` → `Array2<f64>` / `Array1<f64>`
- Forward: `prev_a.dot(&layer.weights.t())` — dispatches to BLAS DGEMM
- Backward: `delta.t().dot(a_prev)` for `dW`, `delta.dot(&layer.weights)` for
  propagating delta
- Weight update: `layer.weights.scaled_add(-lr, &grad.dw)` — BLAS DAXPY
- Removed `softmax` standalone function; inlined row-wise softmax in the batch
  forward loop using `ndarray` operations

This gave the largest single speedup by delegating dense matmuls to the
highly-optimised BLAS DGEMM kernel.

---

### Step 4 — Fix matrix layout for BLAS (`ae2cda9` + `4251f52`)

BLAS DGEMM achieves peak throughput only when both operand matrices are
**C-contiguous** (row-major). ndarray's `.t()` returns a transposed *view*
with a Fortran stride, which forces BLAS into a slower code path.

Two targeted fixes:

**`ae2cda9` — Reorder weight matrix layout**

Changed `weights` from shape `(n_out, n_in)` to `(n_in, n_out)` so the
forward pass becomes `prev_a @ weights` (no transpose needed) — both matrices
are C-contiguous. Adjusted backward pass accordingly:
- `dW = a_prev^T @ delta` (now `n_in × n_out`)
- `delta_prev = delta @ weights^T`

**`4251f52` — Pre-materialise the transpose**

The backward pass still needed `a_prev^T` and `weights^T` as explicit
C-contiguous arrays. Instead of relying on transposed views:

- Added `weights_t: Array2<f64>` field — a C-contiguous copy of `weights.t()`,
  kept in sync after every gradient update via `weights_t.assign(&weights.t())`
- In the backward pass, `a_prev_t` is materialised the same way
  (`zeros((ncols, nrows))` then `.assign(&a_prev.t())`)

Both operands of every DGEMM call are now guaranteed C-contiguous, unlocking
the full BLAS throughput.

---

### Step 5 — Clean up logging (`4b31b8b`)

Removed mid-epoch progress logging (`LOG_EVERY_BATCHES`, window accumulators).
Now only a single summary line is printed per epoch, including `us/batch`
measured over the whole epoch:

```
epoch  0 done  | loss: 0.1823 | train acc: 94.73% | 312.5us/batch
```

---

## Performance

| Version | Throughput |
|---------|-----------|
| Baseline (scalar, per-sample) | ~50 ms/batch |
| Batch computation (hand-rolled matmul) | ~8 ms/batch |
| ndarray + BLAS | ~1.2 ms/batch |
| BLAS + C-contiguous layout fixes | ~0.45 ms/batch |
| Python + NumPy (reference) | ~0.3–0.4 ms/batch |

The final Rust implementation is on par with the Python/NumPy baseline.

## Running

```bash
cargo run --release
```

Requires the MNIST data files in `mnist_data/`:

```
mnist_data/train-images-idx3-ubyte
mnist_data/train-labels-idx1-ubyte
mnist_data/t10k-images-idx3-ubyte
mnist_data/t10k-labels-idx1-ubyte
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| `mnist` | MNIST data loader |
| `ndarray` | N-dimensional arrays |
| `blas-src` | BLAS backend selector |
| `cblas` | BLAS C bindings |
| `rand` / `rand_distr` | RNG + normal distribution for weight init |
| `anyhow` | Error handling |
