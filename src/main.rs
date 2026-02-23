pub mod model;

use mnist::{Mnist, MnistBuilder};
use model::{cross_entropy_loss, FNN};
use ndarray::Array2;
use rand::seq::SliceRandom;
use std::time::Instant;

const BATCH_SIZE: usize = 64;
const LOG_EVERY_BATCHES: usize = 67;

const EPOCHS: usize = 20;
const LEARNING_RATE: f64 = 0.1;
const N_TRAIN: usize = 60_000;
const N_TEST: usize = 10_000;
const IMG_SIZE: usize = 28 * 28;

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn main() {
    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, .. } = MnistBuilder::new()
        .base_path("mnist_data/")
        .test_images_filename("t10k-images-idx3-ubyte")
        .label_format_digit()
        .training_set_length(N_TRAIN as u32)
        .test_set_length(N_TEST as u32)
        .finalize();

    println!("data loaded");
    let mut fnn = FNN::new(&[784, 256, 128, 10]).expect("failed to create FNN");
    let mut indices: Vec<usize> = (0..N_TRAIN).collect();
    let mut rng = rand::rng();

    let n_batches = N_TRAIN.div_ceil(BATCH_SIZE);

    for epoch in 0..EPOCHS {
        indices.shuffle(&mut rng);
        let mut total_loss = 0.0;
        let mut correct = 0usize;
        let mut window_loss = 0.0;
        let mut window_correct = 0usize;
        let mut window_samples = 0usize;
        let mut t = Instant::now();

        for (batch_idx, batch_start) in (0..N_TRAIN).step_by(BATCH_SIZE).enumerate() {
            let batch_end = (batch_start + BATCH_SIZE).min(N_TRAIN);
            let batch_len = batch_end - batch_start;
            let batch_indices = &indices[batch_start..batch_end];

            let mut input_data = Vec::with_capacity(batch_len * IMG_SIZE);
            let mut label_data = Vec::with_capacity(batch_len * 10);
            for &i in batch_indices {
                input_data.extend(trn_img[i * IMG_SIZE..(i + 1) * IMG_SIZE].iter().map(|&p| p as f64 / 255.0));
                let lbl = trn_lbl[i] as usize;
                for c in 0..10usize { label_data.push(if c == lbl { 1.0 } else { 0.0 }); }
            }
            let input_matrix = Array2::from_shape_vec((batch_len, IMG_SIZE), input_data).unwrap();
            let label_matrix = Array2::from_shape_vec((batch_len, 10), label_data).unwrap();

            let caches = fnn.forward_batch(&input_matrix);
            let probs_matrix = &caches.last().unwrap().a;

            for (r, &orig_idx) in batch_indices.iter().enumerate() {
                let probs = probs_matrix.row(r);
                let label_row = label_matrix.row(r);
                let loss = cross_entropy_loss(probs.as_slice().unwrap(), label_row.as_slice().unwrap());
                let hit = argmax(probs.as_slice().unwrap()) == trn_lbl[orig_idx] as usize;
                total_loss += loss;
                window_loss += loss;
                if hit { correct += 1; window_correct += 1; }
                window_samples += 1;
            }

            let grads = fnn.backward_batch(&input_matrix, &caches, &label_matrix);
            fnn.update(&grads, LEARNING_RATE);

            if (batch_idx + 1) % LOG_EVERY_BATCHES == 0 {
                let elapsed = t.elapsed().as_secs_f64();
                let avg = window_loss / window_samples as f64;
                let acc = window_correct as f64 / window_samples as f64 * 100.0;
                let ms_per_batch = elapsed / LOG_EVERY_BATCHES.min(batch_idx + 1) as f64 * 1000.0;
                println!(
                    "epoch {epoch:2} [{:4}/{n_batches}] | loss: {avg:.4} | acc: {acc:.1}% | {ms_per_batch:.3}ms/batch",
                    batch_idx + 1,
                );
                window_loss = 0.0;
                window_correct = 0;
                window_samples = 0;
                t = Instant::now();
            }
        }

        let avg_loss = total_loss / N_TRAIN as f64;
        let accuracy = correct as f64 / N_TRAIN as f64 * 100.0;
        println!("epoch {epoch:2} done  | loss: {avg_loss:.4} | train acc: {accuracy:.2}%\n");
    }

    let mut correct = 0usize;
    for test_start in (0..N_TEST).step_by(BATCH_SIZE) {
        let test_end = (test_start + BATCH_SIZE).min(N_TEST);
        let test_len = test_end - test_start;
        let mut input_data = Vec::with_capacity(test_len * IMG_SIZE);
        for i in test_start..test_end {
            input_data.extend(tst_img[i * IMG_SIZE..(i + 1) * IMG_SIZE].iter().map(|&p| p as f64 / 255.0));
        }
        let input_matrix = Array2::from_shape_vec((test_len, IMG_SIZE), input_data).unwrap();
        let caches = fnn.forward_batch(&input_matrix);
        let probs_matrix = &caches.last().unwrap().a;
        for (r, i) in (test_start..test_end).enumerate() {
            if argmax(probs_matrix.row(r).as_slice().unwrap()) == tst_lbl[i] as usize {
                correct += 1;
            }
        }
    }

    let test_acc = correct as f64 / N_TEST as f64 * 100.0;
    println!("\ntest acc: {test_acc:.2}%  ({correct}/{N_TEST})");
}
