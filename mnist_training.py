# %% Imports
import gzip
import os
import struct
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

# %% Download MNIST Data

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_DIR = "mnist_data"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist():
    os.makedirs(MNIST_DIR, exist_ok=True)
    for name, filename in MNIST_FILES.items():
        filepath = os.path.join(MNIST_DIR, filename)
        if not os.path.exists(filepath):
            url = f"{MNIST_URL}/{filename}"
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, filepath)
            print(f"  -> saved to {filepath}")
        else:
            print(f"Already exists: {filepath}")


def parse_idx_images(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def parse_idx_labels(filepath: str) -> np.ndarray:
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)


download_mnist()

# %% Load & Inspect Data

train_images = parse_idx_images(os.path.join(MNIST_DIR, MNIST_FILES["train_images"]))
train_labels = parse_idx_labels(os.path.join(MNIST_DIR, MNIST_FILES["train_labels"]))
test_images = parse_idx_images(os.path.join(MNIST_DIR, MNIST_FILES["test_images"]))
test_labels = parse_idx_labels(os.path.join(MNIST_DIR, MNIST_FILES["test_labels"]))

print(f"Train images: {train_images.shape}, dtype={train_images.dtype}")
print(f"Train labels: {train_labels.shape}, dtype={train_labels.dtype}")
print(f"Test images:  {test_images.shape}, dtype={test_images.dtype}")
print(f"Test labels:  {test_labels.shape}, dtype={test_labels.dtype}")

# %% Preprocess

X_train = train_images.reshape(-1, 784).astype(np.float64) / 255.0
X_test = test_images.reshape(-1, 784).astype(np.float64) / 255.0


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    n = labels.shape[0]
    mat = np.zeros((n, num_classes), dtype=np.float64)
    mat[np.arange(n), labels] = 1.0
    return mat


Y_train = one_hot(train_labels)
Y_test = one_hot(test_labels)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")

# %% Visualize Samples

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(train_images[i], cmap="gray")
    ax.set_title(f"Label: {train_labels[i]}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% Define Model


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(probs: np.ndarray, y: np.ndarray) -> float:
    n = y.shape[0]
    log_probs = -np.log(probs[np.arange(n), np.argmax(y, axis=1)] + 1e-12)
    return float(np.mean(log_probs))


class FullyConnectedNetwork:
    def __init__(self, layer_dims: list[int]):
        self.num_layers = len(layer_dims) - 1
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(self.num_layers):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        zs: list[np.ndarray] = []
        activations: list[np.ndarray] = [x]
        a = x
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            zs.append(z)
            if i < self.num_layers - 1:
                a = relu(z)
            else:
                a = softmax(z)
            activations.append(a)
        return zs, activations

    def backward(
        self, zs: list[np.ndarray], activations: list[np.ndarray], y: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n = y.shape[0]
        dw_list: list[np.ndarray] = [np.empty(0)] * self.num_layers
        db_list: list[np.ndarray] = [np.empty(0)] * self.num_layers

        # Output layer: softmax + cross-entropy analytic gradient
        dz = activations[-1] - y
        dw_list[-1] = activations[-2].T @ dz / n
        db_list[-1] = np.mean(dz, axis=0, keepdims=True)

        for i in range(self.num_layers - 2, -1, -1):
            dz = (dz @ self.weights[i + 1].T) * relu_derivative(zs[i])
            dw_list[i] = activations[i].T @ dz / n
            db_list[i] = np.mean(dz, axis=0, keepdims=True)

        return dw_list, db_list

    def update(
        self,
        dw_list: list[np.ndarray],
        db_list: list[np.ndarray],
        lr: float,
    ):
        for i in range(self.num_layers):
            self.weights[i] -= lr * dw_list[i]
            self.biases[i] -= lr * db_list[i]

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, activations = self.forward(x)
        return np.argmax(activations[-1], axis=1)


model = FullyConnectedNetwork([784, 256, 128, 10])

# %% Train

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.1

train_losses: list[float] = []
train_accuracies: list[float] = []

n_train = X_train.shape[0]

for epoch in range(EPOCHS):
    indices = np.random.permutation(n_train)
    X_shuffled = X_train[indices]
    Y_shuffled = Y_train[indices]

    epoch_loss = 0.0
    epoch_correct = 0
    n_batches = 0

    for start in range(0, n_train, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_train)
        xb = X_shuffled[start:end]
        yb = Y_shuffled[start:end]

        zs, activations = model.forward(xb)
        loss = cross_entropy_loss(activations[-1], yb)
        dw_list, db_list = model.backward(zs, activations, yb)
        model.update(dw_list, db_list, LEARNING_RATE)

        epoch_loss += loss * (end - start)
        epoch_correct += int(np.sum(np.argmax(activations[-1], axis=1) == np.argmax(yb, axis=1)))
        n_batches += 1

    avg_loss = epoch_loss / n_train
    accuracy = epoch_correct / n_train
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    print(f"Epoch {epoch + 1:>2}/{EPOCHS}  loss={avg_loss:.4f}  accuracy={accuracy:.4f}")

# %% Plot Training Curves

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS + 1), train_losses, "o-")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, EPOCHS + 1), train_accuracies, "o-", color="green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training Accuracy")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% Evaluate on Test Set

test_preds = model.predict(X_test)
test_accuracy = float(np.mean(test_preds == test_labels))
print(f"Test accuracy: {test_accuracy:.4f} ({int(test_accuracy * len(test_labels))}/{len(test_labels)})")

# %% Visualize Predictions

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
correct_shown = 0
wrong_shown = 0

for i in range(len(test_preds)):
    if correct_shown >= 5 and wrong_shown >= 5:
        break
    pred = test_preds[i]
    true = test_labels[i]
    if pred == true and correct_shown < 5:
        ax = axes[0, correct_shown]
        ax.imshow(test_images[i], cmap="gray")
        ax.set_title(f"pred={pred} (correct)", color="green", fontsize=9)
        ax.axis("off")
        correct_shown += 1
    elif pred != true and wrong_shown < 5:
        ax = axes[1, wrong_shown]
        ax.imshow(test_images[i], cmap="gray")
        ax.set_title(f"pred={pred}, true={true}", color="red", fontsize=9)
        ax.axis("off")
        wrong_shown += 1

axes[0, 0].annotate(
    "Correct", xy=(0, 0.5), xytext=(-0.3, 0.5),
    xycoords="axes fraction", textcoords="axes fraction",
    fontsize=12, fontweight="bold", color="green",
    ha="center", va="center", rotation=90,
)
axes[1, 0].annotate(
    "Wrong", xy=(0, 0.5), xytext=(-0.3, 0.5),
    xycoords="axes fraction", textcoords="axes fraction",
    fontsize=12, fontweight="bold", color="red",
    ha="center", va="center", rotation=90,
)

plt.tight_layout()
plt.show()

# %%
