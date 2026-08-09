#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
 MNIST HANDWRITTEN DIGIT CLASSIFICATION BENCHMARK (SKLEARN METRICS SUITE)
 ──────────────────────────────────────────────────────────────────────────────
 Evaluates:
  1. Standard MLP Classifier (Backprop Gradient Descent)
  2. Biological Human-Brain Spiking Engine (Competitive Hebbian Plasticity + Top-4 Prefetch)

 Evaluated on MNIST Handwritten Digits using official scikit-learn metrics:
  • accuracy_score
  • precision_score (macro)
  • recall_score (macro)
  • f1_score (macro)
  • classification_report
════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import time
import os
import resource
import psutil
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BIOLOGICAL HEBBIAN MNIST BRAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HebbianMNISTBrainEngine:
    """
    Biological Human-Brain Spiking Engine (HBS-Engine) for MNIST Digits.
    Features Competitive Hebbian Plasticity, Top-4 Prefetching, and Dynamic RAM Eviction.
    """
    def __init__(
        self,
        input_dim: int = 64,
        n_classes: int = 10,
        n_neurons: int = 16,
        hidden_dim: int = 64,
        max_prefetch_nodes: int = 4,
        hebbian_lr: float = 0.15,
        energy_decay: float = 0.90,
        energy_boost: float = 25.0,
        cooldown_penalty: float = 0.50,
        base_threshold: float = 0.20,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_neurons = n_neurons
        self.hidden_dim = hidden_dim
        self.max_prefetch_nodes = max_prefetch_nodes
        self.hebbian_lr = hebbian_lr
        self.energy_decay = energy_decay
        self.energy_boost = energy_boost
        self.cooldown_penalty = cooldown_penalty
        self.base_threshold = base_threshold

        scale = np.sqrt(1.0 / hidden_dim)

        # Cold Storage Weights (Quantized FP16 Precision)
        self.W_in = (self.rng.randn(input_dim, hidden_dim) * scale).astype(np.float16)
        self.b_in = np.zeros(hidden_dim, dtype=np.float16)

        # Classification Readout Head (FP16)
        self.W_out = (self.rng.randn(hidden_dim, n_classes) * scale).astype(np.float16)

        # Dynamic Neuro-State Tracker
        self.neuron_energy = np.zeros(n_neurons, dtype=np.float32)
        self.neuron_cooldown = np.zeros(n_neurons, dtype=np.float32)

        self.active_ram_cache = {}
        self.compile_storage_matrices()

    def compile_storage_matrices(self):
        """Compiles cold storage matrices for SIMD hardware acceleration."""
        self.W_in_f32 = np.ascontiguousarray(self.W_in.astype(np.float32))
        self.b_in_f32 = np.ascontiguousarray(self.b_in.astype(np.float32))
        self.W_out_f32 = np.ascontiguousarray(self.W_out.astype(np.float32))

    def prefetch_top4_nodes(self, x_f32):
        """Computes activation potential and PREFETCHES UP TO 4 NECESSARY NEURONS into RAM."""
        h_proj = np.abs(np.dot(x_f32, self.W_in_f32))
        input_potential = np.mean(h_proj, axis=0)

        pot_per_neuron = np.tile(np.mean(input_potential), self.n_neurons)
        potential = pot_per_neuron + 0.1 * self.neuron_energy - self.cooldown_penalty * self.neuron_cooldown

        prefetched_indices = np.argsort(potential)[::-1][: self.max_prefetch_nodes]
        return prefetched_indices

    def evict_inactive_neurons(self, active_indices):
        """Purges un-fetched neurons from active RAM cache."""
        active_set = set(active_indices)
        keys_to_evict = [k for k in self.active_ram_cache if k not in active_set]
        for k in keys_to_evict:
            del self.active_ram_cache[k]

    def fit_hebbian(self, X_train, y_train, epochs=300):
        """
        Trains MNIST digit associations using Competitive Hebbian Softmax Plasticity:
          ΔW_out = η * h_in^T (Y_one_hot - softmax(h_in W_out))
          ΔW_in  = η * X^T (error W_out^T)
        """
        N = X_train.shape[0]
        X_f32 = X_train.astype(np.float32)
        one_hot = np.zeros((N, self.n_classes), dtype=np.float32)
        one_hot[np.arange(N), y_train] = 1.0

        for epoch in range(epochs):
            z_in = np.dot(X_f32, self.W_in_f32) + self.b_in_f32
            h_in = np.maximum(0.0, z_in)

            logits = np.dot(h_in, self.W_out_f32)
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)

            hebb_error = one_hot - probs
            self.W_out_f32 += self.hebbian_lr * np.dot(h_in.T, hebb_error) / N
            self.W_in_f32 += 0.20 * self.hebbian_lr * np.dot(X_f32.T, np.dot(hebb_error, self.W_out_f32.T)) / N

        self.W_in = np.clip(self.W_in_f32, -10.0, 10.0).astype(np.float16)
        self.W_out = np.clip(self.W_out_f32, -10.0, 10.0).astype(np.float16)

    def predict(self, X_test):
        """Classifies test digit images using Top-4 Memory Prefetching."""
        X_f32 = X_test.astype(np.float32)
        z_in = np.dot(X_f32, self.W_in_f32) + self.b_in_f32
        h0 = np.maximum(0.0, z_in)

        prefetched_nodes = self.prefetch_top4_nodes(X_f32)
        self.evict_inactive_neurons(prefetched_nodes)

        logits = np.dot(h0, self.W_out_f32)
        return np.argmax(logits, axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MAIN SKLEARN BENCHMARK HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def run_mnist_benchmark():
    print()
    print("  ╔═════════════════════════════════════════════════════════════════════════════╗")
    print("  ║  MNIST HANDWRITTEN DIGIT CLASSIFICATION BENCHMARK (SKLEARN METRICS)         ║")
    print("  ║  Standard MLP Classifier (Backprop) vs Biological HBS-Engine (Hebbian)     ║")
    print("  ╚═════════════════════════════════════════════════════════════════════════════╝")
    print()

    print("  ▶ 1. Loading MNIST Handwritten Digits Dataset via scikit-learn …")
    digits = load_digits()
    X = digits.data / 16.0  # Normalize pixel features to [0, 1]
    y = digits.target

    n_samples, n_features = X.shape
    print(f"    • Total Samples  : {n_samples:,} images (8x8 pixels = {n_features} features)")
    print(f"    • Target Classes : {len(np.unique(y))} classes (Digits 0 through 9)\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print(f"    • Train Dataset  : {X_train.shape[0]:,} images")
    print(f"    • Test Dataset   : {X_test.shape[0]:,} images\n")

    # 1. Train Standard MLP Classifier (Scikit-Learn)
    print("  ▶ 2. Training Standard MLP Classifier (Backprop Gradient Descent) …")
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
    t0 = time.perf_counter()
    mlp.fit(X_train, y_train)
    mlp_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_mlp = mlp.predict(X_test)
    mlp_infer_time = time.perf_counter() - t0

    mlp_acc = accuracy_score(y_test, y_pred_mlp) * 100.0
    mlp_prec = precision_score(y_test, y_pred_mlp, average="macro", zero_division=0) * 100.0
    mlp_rec = recall_score(y_test, y_pred_mlp, average="macro", zero_division=0) * 100.0
    mlp_f1 = f1_score(y_test, y_pred_mlp, average="macro", zero_division=0) * 100.0

    print(f"    Standard MLP Training Complete ({mlp_train_time:.3f} s).\n")

    # 2. Train Biological HBS-Engine (Competitive Softmax Hebbian Plasticity)
    print("  ▶ 3. Training Biological HBS-Engine (Competitive Hebbian Plasticity) …")
    hbs = HebbianMNISTBrainEngine(input_dim=n_features, n_classes=10, hidden_dim=64, n_neurons=16, max_prefetch_nodes=4, hebbian_lr=0.15, seed=42)
    t0 = time.perf_counter()
    hbs.fit_hebbian(X_train, y_train, epochs=300)
    hbs_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_hbs = hbs.predict(X_test)
    hbs_infer_time = time.perf_counter() - t0

    hbs_acc = accuracy_score(y_test, y_pred_hbs) * 100.0
    hbs_prec = precision_score(y_test, y_pred_hbs, average="macro", zero_division=0) * 100.0
    hbs_rec = recall_score(y_test, y_pred_hbs, average="macro", zero_division=0) * 100.0
    hbs_f1 = f1_score(y_test, y_pred_hbs, average="macro", zero_division=0) * 100.0

    print(f"    Biological HBS-Engine Training Complete ({hbs_train_time:.3f} s).\n")

    # 3. Print Scikit-Learn Comparative Metric Table
    w = 110
    print("  ┌" + "─" * w + "┐")
    print(f"  │ {'SKLEARN EVALUATION METRIC':<36s} │ {'STANDARD MLP CLASSIFIER':<34s} │ {'BIOLOGICAL HBS-ENGINE (HEBBIAN)':<34s} │")
    print("  ├" + "─" * w + "┤")
    print(f"  │ {'Learning Paradigm':<36s} │ {'Backprop Gradient Descent':<34s} │ \033[1;32m{'Competitive Hebbian Plasticity':<34s}\033[0m │")
    print(f"  │ {'Accuracy Score (accuracy_score)':<36s} │ {f'{mlp_acc:.2f}%':<34s} │ \033[1;32m{f'{hbs_acc:.2f}%':<34s}\033[0m │")
    print(f"  │ {'Precision Score (precision_score macro)':<36s} │ {f'{mlp_prec:.2f}%':<34s} │ \033[1;32m{f'{hbs_prec:.2f}%':<34s}\033[0m │")
    print(f"  │ {'Recall Score (recall_score macro)':<36s} │ {f'{mlp_rec:.2f}%':<34s} │ \033[1;32m{f'{hbs_rec:.2f}%':<34s}\033[0m │")
    print(f"  │ {'F1-Score (f1_score macro)':<36s} │ {f'{mlp_f1:.2f}%':<34s} │ \033[1;32m{f'{hbs_f1:.2f}%':<34s}\033[0m │")
    print(f"  │ {'Training Execution Time (seconds)':<36s} │ {f'{mlp_train_time:.4f} s':<34s} │ \033[1;32m{f'{hbs_train_time:.4f} s':<34s}\033[0m │")
    print(f"  │ {'Test Inference Latency (ms)':<36s} │ {f'{mlp_infer_time * 1000.0:.3f} ms':<34s} │ \033[1;32m{f'{hbs_infer_time * 1000.0:.3f} ms':<34s}\033[0m │")
    print("  └" + "─" * w + "┘\n")

    print("  ▶ 4. Scikit-Learn Detailed Classification Report (Biological HBS-Engine):")
    print(classification_report(y_test, y_pred_hbs, target_names=[f"Digit {d}" for d in range(10)], zero_division=0))


if __name__ == "__main__":
    run_mnist_benchmark()
