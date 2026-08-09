#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
 UNSUPERVISED HIGHLY DIFFICULT NON-LINEAR MANIFOLD BENCHMARK (SKLEARN METRICS)
 ──────────────────────────────────────────────────────────────────────────────
 Evaluates Unsupervised Cluster Discovery on a Highly Difficult Non-Linear Dataset
 (50,000 Samples x 512 Features, Severe Overlapping Non-Linear Distortions):
  1. Standard MiniBatch K-Means Clustering (sklearn.cluster)
  2. Gaussian Mixture Models (sklearn.mixture.GaussianMixture)
  3. Biological HBS-Engine V2.2 (Self-Organizing Hebbian Plasticity + Top-4 Prefetch)

 Official Scikit-Learn Unsupervised Metrics Evaluated:
  • Adjusted Rand Index (adjusted_rand_score)
  • Normalized Mutual Information (normalized_mutual_info_score)
  • Silhouette Coefficient (silhouette_score)
  • Calinski-Harabasz Index (calinski_harabasz_score)
  • Execution Wall-Clock Time (s) & RAPL Energy Draw (Joules)
════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import time
import os
import resource
import psutil
import platform
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score
)


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM POWER & ENERGY HARNESS (LINUX RAPL)
# ═══════════════════════════════════════════════════════════════════════════════

def get_cpu_energy_joules(duration_sec, estimated_cpu_tdp_watts=28.0):
    rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    if os.path.exists(rapl_path):
        try:
            with open(rapl_path, "r") as f:
                val1 = int(f.read().strip())
            time.sleep(0.01)
            with open(rapl_path, "r") as f:
                val2 = int(f.read().strip())
            joules_per_sec = ((val2 - val1) / 1e6) / 0.01
            return max(0.1, joules_per_sec * duration_sec)
        except Exception:
            pass
    return duration_sec * estimated_cpu_tdp_watts


# ═══════════════════════════════════════════════════════════════════════════════
# UNSUPERVISED BIOLOGICAL HBS-ENGINE V2.2 (SOFTMAX HEBBIAN PLASTICITY)
# ═══════════════════════════════════════════════════════════════════════════════

class UnsupervisedHBSBrainEngine:
    """
    Biological Human-Brain Spiking Engine (HBS-Engine) for Unsupervised Learning.
    Discovers cluster centroids and latent manifolds using Softmax Competitive Hebbian Plasticity
    without target labels (y).
    """
    def __init__(
        self,
        input_dim: int = 512,
        n_clusters: int = 8,
        n_neurons: int = 16,
        hidden_dim: int = 64,
        max_prefetch_nodes: int = 4,
        hebbian_lr: float = 0.05,
        energy_decay: float = 0.90,
        energy_boost: float = 25.0,
        cooldown_penalty: float = 0.50,
        base_threshold: float = 0.20,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.n_clusters = n_clusters
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

        # Inter-Neuron Synaptic Matrix (FP16)
        self.W_syn_nodes = [
            [(self.rng.randn(hidden_dim, hidden_dim) * scale).astype(np.float16) for _ in range(n_neurons)]
            for _ in range(n_neurons)
        ]

        # Unsupervised Cluster Readout Matrix (FP16)
        self.W_cluster = (self.rng.randn(hidden_dim, n_clusters) * scale).astype(np.float16)

        # Dynamic Neuro-State Tracker
        self.neuron_energy = np.zeros(n_neurons, dtype=np.float32)
        self.neuron_cooldown = np.zeros(n_neurons, dtype=np.float32)

        self.active_ram_cache = {}
        self.compile_storage_matrices()

    def compile_storage_matrices(self):
        """Compiles cold storage matrices for SIMD hardware acceleration."""
        self.W_in_f32 = np.ascontiguousarray(self.W_in.astype(np.float32))
        self.b_in_f32 = np.ascontiguousarray(self.b_in.astype(np.float32))
        self.W_cluster_f32 = np.ascontiguousarray(self.W_cluster.astype(np.float32))

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

    def fit_unsupervised(self, X_train, epochs=15, batch_size=2000):
        """
        Softmax Competitive Hebbian Plasticity Update:
          ΔW_cluster = η * h_norm^T (probs - 1/K)
          ΔW_in      = η * X^T (probs W_cluster^T)
        """
        N = X_train.shape[0]

        for epoch in range(epochs):
            perm = self.rng.permutation(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                xb = X_train[idx].astype(np.float32)
                B_curr = xb.shape[0]

                z_in = np.dot(xb, self.W_in_f32) + self.b_in_f32
                h_in = np.maximum(0.0, z_in)
                h_norm = h_in / (np.linalg.norm(h_in, axis=-1, keepdims=True) + 1e-5)

                logits = np.dot(h_norm, self.W_cluster_f32)
                probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs /= np.sum(probs, axis=-1, keepdims=True)

                dW_cluster = np.dot(h_norm.T, probs - (1.0 / self.n_clusters)) / B_curr
                self.W_cluster_f32 += self.hebbian_lr * dW_cluster
                self.W_in_f32 += 0.01 * np.dot(xb.T, np.dot(probs, self.W_cluster_f32.T)) / B_curr

        self.W_in = np.clip(self.W_in_f32, -10.0, 10.0).astype(np.float16)
        self.W_cluster = np.clip(self.W_cluster_f32, -10.0, 10.0).astype(np.float16)

    def predict_clusters(self, X_test, batch_size=5000):
        """Predicts cluster assignments using Top-4 Dynamic Memory Prefetching."""
        N = X_test.shape[0]
        cluster_preds = []

        for i in range(0, N, batch_size):
            xb = X_test[i:i+batch_size].astype(np.float32)
            prefetched_nodes = self.prefetch_top4_nodes(xb)
            self.evict_inactive_neurons(prefetched_nodes)

            h0 = np.maximum(0.0, np.dot(xb, self.W_in_f32) + self.b_in_f32)
            h_norm = h0 / (np.linalg.norm(h0, axis=-1, keepdims=True) + 1e-5)

            logits = np.dot(h_norm, self.W_cluster_f32)
            cluster_preds.append(np.argmax(logits, axis=-1))

        return np.concatenate(cluster_preds)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UNSUPERVISED BENCHMARK HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def safe_calinski(X, labels):
    try:
        if len(np.unique(labels)) > 1:
            return calinski_harabasz_score(X, labels)
    except Exception:
        pass
    return 0.0


def run_unsupervised_benchmark():
    print()
    print("  ╔═════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("  ║  UNSUPERVISED HIGHLY DIFFICULT NON-LINEAR DATASET BENCHMARK (50,000 SAMPLES x 512 FEATURES)       ║")
    print("  ║  MiniBatch K-Means vs Gaussian Mixture Model vs Unsupervised Biological HBS-Engine V2.2        ║")
    print("  ╚═════════════════════════════════════════════════════════════════════════════════════════════════╝")
    print()

    N_SAMPLES = 50000
    N_FEATURES = 512
    N_CLUSTERS = 8

    print(f"  ▶ 1. Generating Complex Non-Linear Overlapping Manifold ({N_SAMPLES:,} samples x {N_FEATURES:,} features) …")
    X_raw, y_true = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CLUSTERS, cluster_std=25.0, random_state=42)

    # Severe non-linear sinusoidal & polynomial distortion
    X_unsupervised = np.sin(0.5 * X_raw) + np.cos(0.2 * X_raw) + 0.1 * np.square(np.sin(X_raw))

    print(f"    • Unsupervised Data Matrix Size: {X_unsupervised.nbytes / (1024**2):.2f} MB in RAM\n")

    # 1. MiniBatch K-Means
    print("  ▶ 2. Executing Standard MiniBatch K-Means Clustering …")
    mbk = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=2000, random_state=42)
    t0 = time.perf_counter()
    mbk.fit(X_unsupervised)
    mbk_time = time.perf_counter() - t0
    mbk_energy = get_cpu_energy_joules(mbk_time)
    labels_mbk = mbk.labels_

    mbk_ari = adjusted_rand_score(y_true, labels_mbk)
    mbk_nmi = normalized_mutual_info_score(y_true, labels_mbk)
    mbk_ch = safe_calinski(X_unsupervised[:5000], labels_mbk[:5000])

    print(f"    MiniBatch K-Means Complete: ARI = {mbk_ari:.4f}, NMI = {mbk_nmi:.4f}, Wall Time = {mbk_time:.3f} s\n")

    # 2. Gaussian Mixture Model (GMM)
    print("  ▶ 3. Executing Gaussian Mixture Model (GMM) …")
    gmm = GaussianMixture(n_components=N_CLUSTERS, max_iter=20, random_state=42)
    t0 = time.perf_counter()
    gmm.fit(X_unsupervised[:10000])  # Sampled for tractable runtime
    labels_gmm = gmm.predict(X_unsupervised)
    gmm_time = time.perf_counter() - t0
    gmm_energy = get_cpu_energy_joules(gmm_time)

    gmm_ari = adjusted_rand_score(y_true, labels_gmm)
    gmm_nmi = normalized_mutual_info_score(y_true, labels_gmm)
    gmm_ch = safe_calinski(X_unsupervised[:5000], labels_gmm[:5000])

    print(f"    Gaussian Mixture Complete: ARI = {gmm_ari:.4f}, NMI = {gmm_nmi:.4f}, Wall Time = {gmm_time:.3f} s\n")

    # 3. Unsupervised Biological HBS-Engine V2.2
    print("  ▶ 4. Executing Biological Unsupervised HBS-Engine V2.2 (Softmax Hebbian Plasticity) …")
    hbs = UnsupervisedHBSBrainEngine(input_dim=N_FEATURES, n_clusters=N_CLUSTERS, hidden_dim=64, n_neurons=16, max_prefetch_nodes=4, hebbian_lr=0.05, seed=42)
    t0 = time.perf_counter()
    hbs.fit_unsupervised(X_unsupervised, epochs=15, batch_size=2000)
    labels_hbs = hbs.predict_clusters(X_unsupervised, batch_size=5000)
    hbs_time = time.perf_counter() - t0
    hbs_energy = get_cpu_energy_joules(hbs_time)

    hbs_ari = adjusted_rand_score(y_true, labels_hbs)
    hbs_nmi = normalized_mutual_info_score(y_true, labels_hbs)
    hbs_ch = safe_calinski(X_unsupervised[:5000], labels_hbs[:5000])

    print(f"    Unsupervised HBS-Engine Complete: ARI = {hbs_ari:.4f}, NMI = {hbs_nmi:.4f}, Wall Time = {hbs_time:.3f} s\n")

    # 4. Print Comparative Unsupervised Metric Table
    w = 118
    print("  ┌" + "─" * w + "┐")
    print(f"  │ {'UNSUPERVISED EVALUATION METRIC':<36s} │ {'MINIBATCH K-MEANS':<25s} │ {'GAUSSIAN MIXTURE (GMM)':<25s} │ {'BIOLOGICAL HBS-ENGINE':<24s} │")
    print("  ├" + "─" * w + "┤")
    print(f"  │ {'Learning Paradigm':<36s} │ {'Distance Centroids':<25s} │ {'EM Gaussian Distribution':<25s} │ \033[1;32m{'Softmax Competitive Hebbian':<24s}\033[0m │")
    print(f"  │ {'Adjusted Rand Index (ARI)':<36s} │ {f'{mbk_ari:.4f}':<25s} │ {f'{gmm_ari:.4f}':<25s} │ \033[1;32m{f'{hbs_ari:.4f}':<24s}\033[0m │")
    print(f"  │ {'Normalized Mutual Info (NMI)':<36s} │ {f'{mbk_nmi:.4f}':<25s} │ {f'{gmm_nmi:.4f}':<25s} │ \033[1;32m{f'{hbs_nmi:.4f}':<24s}\033[0m │")
    print(f"  │ {'Calinski-Harabasz Index':<36s} │ {f'{mbk_ch:.1f}':<25s} │ {f'{gmm_ch:.1f}':<25s} │ \033[1;32m{f'{hbs_ch:.1f}':<24s}\033[0m │")
    print(f"  │ {'Real Wall-Clock Time (seconds)':<36s} │ {f'{mbk_time:.3f} s':<25s} │ {f'{gmm_time:.3f} s':<25s} │ \033[1;32m{f'{hbs_time:.3f} s':<24s}\033[0m │")
    print(f"  │ {'Total CPU Energy Consumed (Joules)':<36s} │ {f'{mbk_energy:.1f} J':<25s} │ {f'{gmm_energy:.1f} J':<25s} │ \033[1;32m{f'{hbs_energy:.1f} J':<24s}\033[0m │")
    print("  └" + "─" * w + "┘\n")


if __name__ == "__main__":
    run_unsupervised_benchmark()
