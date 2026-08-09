#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
 DENSE-INTERACTION RESISTLESS NETWORK (DIR-Net)
 A Non-DAG, Densely Interacting Hidden Layer Architecture with
 Low-Resistance Output Sinks
════════════════════════════════════════════════════════════════════════════════

 Key Design Principles
 ─────────────────────
 1. Non-DAG Cross-Layer Interactions:
    Hidden layers H_1, H_2, ..., H_K are not restricted to sequential feed-forward
    propagation (H_1 -> H_2 -> H_3). Instead, at each interaction step t, every
    hidden layer exchanges state signals with ALL OTHER hidden layers via dense
    cross-layer interaction weights W_{j -> k}.

 2. Low-Resistance Output Sinks:
    Each hidden layer H_k (and input embedding h_0) connects directly to the final
    output layer via residual low-resistance linear sink projections:
        Logits = W_out0 * h_0 + sum_{k=1}^K (W_outK * h_k) + b_out
    This guarantees zero vanishing-gradient resistance directly to all hidden layers.

 3. Modern Neural Mechanics:
    Uses GELU/ReLU non-linearities, LayerNorm stabilization, residual identity
    shortcuts, and Adam/AdamW optimization.
════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_moons, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PYTORCH IMPLEMENTATION OF DIR-Net
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class DIRNetTorch(nn.Module):
        """
        PyTorch implementation of Dense-Interaction Resistless Network (DIR-Net).
        """
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            n_layers: int = 4,
            n_classes: int = 2,
            n_steps: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.n_classes = n_classes
            self.n_steps = n_steps

            # Input projection h0
            self.input_proj = nn.Linear(input_dim, hidden_dim)

            # Projections for input h0 into each hidden layer k
            self.in_to_layer = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
            ])

            # Self-recurrent weights for each layer
            self.self_weights = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
            ])

            # Cross-layer dense interaction matrix: W[j -> k] for all j != k
            self.cross_weights = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim) if j != k else None
                    for j in range(n_layers)
                ])
                for k in range(n_layers)
            ])

            # Layer Normalization for each hidden layer
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(n_layers)
            ])

            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)

            # Low-Resistance Output Sinks (direct linear readout from h0 and all h_k)
            self.sink_h0 = nn.Linear(hidden_dim, n_classes, bias=False)
            self.sinks = nn.ModuleList([
                nn.Linear(hidden_dim, n_classes, bias=True) for _ in range(n_layers)
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Step 1: Input projection h0
            h0 = self.act(self.input_proj(x))  # (B, hidden_dim)

            # Step 2: Initialize hidden layer states [h_1, h_2, ..., h_K]
            layer_states = [self.act(self.in_to_layer[k](h0)) for k in range(self.n_layers)]

            # Step 3: Recurrent Dense Inter-Layer Interaction Ticks
            scale = 1.0 / np.sqrt(self.n_layers)
            for step in range(self.n_steps):
                next_states = []
                for k in range(self.n_layers):
                    net_k = self.in_to_layer[k](h0) + self.self_weights[k](layer_states[k])
                    for j in range(self.n_layers):
                        if j != k:
                            net_k = net_k + scale * self.cross_weights[k][j](layer_states[j])

                    h_next = self.layer_norms[k](layer_states[k] + self.drop(self.act(net_k)))
                    next_states.append(h_next)

                layer_states = next_states

            # Step 4: Low-Resistance Direct Output Sinks Readout
            logits = scale * self.sink_h0(h0)
            for k in range(self.n_layers):
                logits = logits + scale * self.sinks[k](layer_states[k])

            return logits


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NUMPY IMPLEMENTATION OF DIR-Net WITH STABLE NORM & GRADIENTS
# ═══════════════════════════════════════════════════════════════════════════════

class DIRNetNumPy:
    """
    Stabilized NumPy implementation of DIR-Net with Layer Normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        n_layers: int = 3,
        n_classes: int = 2,
        n_steps: int = 2,
        lr: float = 0.01,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_steps = n_steps
        self.lr = lr

        scale = np.sqrt(1.0 / hidden_dim)
        self.W_in = self.rng.randn(input_dim, hidden_dim) * np.sqrt(1.0 / input_dim)
        self.b_in = np.zeros(hidden_dim)

        self.W_in_k = [self.rng.randn(hidden_dim, hidden_dim) * scale for _ in range(n_layers)]
        self.b_in_k = [np.zeros(hidden_dim) for _ in range(n_layers)]
        self.W_self = [self.rng.randn(hidden_dim, hidden_dim) * scale for _ in range(n_layers)]

        self.W_cross = [
            [self.rng.randn(hidden_dim, hidden_dim) * scale if j != k else None for j in range(n_layers)]
            for k in range(n_layers)
        ]

        self.W_out0 = self.rng.randn(hidden_dim, n_classes) * scale
        self.W_out_k = [self.rng.randn(hidden_dim, n_classes) * scale for _ in range(n_layers)]
        self.b_out = np.zeros(n_classes)

    def _gelu(self, z):
        return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * np.power(z, 3))))

    def _layernorm(self, x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def _softmax(self, logits):
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, X):
        h0 = self._gelu(np.dot(X, self.W_in) + self.b_in)
        h0 = self._layernorm(h0)

        h_k = [self._layernorm(self._gelu(np.dot(h0, self.W_in_k[k]) + self.b_in_k[k])) for k in range(self.n_layers)]

        inter_scale = 1.0 / np.sqrt(self.n_layers)
        for step in range(self.n_steps):
            next_h_k = []
            for k in range(self.n_layers):
                net_k = np.dot(h0, self.W_in_k[k]) + np.dot(h_k[k], self.W_self[k])
                for j in range(self.n_layers):
                    if j != k:
                        net_k += inter_scale * np.dot(h_k[j], self.W_cross[k][j])
                h_new = self._layernorm(h_k[k] + self._gelu(net_k))
                next_h_k.append(h_new)
            h_k = next_h_k

        out_scale = 1.0 / np.sqrt(self.n_layers + 1)
        logits = out_scale * np.dot(h0, self.W_out0) + self.b_out
        for k in range(self.n_layers):
            logits += out_scale * np.dot(h_k[k], self.W_out_k[k])

        probs = self._softmax(logits)
        return probs, h0, h_k

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 80, batch_size: int = 32):
        N = len(X)
        for epoch in range(epochs):
            indices = self.rng.permutation(N)
            total_loss = 0.0

            for b in range(0, N, batch_size):
                b_idx = indices[b:b+batch_size]
                xb = X[b_idx]
                yb = y[b_idx]

                probs, h0, h_k = self.forward(xb)

                one_hot = np.zeros_like(probs)
                one_hot[np.arange(len(yb)), yb] = 1.0
                loss = -np.mean(np.log(probs[np.arange(len(yb)), yb] + 1e-8))
                total_loss += loss * len(yb)

                dlogits = (probs - one_hot) / len(yb)

                # Direct backprop via Low-Resistance Output Sinks
                out_scale = 1.0 / np.sqrt(self.n_layers + 1)
                dW_out0 = np.dot(h0.T, dlogits) * out_scale
                self.W_out0 -= self.lr * np.clip(dW_out0, -1.0, 1.0)
                self.b_out -= self.lr * np.sum(dlogits, axis=0)

                for k in range(self.n_layers):
                    dW_outk = np.dot(h_k[k].T, dlogits) * out_scale
                    self.W_out_k[k] -= self.lr * np.clip(dW_outk, -1.0, 1.0)

                    dh_k = np.dot(dlogits, self.W_out_k[k].T) * out_scale
                    dW_ink = np.dot(h0.T, dh_k)
                    dW_selfk = np.dot(h_k[k].T, dh_k)

                    self.W_in_k[k] -= self.lr * np.clip(dW_ink, -1.0, 1.0)
                    self.W_self[k] -= self.lr * np.clip(dW_selfk, -1.0, 1.0)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/N:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs, _, _ = self.forward(X)
        return np.argmax(probs, axis=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING & EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def train_torch_model(model, train_loader, test_loader, epochs=40, lr=0.003):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"  ▶ Training DIR-Net (PyTorch) for {epochs} epochs …")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        if (epoch + 1) % 10 == 0:
            acc = eval_torch_model(model, train_loader)
            print(f"    Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader.dataset):.4f} | Train Acc: {acc*100:.2f}%")

    return model


def eval_torch_model(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    return accuracy_score(all_targets, all_preds)


def print_metrics(name: str, y_true, y_pred, infer_ms: float) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    w = 54
    print(f"\n  ┌{'─' * w}┐")
    print(f"  │ {name:^{w}s} │")
    print(f"  ├{'─' * w}┤")
    print(f"  │  Accuracy:   {acc:.4f}  ({acc * 100:6.2f}%){' ' * 19}│")
    print(f"  │  Precision:  {prec:.4f}{' ' * 32}│")
    print(f"  │  Recall:     {rec:.4f}{' ' * 32}│")
    print(f"  │  F1-Score:   {f1:.4f}{' ' * 32}│")
    print(f"  │  Infer Time: {infer_ms:>8.2f} ms{' ' * 26}│")
    print(f"  └{'─' * w}┘")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "time_ms": infer_ms}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiments():
    print()
    print("  ╔═══════════════════════════════════════════════════════════╗")
    print("  ║  DENSE-INTERACTION RESISTLESS NETWORK (DIR-Net)           ║")
    print("  ║  Non-DAG Inter-Layer Recurrence + Low-Resistance Sinks    ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print()

    # ── 1. MAKE MOONS EXPERIMENT ──────────────────────────────────────
    print("  ▶ 1. EVALUATING ON MAKE-MOONS BENCHMARK …")
    X_m, y_m = make_moons(n_samples=1200, noise=0.20, random_state=42)
    X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_m, y_m, test_size=0.20, random_state=42, stratify=y_m)

    scaler_m = StandardScaler()
    X_tr_m_sc = scaler_m.fit_transform(X_tr_m)
    X_te_m_sc = scaler_m.transform(X_te_m)

    if HAS_TORCH:
        tr_ds = TensorDataset(torch.tensor(X_tr_m_sc, dtype=torch.float32), torch.tensor(y_tr_m, dtype=torch.long))
        te_ds = TensorDataset(torch.tensor(X_te_m_sc, dtype=torch.float32), torch.tensor(y_te_m, dtype=torch.long))
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

        dir_net = DIRNetTorch(input_dim=2, hidden_dim=64, n_layers=4, n_classes=2, n_steps=3)
        t0 = time.perf_counter()
        train_torch_model(dir_net, tr_loader, te_loader, epochs=50, lr=0.005)
        train_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dir_net.eval()
        with torch.no_grad():
            y_pred_dir = torch.argmax(dir_net(torch.tensor(X_te_m_sc, dtype=torch.float32)), dim=-1).numpy()
        infer_ms = (time.perf_counter() - t0) * 1000

        m_dir = print_metrics("DIR-Net (PyTorch)", y_te_m, y_pred_dir, infer_ms)
    else:
        dir_net = DIRNetNumPy(input_dim=2, hidden_dim=32, n_layers=3, n_classes=2, n_steps=2, lr=0.01)
        t0 = time.perf_counter()
        dir_net.fit(X_tr_m_sc, y_tr_m, epochs=80, batch_size=32)
        train_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        y_pred_dir = dir_net.predict(X_te_m_sc)
        infer_ms = (time.perf_counter() - t0) * 1000
        m_dir = print_metrics("DIR-Net (NumPy)", y_te_m, y_pred_dir, infer_ms)

    # Baselines
    lr_m = LogisticRegression(random_state=42).fit(X_tr_m_sc, y_tr_m)
    t0 = time.perf_counter()
    y_pred_lr = lr_m.predict(X_te_m_sc)
    m_lr = print_metrics("Logistic Regression", y_te_m, y_pred_lr, (time.perf_counter() - t0) * 1000)

    dt_m = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_tr_m_sc, y_tr_m)
    t0 = time.perf_counter()
    y_pred_dt = dt_m.predict(X_te_m_sc)
    m_dt = print_metrics("Decision Tree (max_depth=5)", y_te_m, y_pred_dt, (time.perf_counter() - t0) * 1000)

    # ── 2. TITANIC EXPERIMENT ─────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  ▶ 2. EVALUATING ON TITANIC BENCHMARK …")
    print("═" * 80 + "\n")

    X_df, y_raw = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    X_df['sex'] = (X_df['sex'] == 'female').astype(float)
    X_df['has_cabin'] = X_df['cabin'].notna().astype(float)
    X_df['embarked_C'] = (X_df['embarked'] == 'C').astype(float)
    X_df['embarked_Q'] = (X_df['embarked'] == 'Q').astype(float)
    X_df['embarked_S'] = (X_df['embarked'] == 'S').astype(float)

    features = ['sex', 'pclass', 'age', 'fare', 'sibsp', 'parch', 'has_cabin', 'embarked_C', 'embarked_Q', 'embarked_S']
    X = X_df[features].values
    y = y_raw.astype(int).values

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)

    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_sc, y, test_size=0.20, random_state=42, stratify=y)

    if HAS_TORCH:
        tr_ds_t = TensorDataset(torch.tensor(X_tr_t, dtype=torch.float32), torch.tensor(y_tr_t, dtype=torch.long))
        te_ds_t = TensorDataset(torch.tensor(X_te_t, dtype=torch.float32), torch.tensor(y_te_t, dtype=torch.long))
        tr_loader_t = DataLoader(tr_ds_t, batch_size=32, shuffle=True)
        te_loader_t = DataLoader(te_ds_t, batch_size=128, shuffle=False)

        dir_net_t = DIRNetTorch(input_dim=len(features), hidden_dim=64, n_layers=4, n_classes=2, n_steps=3, dropout=0.15)
        t0 = time.perf_counter()
        train_torch_model(dir_net_t, tr_loader_t, te_loader_t, epochs=60, lr=0.003)
        train_ms_t = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dir_net_t.eval()
        with torch.no_grad():
            y_pred_dir_t = torch.argmax(dir_net_t(torch.tensor(X_te_t, dtype=torch.float32)), dim=-1).numpy()
        infer_ms_t = (time.perf_counter() - t0) * 1000

        m_dir_t = print_metrics("DIR-Net (Titanic)", y_te_t, y_pred_dir_t, infer_ms_t)
    else:
        dir_net_t = DIRNetNumPy(input_dim=len(features), hidden_dim=32, n_layers=3, n_classes=2, n_steps=2, lr=0.01)
        t0 = time.perf_counter()
        dir_net_t.fit(X_tr_t, y_tr_t, epochs=80, batch_size=32)
        train_ms_t = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        y_pred_dir_t = dir_net_t.predict(X_te_t)
        infer_ms_t = (time.perf_counter() - t0) * 1000
        m_dir_t = print_metrics("DIR-Net (Titanic)", y_te_t, y_pred_dir_t, infer_ms_t)

    # Baselines
    lr_t = LogisticRegression(random_state=42, max_iter=1000).fit(X_tr_t, y_tr_t)
    t0 = time.perf_counter()
    y_pred_lr_t = lr_t.predict(X_te_t)
    m_lr_t = print_metrics("LR (Titanic)", y_te_t, y_pred_lr_t, (time.perf_counter() - t0) * 1000)

    dt_t = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_tr_t, y_tr_t)
    t0 = time.perf_counter()
    y_pred_dt_t = dt_t.predict(X_te_t)
    m_dt_t = print_metrics("DT (Titanic)", y_te_t, y_pred_dt_t, (time.perf_counter() - t0) * 1000)

    print()
    print("  ╔═════════════════════════════════════════════════════════════╗")
    print("  ║               FINAL COMPARISON SUMMARY                      ║")
    print("  ╠═════════════════════════════════════════════════════════════╣")
    print(f"  ║  {'Model':<28s}  {'Acc':>7s}  {'Prec':>6s}  {'F1-Score':>8s}  ║")
    print(f"  ╟─{'─' * 28}──{'─' * 7}──{'─' * 6}──{'─' * 8}──╢")

    for label, m in [
        ("DIR-Net (New Architecture)", m_dir_t),
        ("Logistic Regression", m_lr_t),
        ("Decision Tree (d=5)", m_dt_t),
    ]:
        print(f"  ║  {label:<28s}  {m['accuracy'] * 100:>6.2f}%  {m['precision']:>.4f}  {m['f1']:>.4f}    ║")

    print("  ╚═════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    run_experiments()
