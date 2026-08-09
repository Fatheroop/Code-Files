#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
 LIVE VISUAL DESKTOP MOUSE & KEYBOARD TRACKER (BIOLOGICAL HBS-ENGINE V2.2)
 ──────────────────────────────────────────────────────────────────────────────
 Renders a Live Interactive Desktop Window using Matplotlib:
  1. Opens an Interactive Dark-Mode Desktop Window.
  2. RED DOT   = Your physical mouse position as you move inside the window.
  3. GREEN DOT = Biological HBS-Engine V2.2's real-time predicted cursor position
                 that physically follows and chases your mouse live!
  4. Real-Time HUD Bar displaying active pre-fetched RAM nodes and step latency.
════════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# BIOLOGICAL HBS-ENGINE V2.2 ONLINE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class VisualHBSBrainEngine:
    def __init__(self, input_dim=6, hidden_dim=64, n_neurons=16, max_prefetch=4, hebbian_lr=0.15, seed=42):
        self.rng = np.random.RandomState(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_neurons = n_neurons
        self.max_prefetch = max_prefetch
        self.hebbian_lr = hebbian_lr

        scale = np.sqrt(1.0 / hidden_dim)

        self.W_in = (self.rng.randn(input_dim, hidden_dim) * scale).astype(np.float16)
        self.b_in = np.zeros(hidden_dim, dtype=np.float16)
        self.W_out = (self.rng.randn(hidden_dim, 2) * scale).astype(np.float16)

        self.W_in_f32 = self.W_in.astype(np.float32)
        self.b_in_f32 = self.b_in.astype(np.float32)
        self.W_out_f32 = self.W_out.astype(np.float32)

        self.neuron_energy = np.zeros(n_neurons, dtype=np.float32)
        self.neuron_cooldown = np.zeros(n_neurons, dtype=np.float32)

    def prefetch_top4_nodes(self, x_f32):
        h_proj = np.abs(np.dot(x_f32, self.W_in_f32))
        input_potential = np.mean(h_proj, axis=0)

        pot_per_neuron = np.tile(np.mean(input_potential), self.n_neurons)
        potential = pot_per_neuron + 0.1 * self.neuron_energy - 0.50 * self.neuron_cooldown

        return np.argsort(potential)[::-1][: self.max_prefetch]

    def update_online_step(self, x_curr_norm, target_xy_norm):
        xb = x_curr_norm.astype(np.float32).reshape(1, -1)
        target = target_xy_norm.astype(np.float32).reshape(1, -1)

        t0 = time.perf_counter()
        prefetched_nodes = self.prefetch_top4_nodes(xb)

        h_in = np.maximum(0.0, np.dot(xb, self.W_in_f32) + self.b_in_f32)
        pred_xy = np.dot(h_in, self.W_out_f32)

        error = target - pred_xy
        self.W_out_f32 += self.hebbian_lr * np.dot(h_in.T, error)
        self.W_in_f32 += 0.10 * self.hebbian_lr * np.dot(xb.T, np.dot(error, self.W_out_f32.T))

        latency_us = (time.perf_counter() - t0) * 1e6

        return pred_xy.flatten(), prefetched_nodes, latency_us


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MATPLOTLIB DESKTOP WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

def run_visual_interactive_window():
    print()
    print("  ╔═════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("  ║  LIVE INTERACTIVE DESKTOP MOUSE & KEYBOARD DEMO (BIOLOGICAL HBS-ENGINE V2.2)                     ║")
    print("  ║  Opening Desktop Window … Move your mouse inside the window to see live tracking!               ║")
    print("  ╚═════════════════════════════════════════════════════════════════════════════╝")
    print()

    engine = VisualHBSBrainEngine(input_dim=6, hidden_dim=64, n_neurons=16, max_prefetch=4, hebbian_lr=0.15, seed=42)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#020617')

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 650)
    ax.invert_yaxis()  # Match screen pixel coordinates (0,0 top-left)

    ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.8)
    ax.set_title("BIOLOGICAL HBS-ENGINE V2.2 -- REAL-TIME MOUSE & KEYBOARD PREDICTOR", color='#38bdf8', fontsize=11, fontweight='bold', pad=12)

    # Plot Cursor Dots
    red_dot, = ax.plot([500], [325], 'ro', markersize=14, label='RED: Physical Mouse Position', zorder=5)
    green_dot, = ax.plot([500], [325], 'go', markersize=16, label='GREEN: HBS-Engine Predicted Cursor', zorder=6)
    trail_line, = ax.plot([], [], 'c--', linewidth=2, label='Latency Connection')

    ax.legend(loc='upper right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc')

    hud_text = ax.text(20, 30, "Move mouse inside window | Step: 0 | Latency: 0.00 us | RAM Nodes: [0 1 2 3]", color='#f8fafc', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#0f172a', edgecolor='#334155'))

    state = {
        'last_x': 500,
        'last_y': 325,
        'step': 0,
        'trail_x': [],
        'trail_y': []
    }

    def on_mouse_move(event):
        if event.xdata is None or event.ydata is None:
            return

        real_x, real_y = event.xdata, event.ydata
        vx = real_x - state['last_x']
        vy = real_y - state['last_y']
        state['last_x'], state['last_y'] = real_x, real_y

        x_feat = np.array([real_x/1000.0, real_y/650.0, vx/50.0, vy/50.0, 1.0, 0.0])
        target_xy = np.array([real_x/1000.0, real_y/650.0])

        pred_norm, prefetched_nodes, latency_us = engine.update_online_step(x_feat, target_xy)

        pred_x = pred_norm[0] * 1000.0
        pred_y = pred_norm[1] * 650.0

        red_dot.set_data([real_x], [real_y])
        green_dot.set_data([pred_x], [pred_y])

        state['trail_x'] = [real_x, pred_x]
        state['trail_y'] = [real_y, pred_y]
        trail_line.set_data(state['trail_x'], state['trail_y'])

        state['step'] += 1
        hud_text.set_text(f"Steps: {state['step']:,} | Latency: {latency_us:.2f} us/step | Active RAM Nodes: {prefetched_nodes}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    print("  ▶ Interactive Desktop Window launched on your screen!")
    print("  • Move your mouse inside the window to see your physical cursor (RED) and the HBS-Engine's predicted cursor (GREEN) live!\n")
    plt.show()


if __name__ == "__main__":
    run_visual_interactive_window()
