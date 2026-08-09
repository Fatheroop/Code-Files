#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
 INTERACTIVE VISUAL DESKTOP MOUSE & KEYBOARD GUI (BIOLOGICAL HBS-ENGINE V2.2)
 ──────────────────────────────────────────────────────────────────────────────
 Dedicated Tkinter Visual Window for Real-Time Mouse & Keystroke Tracking:
  1. Opens a 1000x700 Visual Window on your screen.
  2. RED DOT   = Your physical mouse position as you move inside the window.
  3. GREEN DOT = Biological HBS-Engine V2.2's online predicted cursor position
                 that physically follows and chases your mouse live!
  4. Real-Time HUD Bar displaying active pre-fetched RAM nodes, latency, and keystrokes.
════════════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import os
import tkinter as tk
import numpy as np


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

        # Cold Storage Weights (FP16 Precision)
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
# TKINTER DESKTOP INTERACTIVE GUI WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class InteractiveBrainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("BIOLOGICAL HBS-ENGINE V2.2 — LIVE DESKTOP VISUAL DEMO")
        self.root.geometry("1050x750")
        self.root.configure(bg="#0f172a")

        self.width = 1000
        self.height = 650

        # Header Title Banner
        title_label = tk.Label(
            root,
            text="🧠 BIOLOGICAL HUMAN-BRAIN SPIKING ENGINE V2.2 — REAL-TIME MOUSE & KEYBOARD DEMO",
            font=("Helvetica", 12, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
            pady=8
        )
        title_label.pack()

        # Canvas Window
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="#020617", highlightthickness=2, highlightbackground="#334155")
        self.canvas.pack(padx=10, pady=5)

        # Draw grid background
        for x in range(0, self.width, 50):
            self.canvas.create_line(x, 0, x, self.height, fill="#1e293b", dash=(2, 4))
        for y in range(0, self.height, 50):
            self.canvas.create_line(0, y, self.width, y, fill="#1e293b", dash=(2, 4))

        # HUD Info Bar
        self.hud_label = tk.Label(
            root,
            text="Move mouse inside box | Red Dot = Physical Mouse | Green Dot = HBS-Engine Real-Time Prediction",
            font=("Monospace", 10, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
            pady=5
        )
        self.hud_label.pack()

        # Initialize HBS-Engine Predictor
        self.engine = VisualHBSBrainEngine(input_dim=6, hidden_dim=64, n_neurons=16, max_prefetch=4, hebbian_lr=0.15, seed=42)

        # Cursor positions & state tracking
        self.real_x, self.real_y = 500, 325
        self.pred_x, self.pred_y = 500, 325
        self.last_x, self.last_y = 500, 325
        self.last_key = "None"
        self.step_count = 0

        # Canvas Cursor Dots
        self.red_dot = self.canvas.create_oval(self.real_x-8, self.real_y-8, self.real_x+8, self.real_y+8, fill="#ef4444", outline="#ffffff", width=2)
        self.green_dot = self.canvas.create_oval(self.pred_x-10, self.pred_y-10, self.pred_x+10, self.pred_y+10, fill="#22c55e", outline="#ffffff", width=2)
        self.trail_lines = []

        # Bind mouse & keyboard events
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Key>", self.on_key_press)

    def on_mouse_move(self, event):
        self.real_x = event.x
        self.real_y = event.y

        # Update physical red dot position
        self.canvas.coords(self.red_dot, self.real_x-8, self.real_y-8, self.real_x+8, self.real_y+8)

        vx = self.real_x - self.last_x
        vy = self.real_y - self.last_y
        self.last_x, self.last_y = self.real_x, self.real_y

        # Scale features for HBS-Engine online prediction step
        x_feat = np.array([self.real_x/self.width, self.real_y/self.height, vx/50.0, vy/50.0, 1.0, 0.0])
        target_xy = np.array([self.real_x/self.width, self.real_y/self.height])

        # HBS-Engine real-time Hebbian update
        pred_norm, prefetched_nodes, latency_us = self.engine.update_online_step(x_feat, target_xy)

        # Smooth predicted green dot coordinates
        self.pred_x = int(pred_norm[0] * self.width)
        self.pred_y = int(pred_norm[1] * self.height)

        self.canvas.coords(self.green_dot, self.pred_x-10, self.pred_y-10, self.pred_x+10, self.pred_y+10)

        # Draw trajectory line between real & predicted cursors
        line = self.canvas.create_line(self.real_x, self.real_y, self.pred_x, self.pred_y, fill="#38bdf8", width=2, dash=(2, 2))
        self.trail_lines.append(line)
        if len(self.trail_lines) > 15:
            self.canvas.delete(self.trail_lines.pop(0))

        self.step_count += 1
        self.hud_label.config(
            text=f"Steps: {self.step_count:,} | Latency: {latency_us:.2f} μs/step | Active RAM Nodes: {prefetched_nodes} | Key Pressed: {self.last_key}"
        )

    def on_key_press(self, event):
        self.last_key = event.char if event.char else event.keysym
        # Create visual ripple effect on key press
        ripple = self.canvas.create_oval(self.real_x-30, self.real_y-30, self.real_x+30, self.real_y+30, outline="#eab308", width=3)
        self.root.after(300, lambda: self.canvas.delete(ripple))


def run_interactive_gui():
    root = tk.Tk()
    app = InteractiveBrainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    run_interactive_gui()
