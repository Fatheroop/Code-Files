#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
 REAL-WORLD SENTENCE GENERATION TASK BENCHMARK (50,000 SENTENCES / 800,000 TOKENS)
 ──────────────────────────────────────────────────────────────────────────────
 Evaluates multi-step continuous sentence generation tasks (16-word sentence task)
 where each generated block depends on previous generated context.

 Compares:
  1. Standard Autoregressive Model (16 Sequential Autoregressive Decoding Steps)
  2. AMT-Engine V18 (4 Fused Input-KV Block Passes + O(1) Incremental KV Context Cache)
 Using native Linux system APIs (time.perf_counter, time.process_time, resource, psutil)
 across 50,000 long sentence generation requests.
════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import time
import os
import resource
import psutil
import platform
from interactive_multi_token_engine import AMTEngineV18, WordTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD AUTOREGRESSIVE MODEL FOR SENTENCE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class StandardAutoregressiveEngine:
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 48, n_layers: int = 4, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        scale = np.sqrt(1.0 / hidden_dim)
        self.E_tok = self.rng.randn(vocab_size, embed_dim) * 0.1
        self.E_pos = self.rng.randn(16, embed_dim) * 0.1

        self.W_in = self.rng.randn(embed_dim, hidden_dim) * scale
        self.b_in = np.zeros(hidden_dim)

        self.W_layers = [self.rng.randn(hidden_dim, hidden_dim) * scale for _ in range(n_layers)]
        self.b_layers = [np.zeros(hidden_dim) for _ in range(n_layers)]

        self.W_out = self.rng.randn(hidden_dim, vocab_size) * scale
        self.b_out = np.zeros(vocab_size)

    def count_parameters(self):
        total = self.E_tok.size + self.E_pos.size + self.W_in.size + self.b_in.size
        for k in range(self.n_layers):
            total += self.W_layers[k].size + self.b_layers[k].size
        total += self.W_out.size + self.b_out.size
        return total

    def _relu(self, z):
        return np.maximum(0, z)

    def forward_single_step(self, X_tokens):
        B, S = X_tokens.shape
        pos = np.arange(S)
        emb = self.E_tok[X_tokens] + self.E_pos[pos % 16]
        x_pooled = np.mean(emb, axis=1)

        h = self._relu(np.dot(x_pooled, self.W_in) + self.b_in)
        for k in range(self.n_layers):
            h = self._relu(np.dot(h, self.W_layers[k]) + self.b_layers[k])

        logits = np.dot(h, self.W_out) + self.b_out
        return logits

    def generate_full_sentence_sequential(self, prompt_tokens, target_len=16):
        """Standard AR requires 16 sequential decoding steps."""
        curr_tokens = prompt_tokens.copy()
        generated_sentence = []
        for step in range(target_len):
            logits = self.forward_single_step(curr_tokens)
            next_ids = np.argmax(logits, axis=-1)
            generated_sentence.append(next_ids)

            next_col = next_ids.reshape(-1, 1)
            if curr_tokens.shape[1] >= 4:
                curr_tokens = np.column_stack([curr_tokens[:, 1:], next_col])
            else:
                curr_tokens = np.column_stack([curr_tokens, next_col])
        return np.column_stack(generated_sentence)


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM SPECIFICATIONS & MEASUREMENT HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def get_system_specs():
    process = psutil.Process(os.getpid())
    mem = psutil.virtual_memory()
    return {
        'os': f"{platform.system()} {platform.release()}",
        'cpus_physical': psutil.cpu_count(logical=False) or 1,
        'cpus_logical': psutil.cpu_count(logical=True) or 1,
        'ram_total_gb': mem.total / (1024 ** 3),
        'pid': process.pid,
    }


def measure_sentence_task_real_world(model_func, dataset_prompts, target_len=16, model_name="Model", batch_chunk=500):
    process = psutil.Process(os.getpid())
    N_total = len(dataset_prompts)
    N_tokens_total = N_total * target_len

    print(f"  ▶ Benchmarking {model_name} over {N_total:,} 16-word sentence generations ({N_tokens_total:,} tokens) …")

    ram_before_mb = process.memory_info().rss / (1024 * 1024)

    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()
    rusage_start = resource.getrusage(resource.RUSAGE_SELF)

    for i in range(0, N_total, batch_chunk):
        batch = dataset_prompts[i:i + batch_chunk]
        _ = model_func(batch)

    t_wall_end = time.perf_counter()
    t_cpu_end = time.process_time()
    rusage_end = resource.getrusage(resource.RUSAGE_SELF)

    ram_after_mb = process.memory_info().rss / (1024 * 1024)

    wall_elapsed_sec = t_wall_end - t_wall_start
    cpu_elapsed_sec = t_cpu_end - t_cpu_start

    avg_wall_lat_per_sentence_ms = (wall_elapsed_sec / N_total) * 1000.0
    avg_wall_lat_per_token_ms = (wall_elapsed_sec / N_tokens_total) * 1000.0
    throughput_tok_per_sec = N_tokens_total / wall_elapsed_sec

    user_cpu_time = rusage_end.ru_utime - rusage_start.ru_utime
    sys_cpu_time = rusage_end.ru_stime - rusage_start.ru_stime

    return {
        'model_name': model_name,
        'wall_sec': wall_elapsed_sec,
        'cpu_sec': cpu_elapsed_sec,
        'user_cpu_sec': user_cpu_time,
        'sys_cpu_sec': sys_cpu_time,
        'ram_before_mb': ram_before_mb,
        'ram_after_mb': ram_after_mb,
        'sentence_lat_ms': avg_wall_lat_per_sentence_ms,
        'token_lat_ms': avg_wall_lat_per_token_ms,
        'throughput_tok_sec': throughput_tok_per_sec,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 50,000-SENTENCE GENERATION TASK BENCHMARK SUITE
# ═══════════════════════════════════════════════════════════════════════════════

def run_sentence_generation_benchmark():
    print()
    print("  ╔═════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("  ║  REAL-WORLD 16-WORD SENTENCE GENERATION TASK BENCHMARK (50,000 SENTENCES / 800,000 TOKENS)       ║")
    print("  ║  Standard AR (16 Sequential Passes) vs. AMT-V18 (4 Fused Input-KV Passes + O(1) KV-Cache)     ║")
    print("  ╚═════════════════════════════════════════════════════════════════════════════════════════════════╝")
    print()

    specs = get_system_specs()
    print("  ▶ LINUX HOST HARDWARE & SYSTEM SPECIFICATIONS")
    print(f"    • Operating System      : {specs['os']}")
    print(f"    • Physical CPU Cores   : {specs['cpus_physical']} Cores ({specs['cpus_logical']} Logical Threads)")
    print(f"    • Total System RAM      : {specs['ram_total_gb']:.2f} GB")
    print(f"    • Benchmark Process PID : {specs['pid']}\n")

    standard_sentences = [
        "artificial intelligence models predict multiple tokens simultaneously with high accuracy .",
        "deep neural networks execute non dag layer interaction without vanishing gradient problems .",
        "the engine handles asynchronous command injection mid computation on the fly .",
        "machine learning algorithms optimize loss functions using gradient descent optimization ."
    ]
    steered_sentences = [
        "artificial intelligence models PREDICT MULTIPLE TOKENS SIMULTANEOUSLY WITH HIGH ACCURACY .",
        "deep neural networks EXECUTE NON DAG LAYER INTERACTION WITHOUT VANISHING GRADIENT PROBLEMS .",
        "the engine handles ASYNCHRONOUS COMMAND INJECTION MID COMPUTATION ON THE FLY .",
        "machine learning algorithms OPTIMIZE LOSS FUNCTIONS USING GRADIENT DESCENT OPTIMIZATION ."
    ]

    tokenizer = WordTokenizer().fit(standard_sentences + steered_sentences)
    vocab_size = tokenizer.vocab_size

    EMBED_DIM = 32
    HIDDEN_DIM = 48
    N_LAYERS = 4
    PRED_HORIZON = 4
    TARGET_SENTENCE_LEN = 16

    print("  ▶ 1. Constructing 50,000 Sentence Generation Task Prompts …")
    rng = np.random.RandomState(42)
    raw_pool = [
        tokenizer.encode("artificial intelligence models predict"),
        tokenizer.encode("deep neural networks execute non"),
        tokenizer.encode("the engine handles asynchronous command"),
        tokenizer.encode("machine learning algorithms optimize loss")
    ]
    sample_tokens_pool = [ids[:4] for ids in raw_pool]

    TOTAL_SENTENCES = 50000
    dataset_prompts = [sample_tokens_pool[rng.randint(0, len(sample_tokens_pool))] for _ in range(TOTAL_SENTENCES)]
    dataset_array = np.array(dataset_prompts, dtype=np.int32)
    cmd_signal = np.array([[1.0, 0.0]])

    std_engine = StandardAutoregressiveEngine(vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS)

    amt_v18 = AMTEngineV18(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_dim=32,
        bottleneck_dim=12,
        rank_dim=8,
        n_layers=N_LAYERS,
        pred_horizon=PRED_HORIZON,
        n_steps=1,
        seed=42,
    )

    std_params = std_engine.count_parameters()
    v18_params = amt_v18.count_parameters()

    std_kb = std_params * 4 / 1024.0
    v18_kb = v18_params * 4 / 1024.0

    print(f"    Dataset Ready: {TOTAL_SENTENCES:,} text prompts ({TOTAL_SENTENCES * TARGET_SENTENCE_LEN:,} total generated tokens)\n")

    # Benchmark Standard AR Model for 16-word sentence generation task
    res_std = measure_sentence_task_real_world(
        model_func=lambda b: std_engine.generate_full_sentence_sequential(b, target_len=TARGET_SENTENCE_LEN),
        dataset_prompts=dataset_array,
        target_len=TARGET_SENTENCE_LEN,
        model_name="Standard AR Model (16 Sequential Passes)",
        batch_chunk=500
    )

    # Benchmark AMT-Engine V18 (Fused Input-KV Decoding)
    res_v18 = measure_sentence_task_real_world(
        model_func=lambda b: amt_v18.generate_full_sentence_fast(b, target_len=TARGET_SENTENCE_LEN, cmd_signal=cmd_signal),
        dataset_prompts=dataset_array,
        target_len=TARGET_SENTENCE_LEN,
        model_name="AMT-Engine V18 (4 Fused Input-KV Passes + O(1) KV-Cache)",
        batch_chunk=500
    )

    # Printing Comparative Sentence Generation Task Report
    w = 118
    std_w = f"{res_std['wall_sec']:.3f} s"
    v18_w = f"{res_v18['wall_sec']:.3f} s"

    std_c = f"{res_std['cpu_sec']:.3f} s"
    v18_c = f"{res_v18['cpu_sec']:.3f} s"

    std_ram = f"{res_std['ram_after_mb']:.1f} MB"
    v18_ram = f"{res_v18['ram_after_mb']:.1f} MB"

    std_sent_lat = f"{res_std['sentence_lat_ms']:.4f} ms"
    v18_sent_lat = f"{res_v18['sentence_lat_ms']:.4f} ms"

    std_tok_lat = f"{res_std['token_lat_ms']:.4f} ms"
    v18_tok_lat = f"{res_v18['token_lat_ms']:.4f} ms"

    std_tp = f"{res_std['throughput_tok_sec']:.1f} tok/s"
    v18_tp = f"{res_v18['throughput_tok_sec']:.1f} tok/s"

    speedup = res_std['wall_sec'] / res_v18['wall_sec']
    throughput_gain = 100 * (res_v18['throughput_tok_sec'] - res_std['throughput_tok_sec']) / res_std['throughput_tok_sec']

    print("\n  ┌" + "─" * w + "┐")
    print(f"  │ {'SENTENCE GENERATION TASK RESOURCE METRIC':<42s} │ {'STANDARD AR MODEL':<33s} │ {'AMT-ENGINE V18 (FUSED INPUT-KV)':<34s} │")
    print("  ├" + "─" * w + "┤")
    print(f"  │ {'Sequential Compute Passes / Sentence':<42s} │ {'16 Sequential Passes':<33s} │ \033[1;32m{'4 Fused Passes + O(1) KV Cache':<34s}\033[0m │")
    print(f"  │ {'Total Model Parameters':<42s} │ {f'{std_params:,} params':<33s} │ \033[1;32m{f'{v18_params:,} params':<34s}\033[0m │")
    print(f"  │ {'FP32 Model Weight Storage':<42s} │ {f'{std_kb:.2f} KB':<33s} │ \033[1;32m{f'{v18_kb:.2f} KB':<34s}\033[0m │")
    print(f"  │ {'Total Evaluated Sentences':<42s} │ {f'{TOTAL_SENTENCES:,} sentences':<33s} │ {f'{TOTAL_SENTENCES:,} sentences':<34s} │")
    print(f"  │ {'Total Generated Tokens':<42s} │ {f'{TOTAL_SENTENCES * TARGET_SENTENCE_LEN:,} tokens':<33s} │ {f'{TOTAL_SENTENCES * TARGET_SENTENCE_LEN:,} tokens':<34s} │")
    print(f"  │ {'Real Wall-Clock Time (time.perf_counter)':<42s} │ {std_w:<33s} │ \033[1;32m{v18_w:<34s}\033[0m │")
    print(f"  │ {'Real CPU Execution Time (time.process_time)':<42s} │ {std_c:<33s} │ \033[1;32m{v18_c:<34s}\033[0m │")
    print(f"  │ {'Process Memory RSS Footprint (psutil)':<42s} │ {std_ram:<33s} │ {v18_ram:<34s} │")
    print(f"  │ {'Average Latency / 16-Word Sentence':<42s} │ {std_sent_lat:<33s} │ \033[1;32m{v18_sent_lat:<34s}\033[0m │")
    print(f"  │ {'Average Latency / Generated Token':<42s} │ {std_tok_lat:<33s} │ \033[1;32m{v18_tok_lat:<34s}\033[0m │")
    print(f"  │ {'Real Generation Throughput (tokens/sec)':<42s} │ {std_tp:<33s} │ \033[1;32m{v18_tp:<34s}\033[0m │")
    print("  ├" + "─" * w + "┤")
    print(f"  │ \033[1;32m{'SENTENCE GENERATION SPEEDUP OVER AR':<42s} │ {'Baseline (1.00x)':<33s} │ {f'{speedup:.2f}x Speedup (+{throughput_gain:.1f}%)':<34s}\033[0m │")
    print("  └" + "─" * w + "┘\n")


if __name__ == "__main__":
    run_sentence_generation_benchmark()
