#!/usr/bin/env python3
"""
Apply RoPE (Rotary Position Embeddings) to pre-extracted Q, K vectors.

The original extract_vectors.py captured Q, K, V BEFORE RoPE was applied.
This script reads the existing JSONL, applies the exact same RoPE that
Llama-3-8B uses (standard RoPE with theta=500000), and writes a new file.

RoPE is applied to Q and K only — V is unchanged.

No GPU required: pure numpy implementation matching HuggingFace Transformers'
apply_rotary_pos_emb + rotate_half.

Usage:
    python apply_rope_to_vectors.py
"""

import json
import numpy as np
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
INPUT_FILE = Path(__file__).parent.parent / "data" / "attention_vectors_long_bench_llama_8b.jsonl"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "attention_vectors_llama_8b_with_rope.jsonl"
NUM_EXAMPLES = 50

# Llama-3-8B RoPE config (from model config.json)
ROPE_THETA = 500000.0
HEAD_DIM = 128


# ── RoPE implementation (matches HF Transformers exactly) ──────
def compute_rope_cache(seq_len, head_dim=HEAD_DIM, rope_theta=ROPE_THETA):
    """
    Compute cos/sin caches for RoPE.

    Matches LlamaRotaryEmbedding.forward():
        inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))
        freqs = outer(position_ids, inv_freq)
        emb = cat(freqs, freqs)
        cos, sin = emb.cos(), emb.sin()
    """
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    position_ids = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(position_ids, inv_freq)  # [seq_len, head_dim//2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [seq_len, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    """
    Matches HF Transformers rotate_half():
        x1 = x[..., :dim//2]
        x2 = x[..., dim//2:]
        return cat(-x2, x1)
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Matches HF Transformers apply_rotary_pos_emb():
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ── Main ───────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Apply RoPE to extracted attention vectors")
    print(f"  Input:  {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Examples: {NUM_EXAMPLES}")
    print(f"  RoPE theta: {ROPE_THETA}")
    print(f"  Head dim: {HEAD_DIM}")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    t0 = time.time()
    with open(INPUT_FILE, 'r') as fin, open(OUTPUT_FILE, 'w') as fout:
        for idx, line in enumerate(fin):
            if idx >= NUM_EXAMPLES:
                break

            example = json.loads(line)
            seq_len = example['sequence_length']
            print(f"\n[{idx+1}/{NUM_EXAMPLES}] example_id={example['example_id']}, "
                  f"seq_len={seq_len}, domain={example['domain']}")

            cos, sin = compute_rope_cache(seq_len)

            for layer_key in ['first_layer', 'last_layer']:
                layer = example[layer_key]

                Q = np.array(layer['Q'], dtype=np.float32)  # [seq_len, head_dim]
                K = np.array(layer['K'], dtype=np.float32)
                V = np.array(layer['V'], dtype=np.float32)

                print(f"  {layer_key}: Q{Q.shape}, K{K.shape} -> applying RoPE")

                Q_rope, K_rope = apply_rotary_pos_emb(Q, K, cos, sin)

                # Sanity checks
                assert Q_rope.shape == Q.shape
                assert K_rope.shape == K.shape
                q_delta = np.linalg.norm(Q_rope - Q) / (np.linalg.norm(Q) + 1e-10)
                k_delta = np.linalg.norm(K_rope - K) / (np.linalg.norm(K) + 1e-10)
                print(f"    Relative change: Q={q_delta:.4f}, K={k_delta:.4f}")

                # Verify norms preserved (RoPE is a rotation — should preserve norms)
                q_norm_before = np.linalg.norm(Q, axis=-1)
                q_norm_after = np.linalg.norm(Q_rope, axis=-1)
                norm_err = np.max(np.abs(q_norm_after - q_norm_before) / (q_norm_before + 1e-10))
                print(f"    Max norm change (should be ~0): {norm_err:.6f}")

                layer['Q'] = Q_rope.tolist()
                layer['K'] = K_rope.tolist()
                # V unchanged
                layer['Q_shape'] = list(Q_rope.shape)
                layer['K_shape'] = list(K_rope.shape)

            example['rope_applied'] = True
            example['rope_config'] = {
                'rope_theta': ROPE_THETA,
                'head_dim': HEAD_DIM,
                'method': 'standard_rope',
                'formula': 'q_embed = (q * cos) + (rotate_half(q) * sin)',
                'note': 'Matches HuggingFace Transformers LlamaRotaryEmbedding + apply_rotary_pos_emb'
            }

            fout.write(json.dumps(example) + '\n')
            fout.flush()

    elapsed = time.time() - t0
    file_size_gb = OUTPUT_FILE.stat().st_size / (1024 ** 3)
    print("\n" + "=" * 70)
    print(f"Done! {NUM_EXAMPLES} examples processed in {elapsed:.1f}s")
    print(f"Output: {OUTPUT_FILE} ({file_size_gb:.2f} GB)")
    print("=" * 70)


if __name__ == '__main__':
    main()
