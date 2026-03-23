#!/usr/bin/env python3
"""
Apply RoPE (Rotary Position Embeddings) to pre-extracted Q, K vectors.

The original extract_vectors.py captured Q, K, V BEFORE RoPE was applied.
This script reads the existing JSONL, applies the exact same RoPE that
Llama-3-8B uses (standard RoPE with theta=500000), and writes a new file.

RoPE is applied to Q and K only — V is unchanged.

Speed tips (see argparse --help):
  - Install `orjson` (large win: serializes numpy without slow .tolist()).
  - Use `--fast` to skip per-layer norm / relative-change checks and extra prints.
  - cos/sin caches are reused when `sequence_length` repeats across examples.

No GPU required: pure numpy implementation matching HuggingFace Transformers'
apply_rotary_pos_emb + rotate_half.

Usage:
    python apply_rope_to_vectors.py
    python apply_rope_to_vectors.py --fast --num-examples 0
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

# ── Configuration (defaults; override via CLI) ─────────────────
INPUT_FILE = Path(__file__).parent.parent / "data" / "attention_vectors_infinitebench_math_calc_128k.json"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "attention_vectors_infinitebench_math_calc_128k_with_rope.json"
NUM_EXAMPLES = 50

# Llama-3-8B RoPE config (from model config.json)
ROPE_THETA = 500000.0
HEAD_DIM = 128


# ── RoPE implementation (matches HF Transformers exactly) ──────
@lru_cache(maxsize=128)
def compute_rope_cache(seq_len: int, head_dim: int = HEAD_DIM, rope_theta: float = ROPE_THETA) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cos/sin caches for RoPE.

    Matches LlamaRotaryEmbedding.forward():
        inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))
        freqs = outer(position_ids, inv_freq)
        emb = cat(freqs, freqs)
        cos, sin = emb.cos(), emb.sin()

    Cached by seq_len so repeated lengths (e.g. 8192) only compute once.
    """
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    position_ids = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(position_ids, inv_freq)  # [seq_len, head_dim//2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [seq_len, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
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


def apply_rotary_pos_emb(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches HF Transformers apply_rotary_pos_emb():
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def dumps_line(example: Dict[str, Any], use_orjson: bool) -> bytes:
    """Serialize one JSON object to a line (bytes for orjson, utf-8 for stdlib)."""
    if use_orjson:
        return orjson.dumps(example, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n"
    # stdlib json cannot encode ndarray — convert heavy arrays to lists
    for layer_key in ("first_layer", "last_layer"):
        if layer_key not in example:
            continue
        layer = example[layer_key]
        for key in ("Q", "K", "V"):
            if key in layer and isinstance(layer[key], np.ndarray):
                layer[key] = layer[key].tolist()
    return (json.dumps(example, separators=(",", ":")) + "\n").encode("utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply RoPE to pre-extracted Q, K in a JSONL file.")
    p.add_argument("--input", type=Path, default=INPUT_FILE, help="Input JSONL path")
    p.add_argument("--output", type=Path, default=OUTPUT_FILE, help="Output JSONL path")
    p.add_argument(
        "--num-examples",
        type=int,
        default=NUM_EXAMPLES,
        help=f"Max examples to process (default {NUM_EXAMPLES}). Use 0 for all lines in the file.",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip per-layer relative-change / norm checks and most per-layer prints (faster).",
    )
    p.add_argument(
        "--no-orjson",
        action="store_true",
        help="Force stdlib json + .tolist() even if orjson is installed (slow; for debugging).",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output
    num_examples = args.num_examples
    use_orjson = HAS_ORJSON and not args.no_orjson

    print("=" * 70)
    print("Apply RoPE to extracted attention vectors")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Examples: {'all' if num_examples == 0 else num_examples}")
    print(f"  RoPE theta: {ROPE_THETA}")
    print(f"  Head dim: {HEAD_DIM}")
    print(f"  JSON backend: {'orjson (fast)' if use_orjson else 'stdlib json + tolist() (slow)'}")
    if not HAS_ORJSON:
        print("  Tip: pip install orjson  # large speedup on writing large arrays")
    print(f"  Mode: {'fast (minimal logging)' if args.fast else 'verbose checks'}")
    print("=" * 70)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    t0 = time.time()
    n_written = 0
    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        for idx, line in enumerate(fin):
            if num_examples > 0 and idx >= num_examples:
                break

            example = json.loads(line)
            seq_len = example["sequence_length"]
            if not args.fast:
                print(
                    f"\n[{idx + 1}/{num_examples if num_examples > 0 else '…'}] example_id={example['example_id']}, "
                    f"seq_len={seq_len}, domain={example['domain']}"
                )

            cos, sin = compute_rope_cache(int(seq_len))

            for layer_key in ["first_layer", "last_layer"]:
                layer = example[layer_key]

                Q = np.asarray(layer["Q"], dtype=np.float32)
                K = np.asarray(layer["K"], dtype=np.float32)
                # V unchanged; keep as array for orjson path
                if not isinstance(layer.get("V"), np.ndarray):
                    layer["V"] = np.asarray(layer["V"], dtype=np.float32)

                if not args.fast:
                    print(f"  {layer_key}: Q{Q.shape}, K{K.shape} -> applying RoPE")

                Q_rope, K_rope = apply_rotary_pos_emb(Q, K, cos, sin)

                if not args.fast:
                    assert Q_rope.shape == Q.shape
                    assert K_rope.shape == K.shape
                    q_delta = np.linalg.norm(Q_rope - Q) / (np.linalg.norm(Q) + 1e-10)
                    k_delta = np.linalg.norm(K_rope - K) / (np.linalg.norm(K) + 1e-10)
                    print(f"    Relative change: Q={q_delta:.4f}, K={k_delta:.4f}")

                    q_norm_before = np.linalg.norm(Q, axis=-1)
                    q_norm_after = np.linalg.norm(Q_rope, axis=-1)
                    norm_err = np.max(np.abs(q_norm_after - q_norm_before) / (q_norm_before + 1e-10))
                    print(f"    Max norm change (should be ~0): {norm_err:.6f}")

                if use_orjson:
                    layer["Q"] = Q_rope
                    layer["K"] = K_rope
                else:
                    layer["Q"] = Q_rope.tolist()
                    layer["K"] = K_rope.tolist()

                layer["Q_shape"] = list(Q_rope.shape)
                layer["K_shape"] = list(K_rope.shape)

            example["rope_applied"] = True
            example["rope_config"] = {
                "rope_theta": ROPE_THETA,
                "head_dim": HEAD_DIM,
                "method": "standard_rope",
                "formula": "q_embed = (q * cos) + (rotate_half(q) * sin)",
                "note": "Matches HuggingFace Transformers LlamaRotaryEmbedding + apply_rotary_pos_emb",
            }

            fout.write(dumps_line(example, use_orjson))
            n_written += 1
            if args.fast and (idx + 1) % 10 == 0:
                print(f"  … {idx + 1} examples written …", flush=True)

    elapsed = time.time() - t0
    file_size_gb = output_path.stat().st_size / (1024**3)
    print("\n" + "=" * 70)
    print(f"Done! {n_written} examples processed in {elapsed:.1f}s ({n_written / elapsed:.2f} ex/s)")
    print(f"Output: {output_path} ({file_size_gb:.2f} GB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
