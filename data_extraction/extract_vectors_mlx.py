#!/usr/bin/env python3
"""
Extract Q, K, V vectors with Apple Silicon MLX backend.

This script targets the same JSONL schema produced by extract_vectors.py, while
running model inference through MLX/MLX-LM.

Key points:
- Supports dataset formats: {"examples": [...]}, {"data": [...]}, or top-level [...]
- Extracts two layers (defaults: layer 11 with Q head 11 → `first_layer`, layer 19 with Q head 0 → `last_layer`)
- Supports single-head mode (default) and all-head mode
- Writes one JSON object per example (JSONL)
- By default only InfiniteBench examples with `task == math_calc` are processed (`--only-task` /
  `--no-task-filter`; see argparse).

Note:
- Default model is a **non-quantized** MLX port (weights in **bfloat16**, same class as “full”
  precision—not 4-bit/8-bit). PyTorch `extract_vectors.py` uses fp16; both are full-precision
  formats (bf16 vs fp16).
- For mlx_lm Llama, we step through `LlamaModel.layers` and read `q_proj`/`k_proj`/`v_proj`
  on the RMSNorm'ed sublayer input. Use `--rope-stage pre` (default) for Q/K **before** RoPE,
  or `--rope-stage post` for Q/K **after** the same `self_attn.rope` as the forward pass.
  RoPE never applies to V; V is always the linear projection output.
- Optional per-head tail attention entropy stats: pass `--tail-entropy` (slow; off by default).
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Mean attention entropy is computed over this fraction of query positions at the end of the sequence.
TAIL_ENTROPY_QUERY_FRACTION = 0.1
NEAR_ZERO_NORM_EPS = 1e-8


def _import_mlx() -> Tuple[Any, Any]:
    try:
        import mlx.core as mx
        from mlx_lm import load
    except Exception as exc:
        raise RuntimeError(
            "Missing MLX dependencies. Install with e.g.:\n"
            "  pip install mlx mlx-lm\n"
            f"Import error: {exc}"
        ) from exc
    return mx, load


def load_examples(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if isinstance(data.get("examples"), list):
            return data["examples"]
        if isinstance(data.get("data"), list):
            return data["data"]
        raise ValueError(
            "Unsupported JSON object format. Expected key 'examples' or 'data' containing a list."
        )

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported dataset format. Expected top-level object or list.")


def filter_examples_by_task(
    examples: List[Dict[str, Any]],
    only_task: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Keep examples whose InfiniteBench `task` matches `only_task`, or LongBench `sub_domain`
    if it matches exactly (e.g. rare). If `only_task` is None or empty, return all examples.
    """
    if not only_task:
        return examples
    out: List[Dict[str, Any]] = []
    for e in examples:
        if e.get("task") == only_task:
            out.append(e)
            continue
        if e.get("sub_domain") == only_task:
            out.append(e)
    return out


def example_metadata_for_jsonl(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    LongBench uses `domain` / `sub_domain`. InfiniteBench uses `task`, `task_description`, `id`.
    Never require `domain` to be present.
    """
    eid = example.get("_id", example.get("id", f"example_{idx}"))
    domain = example.get("domain")
    if domain is None or domain == "":
        domain = example.get("task_description") or example.get("task") or "Unknown"
    sub = example.get("sub_domain")
    if sub is None or sub == "":
        sub = example.get("task") or "Unknown"
    return {
        "example_id": eid,
        "domain": domain,
        "sub_domain": sub,
        "question": example.get("question", ""),
        "answer": example.get("answer", ""),
    }


def get_model_cfg(model: Any) -> Any:
    # Common places for config-like objects in HF/MLX wrappers.
    for attr in ("config", "args", "model_args"):
        if hasattr(model, attr):
            return getattr(model, attr)
    if hasattr(model, "model"):
        inner = getattr(model, "model")
        for attr in ("config", "args", "model_args"):
            if hasattr(inner, attr):
                return getattr(inner, attr)
    raise ValueError("Could not locate model config/args object.")


def cfg_get(cfg: Any, names: List[str], default: Optional[int] = None) -> Optional[int]:
    for n in names:
        if hasattr(cfg, n):
            return int(getattr(cfg, n))
        if isinstance(cfg, dict) and n in cfg:
            return int(cfg[n])
    return default


def cfg_get_float(cfg: Any, names: List[str], default: Optional[float] = None) -> Optional[float]:
    for n in names:
        if hasattr(cfg, n):
            return float(getattr(cfg, n))
        if isinstance(cfg, dict) and n in cfg:
            return float(cfg[n])
    return default


def cfg_get_raw(cfg: Any, names: List[str], default: Any = None) -> Any:
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
        if isinstance(cfg, dict) and n in cfg:
            return cfg[n]
    return default


def rope_json_fields(cfg: Any, post_rope: bool) -> Tuple[bool, Dict[str, Any], str]:
    """
    Fields for JSONL compatibility with downstream tools (e.g. attention_concentration.py).

    post_rope=True: Q/K match mlx_lm Attention after RoPE; rope_applied=True.
    post_rope=False: Q/K after linear projections only; rope_applied=False.
    V is never rotated; rope_config always notes v_rope=False.
    """
    theta = cfg_get_float(cfg, ["rope_theta"], default=500000.0)
    rope_traditional = bool(cfg_get_raw(cfg, ["rope_traditional"], default=False))
    rope_scaling = cfg_get_raw(cfg, ["rope_scaling"], default=None)
    max_pos = cfg_get(cfg, ["max_position_embeddings"], default=None)

    base_rope_cfg: Dict[str, Any] = {
        "rope_theta": theta,
        "rope_traditional": rope_traditional,
        "max_position_embeddings": max_pos,
        "rope_scaling": rope_scaling,
        "v_rope": False,
    }
    if post_rope:
        return (
            True,
            {
                **base_rope_cfg,
                "method": "mlx_lm_attention_rope",
                "note": "Q/K captured after self_attn.rope() using model config.",
            },
            "post_rope",
        )
    return (
        False,
        {
            **base_rope_cfg,
            "method": "pre_projection_only",
            "note": "Q/K captured before mlx_lm RoPE. Re-run with --rope-stage post for Q/K after RoPE.",
        },
        "pre_rope",
    )


def get_llama_backbone(model: Any) -> Any:
    """
    mlx_lm Llama: outer `Model` has `.model` as `LlamaModel` with embed_tokens + layers.
    """
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens") and hasattr(model.model, "layers"):
        return model.model
    raise ValueError(
        "Unsupported MLX model layout. This script expects mlx_lm Llama (Model.model = LlamaModel)."
    )


def get_layers_root(model: Any) -> Any:
    return get_llama_backbone(model).layers


def compute_mean_tail_attention_entropy_per_head(
    q: np.ndarray,
    k: np.ndarray,
    group_size: int,
    last_query_fraction: float = TAIL_ENTROPY_QUERY_FRACTION,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    For each Q head, compute the mean causal attention entropy (natural log, nats) over
    query positions in the last `last_query_fraction` of the sequence (inclusive).

    q: [num_heads, seq_len, head_dim]
    k: [num_kv_heads, seq_len, head_dim]
    Returns: [num_heads] float64
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    n_heads, seq_len, hd = q.shape
    n_kv = k.shape[0]
    if n_heads != n_kv * group_size:
        raise ValueError(f"Head/KV layout mismatch: n_heads={n_heads}, n_kv={n_kv}, group_size={group_size}")
    scale = 1.0 / np.sqrt(float(hd))
    n_tail = max(1, int(np.ceil(seq_len * last_query_fraction)))
    tail_start = seq_len - n_tail
    if tail_start < 0:
        tail_start = 0

    entropies = np.zeros(n_heads, dtype=np.float64)
    n_queries = seq_len - tail_start
    for h in range(n_heads):
        kv = h // group_size
        qh = q[h]
        kh = k[kv]
        acc = 0.0
        for i in range(tail_start, seq_len):
            logits = (qh[i] @ kh[: i + 1].T) * scale
            m = float(np.max(logits))
            exp_logits = np.exp(logits - m)
            s = float(np.sum(exp_logits)) + eps
            p = exp_logits / s
            p = np.clip(p, eps, 1.0)
            acc += float(-np.sum(p * np.log(p)))
        entropies[h] = acc / max(n_queries, 1)
    return entropies


def format_per_head_entropy_block(
    layer_desc: str,
    avg_per_head: np.ndarray,
    heads_per_line: int = 8,
) -> str:
    """
    Multi-line string: running-mean tail entropy (nats) for each Q head index.
    """
    avg_per_head = np.asarray(avg_per_head, dtype=np.float64)
    n = int(avg_per_head.shape[0])
    lines: List[str] = [
        f"    {layer_desc} — running mean H (nats) for heads 0..{n - 1}:",
    ]
    for start in range(0, n, heads_per_line):
        end = min(start + heads_per_line, n)
        parts = [f"h{i}={avg_per_head[i]:.4f}" for i in range(start, end)]
        lines.append("      " + "  ".join(parts))
    hm = int(np.argmax(avg_per_head))
    lines.append(f"      (highest running mean: head {hm} = {avg_per_head[hm]:.4f} nats)")
    return "\n".join(lines)


def near_zero_norm_stats(x: np.ndarray, eps: float = NEAR_ZERO_NORM_EPS) -> Tuple[int, int, float]:
    """
    Treat the last axis as vector dim. Return:
      (near_zero_count, total_vectors, percent_near_zero)
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 0:
        nrm = float(abs(arr))
        nz = 1 if nrm <= eps else 0
        return nz, 1, 100.0 * nz
    vecs = arr.reshape(-1, arr.shape[-1])
    norms = np.linalg.norm(vecs, axis=1)
    near_zero = int(np.count_nonzero(norms <= eps))
    total = int(norms.size)
    pct = 100.0 * near_zero / total if total > 0 else 0.0
    return near_zero, total, pct


def to_numpy(x: Any) -> np.ndarray:
    """
    MLX arrays (esp. bfloat16) cannot always be passed straight to np.array(..., dtype=float32);
    that can raise PEP 3118 buffer errors. Cast to float32 in MLX, eval, then copy to NumPy.
    """
    import mlx.core as mx

    if isinstance(x, mx.array):
        if x.dtype != mx.float32:
            x = x.astype(mx.float32)
        mx.eval(x)
        return np.array(x.tolist(), dtype=np.float32, copy=True, order="C")

    return np.array(x, dtype=np.float32, copy=True, order="C")


def to_numpy_chunked(x: Any, seq_chunk: int = 2048) -> np.ndarray:
    """
    Convert an MLX array to numpy in chunks along the sequence dimension to limit
    peak Python-object memory.  For a [seq_len, dim] array with seq_len=45000 and
    seq_chunk=2048, we do ~22 small tolist() calls instead of one giant one.
    """
    import mlx.core as mx

    if not isinstance(x, mx.array):
        return np.array(x, dtype=np.float32, copy=True, order="C")

    if x.dtype != mx.float32:
        x = x.astype(mx.float32)
    mx.eval(x)

    if x.ndim != 2 or x.shape[0] <= seq_chunk:
        return np.array(x.tolist(), dtype=np.float32, copy=True, order="C")

    seq_len, dim = x.shape
    result = np.empty((seq_len, dim), dtype=np.float32)
    for start in range(0, seq_len, seq_chunk):
        end = min(start + seq_chunk, seq_len)
        chunk = x[start:end]
        mx.eval(chunk)
        result[start:end] = np.array(chunk.tolist(), dtype=np.float32, copy=True, order="C")
    return result


def tokenize(tokenizer: Any, prompt: str, max_length: int) -> np.ndarray:
    # mlx_lm returns TokenizerWrapper: not callable; forward to HF tokenizer.
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    enc = inner(prompt, return_tensors="np", truncation=True, max_length=max_length)
    if isinstance(enc, dict) and "input_ids" in enc:
        ids = enc["input_ids"]
    else:
        ids = enc.input_ids
    return np.asarray(ids, dtype=np.int32)


H_NORM_WARN_THRESHOLD = 1e4  # Llama-8B h norms should be O(10); 10^4 means overflow

def _diagnose_h_norms_mlx(
    h: Any, mx: Any, layer_idx: int, bins: int = 10, verbose: bool = False
) -> bool:
    """Optionally print position-binned L2 norms of h. Returns True if norms look healthy."""
    h0 = h[0]  # [seq_len, hidden_dim]
    mx.eval(h0)
    norms = mx.sqrt(mx.sum(h0 * h0, axis=-1))  # [seq_len]
    mx.eval(norms)
    norms_np = np.array(norms.tolist(), dtype=np.float64)
    seq_len = len(norms_np)
    bin_edges = np.linspace(0, seq_len, bins + 1, dtype=int)

    n_bad = int(np.count_nonzero(~np.isfinite(norms_np) | (norms_np > H_NORM_WARN_THRESHOLD)))
    healthy = n_bad == 0
    tag = "" if healthy else f"  *** {n_bad}/{seq_len} positions exceed {H_NORM_WARN_THRESHOLD:.0e} or are inf/nan ***"

    if verbose:
        print(f"    h norms entering layer {layer_idx}  (seq_len={seq_len}):{tag}")
        for b in range(bins):
            s, e = int(bin_edges[b]), int(bin_edges[b + 1])
            if e <= s:
                continue
            bn = norms_np[s:e]
            nz = int(np.count_nonzero(bn < 1e-6))
            n_extreme = int(np.count_nonzero(~np.isfinite(bn) | (bn > H_NORM_WARN_THRESHOLD)))
            suffix = ""
            if nz > 0:
                suffix += f"  NEAR_ZERO={nz}"
            if n_extreme > 0:
                suffix += f"  EXTREME={n_extreme}"
            print(f"      [{s:>6d}:{e:>6d})  mean={bn.mean():.3f}  std={bn.std():.3f}  "
                  f"min={bn.min():.5f}  max={bn.max():.3f}{suffix}")
    del h0, norms
    return healthy


def _mlx_qkv_from_snapshot(
    layer: Any, h_snapshot: Any, mx: Any, post_rope: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a FULLY EVALUATED h snapshot, compute Q/K/V projections and convert to numpy.

    h_snapshot must already have been mx.eval'd and must not share a live graph
    with any ongoing forward-pass computation.
    """
    x_attn = layer.input_layernorm(h_snapshot)
    mx.eval(x_attn)

    attn = layer.self_attn
    q = attn.q_proj(x_attn)
    k = attn.k_proj(x_attn)
    v = attn.v_proj(x_attn)
    mx.eval(q, k, v)
    del x_attn

    bsz, seq_len, _ = q.shape
    if bsz != 1:
        raise ValueError(f"Expected batch size 1, got {bsz}")
    n_heads = attn.n_heads
    n_kv = attn.n_kv_heads
    hd = attn.head_dim

    q = q.reshape(bsz, seq_len, n_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, n_kv, hd).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, seq_len, n_kv, hd).transpose(0, 2, 1, 3)
    mx.eval(q, k, v)

    if post_rope:
        q = attn.rope(q)
        k = attn.rope(k)
        mx.eval(q, k)

    q0 = q[0]
    k0 = k[0]
    v0 = v[0]
    mx.eval(q0, k0, v0)
    del q, k, v

    # MLX-side norm sanity check.
    for tag, arr in [("Q", q0), ("K", k0), ("V", v0)]:
        norms = mx.sqrt(mx.sum(arr * arr, axis=-1))
        mx.eval(norms)
        norms_np = np.array(norms.tolist(), dtype=np.float64)
        near_zero = int(np.count_nonzero(norms_np < NEAR_ZERO_NORM_EPS))
        total = int(norms_np.size)
        if near_zero > 0:
            pct = 100.0 * near_zero / total
            print(f"    MLX-SIDE norm check: {tag} has {near_zero}/{total} ({pct:.1f}%) "
                  f"near-zero vectors BEFORE numpy conversion!")

    # Convert head-by-head with chunked tolist() to limit peak memory.
    q_parts: List[np.ndarray] = []
    for hi in range(n_heads):
        head = q0[hi]
        mx.eval(head)
        q_parts.append(to_numpy_chunked(head))
    q_np = np.stack(q_parts, axis=0)
    del q_parts, q0
    gc.collect()

    k_parts: List[np.ndarray] = []
    for hi in range(n_kv):
        head = k0[hi]
        mx.eval(head)
        k_parts.append(to_numpy_chunked(head))
    k_np = np.stack(k_parts, axis=0)
    del k_parts, k0
    gc.collect()

    v_parts: List[np.ndarray] = []
    for hi in range(n_kv):
        head = v0[hi]
        mx.eval(head)
        v_parts.append(to_numpy_chunked(head))
    v_np = np.stack(v_parts, axis=0)
    del v_parts, v0
    gc.collect()

    return q_np, k_np, v_np


def extract_qkv_first_last_llama_mlx(
    model: Any,
    mx: Any,
    input_ids_np: np.ndarray,
    first_layer_idx: int,
    last_layer_idx: int,
    post_rope: bool = False,
    diagnose_h_norms: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Run LlamaModel forward layer-by-layer and capture Q/K/V at two layers.

    Critical design choices to avoid MLX memory/lazy-eval corruption:
      1. mx.eval(h) is called after EVERY layer — this prevents the computation
         graph from growing across layers (which on 45K-token sequences causes
         Metal buffer scheduling issues).
      2. The forward pass and QKV extraction are SEPARATED: h snapshots are saved
         during the forward loop, and QKV projections are computed only AFTER the
         loop finishes.  This avoids duplicating ops that the layer itself computes
         (input_layernorm, q/k/v_proj) and eliminates potential buffer aliasing.
      3. Each h snapshot is mx.eval'd immediately — its memory is owned and cannot
         be reclaimed by subsequent layer computations.
    """
    from mlx_lm.models.base import create_attention_mask

    inner = get_llama_backbone(model)
    inputs = mx.array(input_ids_np)
    h = inner.embed_tokens(inputs)
    mx.eval(h)

    cache = [None] * len(inner.layers)
    fa_mask = create_attention_mask(h, cache[inner.fa_idx])
    swa_mask = None
    if inner.swa_idx is not None:
        swa_mask = create_attention_mask(
            h, cache[inner.swa_idx], window_size=inner.sliding_window
        )

    stop_after = max(first_layer_idx, last_layer_idx)
    h_snapshots: Dict[int, Any] = {}
    unhealthy_detected = False

    # ---- forward pass: run each layer eagerly, save h at target layers ----
    for i, layer in enumerate(inner.layers):
        # Save h BEFORE the layer runs (QKV is computed from layer input).
        if i in (first_layer_idx, last_layer_idx):
            mx.eval(h)
            h_snapshots[i] = h
            healthy = _diagnose_h_norms_mlx(h, mx, i, bins=10, verbose=diagnose_h_norms)
            if not healthy:
                unhealthy_detected = True
                if diagnose_h_norms:
                    print(
                        f"\n    *** WARNING: residual stream h has extreme norms at layer {i}. ***\n"
                        f"    This typically means the model is numerically unstable at this\n"
                        f"    sequence length or there is runtime instability in this pass.\n"
                        f"    Retrying this example may recover. If not, reduce context length\n"
                        f"    or try a different MLX / mlx-lm version.\n"
                    )

        if i >= stop_after:
            break

        mask = swa_mask if getattr(layer, "use_sliding", False) and swa_mask is not None else fa_mask
        h = layer(h, mask, cache=cache[i])
        mx.eval(h)  # <-- force evaluation every layer to prevent graph explosion

    del h  # free the last residual stream
    gc.collect()

    # ---- extract QKV from saved snapshots (no interference with forward pass) ----
    print(f"    Extracting QKV at layer {first_layer_idx} (from snapshot)...")
    first_out = _mlx_qkv_from_snapshot(
        inner.layers[first_layer_idx], h_snapshots[first_layer_idx], mx, post_rope,
    )
    del h_snapshots[first_layer_idx]
    gc.collect()

    print(f"    Extracting QKV at layer {last_layer_idx} (from snapshot)...")
    last_out = _mlx_qkv_from_snapshot(
        inner.layers[last_layer_idx], h_snapshots[last_layer_idx], mx, post_rope,
    )
    del h_snapshots[last_layer_idx]
    gc.collect()

    fq, fk, fv = first_out
    lq, lk, lv = last_out
    return fq, fk, fv, lq, lk, lv, unhealthy_detected


def build_single_head_result(
    example: Dict[str, Any],
    idx: int,
    seq_len: int,
    model_name: str,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group_size: int,
    first_layer_idx: int,
    last_layer_idx: int,
    head_idx_first_layer: int,
    head_idx_last_layer: int,
    first_q: np.ndarray,
    first_k: np.ndarray,
    first_v: np.ndarray,
    last_q: np.ndarray,
    last_k: np.ndarray,
    last_v: np.ndarray,
    rope_applied: bool,
    rope_config: Dict[str, Any],
    qkv_rope_stage: str,
) -> Dict[str, Any]:
    kv_head_first = head_idx_first_layer // group_size
    kv_head_last = head_idx_last_layer // group_size

    first_q_single = first_q[head_idx_first_layer, :, :]
    first_k_single = first_k[kv_head_first, :, :]
    first_v_single = first_v[kv_head_first, :, :]

    last_q_single = last_q[head_idx_last_layer, :, :]
    last_k_single = last_k[kv_head_last, :, :]
    last_v_single = last_v[kv_head_last, :, :]

    meta = example_metadata_for_jsonl(example, idx)
    return {
        **meta,
        "sequence_length": seq_len,
        "model_metadata": {
            "model_name": model_name,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "group_size": group_size,
            "runtime": "mlx",
            "task": example.get("task"),
        },
        "first_layer": {
            "layer_idx": first_layer_idx,
            "head_idx": head_idx_first_layer,
            "kv_head_idx": kv_head_first,
            "Q": first_q_single.tolist(),
            "K": first_k_single.tolist(),
            "V": first_v_single.tolist(),
            "Q_shape": list(first_q_single.shape),
            "K_shape": list(first_k_single.shape),
            "V_shape": list(first_v_single.shape),
        },
        "last_layer": {
            "layer_idx": last_layer_idx,
            "head_idx": head_idx_last_layer,
            "kv_head_idx": kv_head_last,
            "Q": last_q_single.tolist(),
            "K": last_k_single.tolist(),
            "V": last_v_single.tolist(),
            "Q_shape": list(last_q_single.shape),
            "K_shape": list(last_k_single.shape),
            "V_shape": list(last_v_single.shape),
        },
        "position_ids": list(range(seq_len)),
        "rope_applied": rope_applied,
        "rope_config": rope_config,
        "qkv_rope_stage": qkv_rope_stage,
        "usage_note": (
            f"qkv_rope_stage={qkv_rope_stage} | "
            f"first_layer: layer {first_layer_idx}, Q head {head_idx_first_layer}, KV head {kv_head_first}; "
            f"last_layer: layer {last_layer_idx}, Q head {head_idx_last_layer}, KV head {kv_head_last}"
        ),
    }


def build_all_heads_result(
    example: Dict[str, Any],
    idx: int,
    seq_len: int,
    model_name: str,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group_size: int,
    first_layer_idx: int,
    last_layer_idx: int,
    first_q: np.ndarray,
    first_k: np.ndarray,
    first_v: np.ndarray,
    last_q: np.ndarray,
    last_k: np.ndarray,
    last_v: np.ndarray,
    rope_applied: bool,
    rope_config: Dict[str, Any],
    qkv_rope_stage: str,
) -> Dict[str, Any]:
    head_mapping = []
    for q_head in range(num_heads):
        kv_head = q_head // group_size
        head_mapping.append({"q_head_idx": q_head, "kv_head_idx": kv_head, "group": q_head // group_size})

    meta = example_metadata_for_jsonl(example, idx)
    return {
        **meta,
        "sequence_length": seq_len,
        "model_metadata": {
            "model_name": model_name,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "group_size": group_size,
            "runtime": "mlx",
            "task": example.get("task"),
        },
        "head_mapping": head_mapping,
        "first_layer": {
            "layer_idx": first_layer_idx,
            "Q": first_q.tolist(),
            "K": first_k.tolist(),
            "V": first_v.tolist(),
            "Q_shape": list(first_q.shape),
            "K_shape": list(first_k.shape),
            "V_shape": list(first_v.shape),
        },
        "last_layer": {
            "layer_idx": last_layer_idx,
            "Q": last_q.tolist(),
            "K": last_k.tolist(),
            "V": last_v.tolist(),
            "Q_shape": list(last_q.shape),
            "K_shape": list(last_k.shape),
            "V_shape": list(last_v.shape),
        },
        "position_ids": list(range(seq_len)),
        "rope_applied": rope_applied,
        "rope_config": rope_config,
        "qkv_rope_stage": qkv_rope_stage,
        "usage_note": (
            f"qkv_rope_stage={qkv_rope_stage} | "
            "Q[q_head_idx, :, :] uses K[kv_head_idx, :, :] and V[kv_head_idx, :, :]. "
            "See head_mapping for Q->KV mapping."
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Q/K/V vectors using MLX backend")
    p.add_argument(
        "--model-name",
        type=str,
        #default="mlx-community/Meta-Llama-3-8B-Instruct",
        default="mlx-community/Meta-Llama-3.1-8B-bf16",
        help=(
            "HF repo id for an MLX-format model. Default: full bfloat16 weights (not 4/8-bit). "
            "Use e.g. mlx-community/Meta-Llama-3.1-8B-4bit for a smaller quantized run."
        ),
    )
    p.add_argument("--dataset", type=str, default="data/infinitebench_full.json")
#    p.add_argument("--dataset", type=str, default="data/longbench_v2_truncated_7k_smart.json")
    p.add_argument("--output", type=str, default="data/attention_vectors_infinitebench_full.json")
#    p.add_argument("--output", type=str, default="data/attention_vectors_long_bench_llama_8b.jsonl")
    p.add_argument("--max-length", type=int, default=131072)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--max-examples", type=int, default=0, help="0 means all remaining examples")
    p.add_argument("--extract-all-heads", action="store_true")
    p.add_argument(
        "--first-layer-idx",
        type=int,
        default=17,
        help="Transformer layer index for JSON `first_layer` (default: 17)",
    )
    p.add_argument(
        "--last-layer-idx",
        type=int,
        default=19,
        help="Transformer layer index for JSON `last_layer` (default: 19; use -1 for the final layer)",
    )
    p.add_argument(
        "--head-first-layer",
        type=int,
        default=21,
        help="Q head index for `first_layer` slice (default: 21)",
    )
    p.add_argument(
        "--head-last-layer",
        type=int,
        default=13,
        help="Q head index for `last_layer` slice (default: 13)",
    )
    p.add_argument(
        "--rope-stage",
        type=str,
        choices=["pre", "post"],
        default="pre",
        help=(
            "pre (default): Q/K after q_proj/k_proj, before mlx_lm RoPE. "
            "post: Q/K after the same RoPE as the model forward pass. "
            "V is always the linear v_proj output (RoPE does not apply to V)."
        ),
    )
    p.add_argument(
        "--tail-entropy",
        action="store_true",
        help=(
            "Compute running mean causal attention entropy over the last 10%% of query positions "
            "for every head (both layers). Slow; off by default."
        ),
    )
    p.add_argument(
        "--only-task",
        type=str,
        default="math_calc",
        metavar="TASK",
        help=(
            "Keep only examples with this InfiniteBench `task` (default: math_calc). "
            "LongBench: matches `sub_domain` only if it equals TASK exactly. "
            "Use --no-task-filter to process every example."
        ),
    )
    p.add_argument(
        "--no-task-filter",
        action="store_true",
        help="Ignore --only-task and process all examples (e.g. full LongBench JSON).",
    )
    p.add_argument(
        "--warmup",
        action="store_true",
        help=(
            "Run a throwaway warm-up extraction before the main loop. "
            "Disabled by default because it can destabilize very long-sequence runs."
        ),
    )
    p.add_argument(
        "--max-retries-on-unhealthy",
        type=int,
        default=2,
        help=(
            "Retries per example when extreme h norms are detected (default: 2). "
            "Set 0 to disable retries."
        ),
    )
    p.add_argument(
        "--print-h-norms",
        action="store_true",
        help="Print detailed residual-stream h norm diagnostics at tracked layers (off by default).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mx, load = _import_mlx()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("MLX QKV EXTRACTION")
    print(f"Model:   {args.model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_path}")
    print("=" * 78)

    print("Loading model/tokenizer...")
    model, tokenizer = load(args.model_name)
    _mn = args.model_name.lower()
    if any(x in _mn for x in ("4bit", "8bit", "3bit", "2bit", "-q4", "-q8", "gptq", "awq")):
        print("  Weight kind: quantized (from repo name).")
    else:
        print("  Weight kind: full precision (typ. bfloat16 MLX weights, not int4/int8).")

    cfg = get_model_cfg(model)
    post_rope = args.rope_stage == "post"
    rope_applied, rope_config, qkv_rope_stage = rope_json_fields(cfg, post_rope)

    num_heads = cfg_get(cfg, ["num_attention_heads", "n_heads"])
    num_kv_heads = cfg_get(cfg, ["num_key_value_heads", "n_kv_heads"], default=num_heads)
    hidden_size = cfg_get(cfg, ["hidden_size", "dim"])
    if num_heads is None or hidden_size is None:
        raise ValueError("Could not infer num_heads/hidden_size from model config.")
    group_size = num_heads // num_kv_heads

    layers = get_layers_root(model)
    head_dim = int(layers[0].self_attn.head_dim)
    num_layers = len(layers)
    first_layer_idx = args.first_layer_idx
    last_layer_idx = args.last_layer_idx if args.last_layer_idx >= 0 else (num_layers - 1)

    if not (0 <= first_layer_idx < num_layers):
        raise ValueError(f"--first-layer-idx must be in [0, {num_layers - 1}] (got {first_layer_idx})")
    if not (0 <= last_layer_idx < num_layers):
        raise ValueError(f"--last-layer-idx must be in [0, {num_layers - 1}] or -1 (got {args.last_layer_idx})")

    if not (0 <= args.head_first_layer < num_heads):
        raise ValueError(f"--head-first-layer must be in [0, {num_heads - 1}]")
    if not (0 <= args.head_last_layer < num_heads):
        raise ValueError(f"--head-last-layer must be in [0, {num_heads - 1}]")

    examples = load_examples(dataset_path)
    n_loaded = len(examples)
    only_task: Optional[str] = None if args.no_task_filter else args.only_task
    if only_task:
        examples = filter_examples_by_task(examples, only_task)
        print(
            f"Task filter --only-task {only_task!r}: {len(examples)} example(s) "
            f"(of {n_loaded} loaded). Use --no-task-filter for all examples."
        )
        if len(examples) == 0:
            print(
                "No examples match this task. For LongBench use --no-task-filter; "
                "for InfiniteBench check that the JSON includes a `task` field."
            )
            return

    end_idx = len(examples) if args.max_examples <= 0 else min(len(examples), args.start_idx + args.max_examples)

    num_in_this_run = end_idx - args.start_idx
    print(f"Examples: {len(examples)} total in dataset (after task filter), processing indices [{args.start_idx}:{end_idx})")
    print(f"This run: {num_in_this_run} example(s) to process.")
    if num_in_this_run <= 0:
        print("Nothing to process (check --start-idx / --max-examples). Exiting.")
        return
    print(f"Heads: Q={num_heads}, KV={num_kv_heads}, head_dim={head_dim}, group_size={group_size}")
    print(
        f"Q/K capture: --rope-stage {args.rope_stage} → "
        f"{'Q/K after mlx_lm RoPE' if post_rope else 'Q/K before RoPE (linear projections only)'}; "
        "V = v_proj output (never RoPE'd)."
    )
    print(
        "RoPE config from model: "
        f"theta={rope_config.get('rope_theta')}, "
        f"traditional={rope_config.get('rope_traditional')}, "
        f"max_position_embeddings={rope_config.get('max_position_embeddings')}, "
        f"rope_scaling={rope_config.get('rope_scaling')}"
    )
    if args.extract_all_heads:
        print(f"Mode: ALL HEADS | layers first={first_layer_idx} last={last_layer_idx}")
    else:
        print(
            f"Mode: single-head slices | first_layer=layer {first_layer_idx} Q head {args.head_first_layer} | "
            f"last_layer=layer {last_layer_idx} Q head {args.head_last_layer}"
        )
    if args.tail_entropy:
        print(
            f"Tail entropy: ON (slow) — mean causal attention entropy (nats) over the last "
            f"{100 * TAIL_ENTROPY_QUERY_FRACTION:.0f}% of query positions, per head; running mean "
            "over examples; all per-head values printed for both tracked layers."
        )
    else:
        print("Tail entropy: OFF (default; use --tail-entropy to enable).")

    if args.warmup:
        # Optional warm-up pass. Keep opt-in only for long contexts because this
        # extra run can itself trigger instability on some MLX versions/devices.
        warmup_ex = examples[args.start_idx]
        warmup_prompt = (
            f"Context: {warmup_ex.get('context', '')}\n\n"
            f"Question: {warmup_ex.get('question', '')}\n\nAnswer:"
        )
        warmup_ids = tokenize(tokenizer, warmup_prompt, max_length=args.max_length)
        warmup_seq_len = int(warmup_ids.shape[1])
        print(f"\nWarm-up pass (seq_len={warmup_seq_len})...", end=" ", flush=True)
        t_warmup = time.perf_counter()
        _wq, _wk, _wv, _, _, _, warmup_unhealthy = extract_qkv_first_last_llama_mlx(
            model,
            mx,
            warmup_ids,
            first_layer_idx,
            last_layer_idx,
            post_rope=post_rope,
            diagnose_h_norms=args.print_h_norms,
        )
        z_wk, _, p_wk = near_zero_norm_stats(_wk)
        z_wv, _, p_wv = near_zero_norm_stats(_wv)
        warmup_s = time.perf_counter() - t_warmup
        suffix = " (UNHEALTHY h detected)" if warmup_unhealthy else ""
        print(f"done in {warmup_s:.1f}s  (K zero={p_wk:.1f}%, V zero={p_wv:.1f}%){suffix}")
        del _wq, _wk, _wv, warmup_ids, z_wk, z_wv
        gc.collect()
    else:
        print("\nWarm-up pass: skipped (default). Use --warmup to enable.")

    n_entropy_examples = 0
    entropy_sum_first = np.zeros(num_heads, dtype=np.float64)
    entropy_sum_last = np.zeros(num_heads, dtype=np.float64)

    total_elapsed_s = 0.0  # sum of per-example wall times (for ETA)

    with output_path.open("w") as out_f:
        for run_pos, idx in enumerate(range(args.start_idx, end_idx), start=1):
            ex = examples[idx]
            domain = ex.get("domain") or ex.get("task") or "unknown"
            print(f"\n[run {run_pos}/{num_in_this_run} | dataset {idx + 1}/{len(examples)}] Processing: {domain}")
            t_example_start = time.perf_counter()

            prompt = (
                f"Context: {ex.get('context', '')}\n\nQuestion: {ex.get('question', '')}\n\nAnswer:"
            )
            input_ids_np = tokenize(tokenizer, prompt, max_length=args.max_length)
            seq_len = int(input_ids_np.shape[1])
            print(f"  Sequence length: {seq_len}")

            retries = max(0, int(args.max_retries_on_unhealthy))
            attempt = 0
            while True:
                attempt += 1
                (
                    first_q,
                    first_k,
                    first_v,
                    last_q,
                    last_k,
                    last_v,
                    unhealthy_detected,
                ) = extract_qkv_first_last_llama_mlx(
                    model,
                    mx,
                    input_ids_np,
                    first_layer_idx,
                    last_layer_idx,
                    post_rope=post_rope,
                    diagnose_h_norms=args.print_h_norms,
                )
                if not unhealthy_detected or attempt > retries:
                    if unhealthy_detected:
                        print(
                            f"  Warning: unhealthy h norms persisted after {attempt} attempt(s). "
                            "Writing outputs anyway."
                        )
                    break
                print(f"  Unhealthy h norms detected; retrying extraction ({attempt}/{retries})...")
                gc.collect()
            # Per-example sanity check for potential lazy/memory artifacts:
            # report percentage of vectors with (near-)zero L2 norm.
            z_fq, n_fq, p_fq = near_zero_norm_stats(first_q)
            z_fk, n_fk, p_fk = near_zero_norm_stats(first_k)
            z_fv, n_fv, p_fv = near_zero_norm_stats(first_v)
            z_lq, n_lq, p_lq = near_zero_norm_stats(last_q)
            z_lk, n_lk, p_lk = near_zero_norm_stats(last_k)
            z_lv, n_lv, p_lv = near_zero_norm_stats(last_v)
            z_tot = z_fq + z_fk + z_fv + z_lq + z_lk + z_lv
            n_tot = n_fq + n_fk + n_fv + n_lq + n_lk + n_lv
            p_tot = 100.0 * z_tot / n_tot if n_tot > 0 else 0.0
            print(f"  Near-zero norm check (eps={NEAR_ZERO_NORM_EPS:g}):")
            print(f"    first_layer  Q: {p_fq:6.3f}% ({z_fq}/{n_fq}) | K: {p_fk:6.3f}% ({z_fk}/{n_fk}) | V: {p_fv:6.3f}% ({z_fv}/{n_fv})")
            print(f"    last_layer   Q: {p_lq:6.3f}% ({z_lq}/{n_lq}) | K: {p_lk:6.3f}% ({z_lk}/{n_lk}) | V: {p_lv:6.3f}% ({z_lv}/{n_lv})")
            print(f"    overall QKV near-zero: {p_tot:6.3f}% ({z_tot}/{n_tot})")

            if args.tail_entropy:
                try:
                    ent_first = compute_mean_tail_attention_entropy_per_head(
                        first_q, first_k, group_size, last_query_fraction=TAIL_ENTROPY_QUERY_FRACTION
                    )
                    ent_last = compute_mean_tail_attention_entropy_per_head(
                        last_q, last_k, group_size, last_query_fraction=TAIL_ENTROPY_QUERY_FRACTION
                    )
                except Exception as exc:
                    print(f"  Warning: tail entropy aggregation failed: {exc}")
                else:
                    entropy_sum_first += ent_first
                    entropy_sum_last += ent_last
                    n_entropy_examples += 1
                    avg_first = entropy_sum_first / n_entropy_examples
                    avg_last = entropy_sum_last / n_entropy_examples
                    print(
                        f"  Tail attention entropy — last {100 * TAIL_ENTROPY_QUERY_FRACTION:.0f}% of query positions, "
                        f"causal softmax; running mean over {n_entropy_examples} example(s):"
                    )
                    print(
                        format_per_head_entropy_block(
                            f"Layer {first_layer_idx} (`first_layer`)",
                            avg_first,
                        )
                    )
                    print(
                        format_per_head_entropy_block(
                            f"Layer {last_layer_idx} (`last_layer`)",
                            avg_last,
                        )
                    )

            if args.extract_all_heads:
                result = build_all_heads_result(
                    ex, idx, seq_len, args.model_name, num_layers, num_heads, num_kv_heads,
                    head_dim, group_size, first_layer_idx, last_layer_idx,
                    first_q, first_k, first_v, last_q, last_k, last_v,
                    rope_applied, rope_config, qkv_rope_stage,
                )
                print(f"  Extracted ALL heads - Q: {first_q.shape}, K: {first_k.shape}")
            else:
                result = build_single_head_result(
                    ex, idx, seq_len, args.model_name, num_layers, num_heads, num_kv_heads,
                    head_dim, group_size, first_layer_idx, last_layer_idx,
                    args.head_first_layer, args.head_last_layer,
                    first_q, first_k, first_v, last_q, last_k, last_v,
                    rope_applied, rope_config, qkv_rope_stage,
                )
                print(
                    f"  Extracted L{first_layer_idx} Q{args.head_first_layer} & L{last_layer_idx} Q{args.head_last_layer} "
                    f"- first_layer Q: {result['first_layer']['Q_shape']}, last_layer Q: {result['last_layer']['Q_shape']}"
                )

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            elapsed_s = time.perf_counter() - t_example_start
            total_elapsed_s += elapsed_s
            pct = 100.0 * run_pos / num_in_this_run
            remaining_n = num_in_this_run - run_pos
            time_line = f"  Time for last example: {elapsed_s:.2f}s"
            if remaining_n > 0:
                avg_s = total_elapsed_s / run_pos
                eta_s = avg_s * remaining_n
                if eta_s >= 3600:
                    eta_h, rem = divmod(int(round(eta_s)), 3600)
                    eta_m, eta_sec = divmod(rem, 60)
                    eta_str = f"{eta_h}h {eta_m}m {eta_sec}s"
                elif eta_s >= 60:
                    eta_m, eta_sec = divmod(int(round(eta_s)), 60)
                    eta_str = f"{eta_m}m {eta_sec}s"
                else:
                    eta_str = f"{eta_s:.0f}s"
                time_line += (
                    f"  |  Est. remaining: ~{eta_str}  "
                    f"(avg {avg_s:.2f}s/example × {remaining_n} left)"
                )
            else:
                time_line += "  |  Est. remaining: 0s (run complete)"
            print(time_line)
            print(f"  >>> Progress: {run_pos}/{num_in_this_run} examples completed this run ({pct:.1f}%)")

    print("\n" + "=" * 78)
    print("EXTRACTION COMPLETE")
    print("=" * 78)
    print(f"Processed: {num_in_this_run}/{num_in_this_run} examples in this run.")
    print(f"Output: {output_path}")
    if args.tail_entropy and n_entropy_examples > 0:
        avg_first = entropy_sum_first / n_entropy_examples
        avg_last = entropy_sum_last / n_entropy_examples
        print("\nTail entropy summary (running mean over all examples in this run):")
        print(
            format_per_head_entropy_block(
                f"Layer {first_layer_idx} (`first_layer`)",
                avg_first,
            )
        )
        print(
            format_per_head_entropy_block(
                f"Layer {last_layer_idx} (`last_layer`)",
                avg_last,
            )
        )


if __name__ == "__main__":
    main()
