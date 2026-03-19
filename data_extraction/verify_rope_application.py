#!/usr/bin/env python3
"""
Verify RoPE was applied correctly by comparing original vs RoPE-applied files.

Checks:
1. Data structure integrity (shapes, metadata, fields)
2. V vectors are EXACTLY unchanged (RoPE should not touch V)
3. Q/K vectors are changed (RoPE should rotate them)
4. Per-position norm preservation (RoPE is a rotation)
5. K-V correlation analysis: 100 random positions, before vs after RoPE
6. Cross-check: re-apply RoPE from scratch and compare to stored result
"""

import json
import numpy as np
from pathlib import Path

np.random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"
ORIGINAL_FILE = DATA_DIR / "attention_vectors_long_bench_llama_8b.jsonl"
ROPE_FILE = DATA_DIR / "attention_vectors_llama_8b_with_rope.jsonl"

ROPE_THETA = 500000.0
HEAD_DIM = 128
NUM_VERIFY_EXAMPLES = 3
NUM_CORR_POSITIONS = 100


def compute_rope_cache(seq_len):
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
    position_ids = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(position_ids, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(q, k, cos, sin):
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)


def cosine_similarity_rowwise(A, B):
    """Cosine similarity between corresponding rows of A and B."""
    dot = np.sum(A * B, axis=-1)
    nA = np.linalg.norm(A, axis=-1)
    nB = np.linalg.norm(B, axis=-1)
    return dot / (nA * nB + 1e-10)


def main():
    print("=" * 70)
    print("VERIFICATION: RoPE application correctness")
    print("=" * 70)

    if not ORIGINAL_FILE.exists():
        print(f"ERROR: Original file not found: {ORIGINAL_FILE}")
        return
    if not ROPE_FILE.exists():
        print(f"ERROR: RoPE file not found: {ROPE_FILE}")
        return

    with open(ORIGINAL_FILE, 'r') as f_orig, open(ROPE_FILE, 'r') as f_rope:
        for ex_idx in range(NUM_VERIFY_EXAMPLES):
            orig_line = f_orig.readline()
            rope_line = f_rope.readline()
            orig = json.loads(orig_line)
            rope = json.loads(rope_line)

            print(f"\n{'─'*70}")
            print(f"Example {ex_idx}: id={orig['example_id']}, "
                  f"seq_len={orig['sequence_length']}, domain={orig['domain']}")
            print(f"{'─'*70}")

            # ── CHECK 1: Metadata consistency ──
            print("\n[CHECK 1] Metadata consistency")
            assert orig['example_id'] == rope['example_id'], "Example ID mismatch!"
            assert orig['sequence_length'] == rope['sequence_length'], "Seq len mismatch!"
            assert orig['domain'] == rope['domain'], "Domain mismatch!"
            assert rope.get('rope_applied') == True, "Missing rope_applied flag!"
            assert rope.get('rope_config', {}).get('rope_theta') == ROPE_THETA, "Wrong theta!"
            print("  PASS: IDs, seq_len, domain, rope_applied flag, theta all match")

            seq_len = orig['sequence_length']

            for layer_key in ['first_layer', 'last_layer']:
                print(f"\n  === {layer_key} (layer {orig[layer_key].get('layer_idx', '?')}) ===")

                Q_orig = np.array(orig[layer_key]['Q'], dtype=np.float32)
                K_orig = np.array(orig[layer_key]['K'], dtype=np.float32)
                V_orig = np.array(orig[layer_key]['V'], dtype=np.float32)

                Q_rope = np.array(rope[layer_key]['Q'], dtype=np.float32)
                K_rope = np.array(rope[layer_key]['K'], dtype=np.float32)
                V_rope = np.array(rope[layer_key]['V'], dtype=np.float32)

                # ── CHECK 2: Shape consistency ──
                print(f"\n  [CHECK 2] Shapes")
                print(f"    Q: orig={Q_orig.shape}, rope={Q_rope.shape}")
                print(f"    K: orig={K_orig.shape}, rope={K_rope.shape}")
                print(f"    V: orig={V_orig.shape}, rope={V_rope.shape}")
                assert Q_orig.shape == (seq_len, HEAD_DIM), f"Q shape wrong: {Q_orig.shape}"
                assert K_orig.shape == (seq_len, HEAD_DIM), f"K shape wrong: {K_orig.shape}"
                assert V_orig.shape == (seq_len, HEAD_DIM), f"V shape wrong: {V_orig.shape}"
                assert Q_rope.shape == Q_orig.shape
                assert K_rope.shape == K_orig.shape
                assert V_rope.shape == V_orig.shape
                print("    PASS: All shapes are (seq_len, 128) and match")

                # ── CHECK 3: V unchanged ──
                print(f"\n  [CHECK 3] V vectors unchanged")
                v_diff = np.max(np.abs(V_rope - V_orig))
                print(f"    Max |V_rope - V_orig|: {v_diff}")
                assert v_diff == 0.0, f"V was modified! max diff = {v_diff}"
                print("    PASS: V is bit-for-bit identical")

                # ── CHECK 4: Q and K are changed ──
                print(f"\n  [CHECK 4] Q and K are changed by RoPE")
                q_diff = np.linalg.norm(Q_rope - Q_orig) / np.linalg.norm(Q_orig)
                k_diff = np.linalg.norm(K_rope - K_orig) / np.linalg.norm(K_orig)
                print(f"    Relative L2 change: Q={q_diff:.4f}, K={k_diff:.4f}")
                assert q_diff > 0.1, "Q barely changed — RoPE might not have been applied"
                assert k_diff > 0.1, "K barely changed — RoPE might not have been applied"
                print("    PASS: Both Q and K are substantially changed")

                # ── CHECK 5: Per-position norm preservation ──
                print(f"\n  [CHECK 5] Per-position norm preservation (rotation check)")
                q_norms_orig = np.linalg.norm(Q_orig, axis=-1)
                q_norms_rope = np.linalg.norm(Q_rope, axis=-1)
                k_norms_orig = np.linalg.norm(K_orig, axis=-1)
                k_norms_rope = np.linalg.norm(K_rope, axis=-1)
                q_norm_err = np.max(np.abs(q_norms_rope - q_norms_orig) / (q_norms_orig + 1e-10))
                k_norm_err = np.max(np.abs(k_norms_rope - k_norms_orig) / (k_norms_orig + 1e-10))
                print(f"    Max relative norm error: Q={q_norm_err:.8f}, K={k_norm_err:.8f}")
                assert q_norm_err < 1e-5, f"Q norms changed too much: {q_norm_err}"
                assert k_norm_err < 1e-5, f"K norms changed too much: {k_norm_err}"
                print("    PASS: Norms preserved (confirmed RoPE is a rotation)")

                # ── CHECK 6: Re-derive RoPE and compare ──
                print(f"\n  [CHECK 6] Re-derive RoPE from scratch and compare")
                cos, sin = compute_rope_cache(seq_len)
                Q_recomputed, K_recomputed = apply_rope(Q_orig, K_orig, cos, sin)
                q_recomp_err = np.max(np.abs(Q_rope - Q_recomputed))
                k_recomp_err = np.max(np.abs(K_rope - K_recomputed))
                print(f"    Max |Q_stored - Q_recomputed|: {q_recomp_err:.10f}")
                print(f"    Max |K_stored - K_recomputed|: {k_recomp_err:.10f}")
                assert q_recomp_err < 1e-4, f"Q mismatch with recomputation: {q_recomp_err}"
                assert k_recomp_err < 1e-4, f"K mismatch with recomputation: {k_recomp_err}"
                print("    PASS: Stored values match independent recomputation")

                # ── CHECK 7: K-V correlation before vs after RoPE ──
                print(f"\n  [CHECK 7] K-V correlation at {NUM_CORR_POSITIONS} random positions")

                positions = np.random.choice(seq_len, size=NUM_CORR_POSITIONS, replace=False)
                positions.sort()

                K_sample_orig = K_orig[positions]   # [100, 128]
                K_sample_rope = K_rope[positions]   # [100, 128]
                V_sample = V_orig[positions]         # [100, 128]

                # 7a: Cosine similarity between K and V at each position
                cos_sim_before = cosine_similarity_rowwise(K_sample_orig, V_sample)
                cos_sim_after = cosine_similarity_rowwise(K_sample_rope, V_sample)

                print(f"    Cosine similarity K[i] vs V[i] (per-position):")
                print(f"      Before RoPE: mean={cos_sim_before.mean():.4f}, "
                      f"std={cos_sim_before.std():.4f}, "
                      f"min={cos_sim_before.min():.4f}, max={cos_sim_before.max():.4f}")
                print(f"      After  RoPE: mean={cos_sim_after.mean():.4f}, "
                      f"std={cos_sim_after.std():.4f}, "
                      f"min={cos_sim_after.min():.4f}, max={cos_sim_after.max():.4f}")

                # 7b: Pearson correlation across the 100 positions (norm-based)
                k_norms_sample_orig = np.linalg.norm(K_sample_orig, axis=-1)
                k_norms_sample_rope = np.linalg.norm(K_sample_rope, axis=-1)
                v_norms_sample = np.linalg.norm(V_sample, axis=-1)

                corr_norms_before = np.corrcoef(k_norms_sample_orig, v_norms_sample)[0, 1]
                corr_norms_after = np.corrcoef(k_norms_sample_rope, v_norms_sample)[0, 1]
                print(f"\n    Pearson correlation of ||K|| vs ||V|| across positions:")
                print(f"      Before RoPE: r={corr_norms_before:.4f}")
                print(f"      After  RoPE: r={corr_norms_after:.4f}")
                print(f"      (Should be identical — RoPE preserves norms)")

                # 7c: Flatten to check global correlation pattern
                K_flat_orig = K_sample_orig.flatten()
                K_flat_rope = K_sample_rope.flatten()
                V_flat = V_sample.flatten()

                corr_flat_before = np.corrcoef(K_flat_orig, V_flat)[0, 1]
                corr_flat_after = np.corrcoef(K_flat_rope, V_flat)[0, 1]
                print(f"\n    Pearson correlation K vs V (flattened {NUM_CORR_POSITIONS}x{HEAD_DIM} elements):")
                print(f"      Before RoPE: r={corr_flat_before:.4f}")
                print(f"      After  RoPE: r={corr_flat_after:.4f}")

                # 7d: Per-dimension correlation (average across 128 dims)
                dim_corrs_before = []
                dim_corrs_after = []
                for d in range(HEAD_DIM):
                    r_before = np.corrcoef(K_sample_orig[:, d], V_sample[:, d])[0, 1]
                    r_after = np.corrcoef(K_sample_rope[:, d], V_sample[:, d])[0, 1]
                    if not np.isnan(r_before):
                        dim_corrs_before.append(r_before)
                    if not np.isnan(r_after):
                        dim_corrs_after.append(r_after)
                print(f"\n    Per-dimension correlation K[:,d] vs V[:,d] (mean over {HEAD_DIM} dims):")
                print(f"      Before RoPE: mean_r={np.mean(dim_corrs_before):.4f}, "
                      f"std={np.std(dim_corrs_before):.4f}")
                print(f"      After  RoPE: mean_r={np.mean(dim_corrs_after):.4f}, "
                      f"std={np.std(dim_corrs_after):.4f}")

                # 7e: Position-dependent effect — early vs late positions
                early_pos = positions[positions < seq_len // 4][:25]
                late_pos = positions[positions >= 3 * seq_len // 4]
                if len(late_pos) > 25:
                    late_pos = late_pos[:25]

                if len(early_pos) > 5 and len(late_pos) > 5:
                    cos_early_before = cosine_similarity_rowwise(
                        K_orig[early_pos], V_orig[early_pos])
                    cos_early_after = cosine_similarity_rowwise(
                        K_rope[early_pos], V_orig[early_pos])
                    cos_late_before = cosine_similarity_rowwise(
                        K_orig[late_pos], V_orig[late_pos])
                    cos_late_after = cosine_similarity_rowwise(
                        K_rope[late_pos], V_orig[late_pos])
                    print(f"\n    Position-dependent K-V cosine similarity:")
                    print(f"      Early (pos<{seq_len//4}, n={len(early_pos)}):")
                    print(f"        Before: {cos_early_before.mean():.4f} +/- {cos_early_before.std():.4f}")
                    print(f"        After:  {cos_early_after.mean():.4f} +/- {cos_early_after.std():.4f}")
                    print(f"      Late  (pos>={3*seq_len//4}, n={len(late_pos)}):")
                    print(f"        Before: {cos_late_before.mean():.4f} +/- {cos_late_before.std():.4f}")
                    print(f"        After:  {cos_late_after.mean():.4f} +/- {cos_late_after.std():.4f}")

                # ── CHECK 8: K-K self-similarity changes ──
                print(f"\n  [CHECK 8] K-K similarity structure change")
                sample_20 = positions[:20]
                K20_orig = K_orig[sample_20]
                K20_rope = K_rope[sample_20]
                sim_orig = K20_orig @ K20_orig.T
                sim_rope = K20_rope @ K20_rope.T
                diag_mask = ~np.eye(20, dtype=bool)
                off_diag_orig = sim_orig[diag_mask]
                off_diag_rope = sim_rope[diag_mask]
                print(f"    Off-diagonal K@K^T (20x20 block):")
                print(f"      Before: mean={off_diag_orig.mean():.2f}, std={off_diag_orig.std():.2f}")
                print(f"      After:  mean={off_diag_rope.mean():.2f}, std={off_diag_rope.std():.2f}")
                corr_kk = np.corrcoef(off_diag_orig, off_diag_rope)[0, 1]
                print(f"      Correlation of off-diagonal elements: r={corr_kk:.4f}")

    print(f"\n{'='*70}")
    print("ALL CHECKS PASSED")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
