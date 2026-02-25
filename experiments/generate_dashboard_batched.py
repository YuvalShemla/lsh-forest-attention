#!/usr/bin/env python3
"""
Attention Dashboard Generator - Batched Processing

Processes examples in batches of 5 to avoid memory issues.
Saves intermediate results, then aggregates at the end.

Usage:
    python3 generate_dashboard_batched.py
    
This will process all batches and generate the final HTML automatically.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_PATH = '../data/attention_vectors_updated_long.jsonl'
OUTPUT_PATH = '../results/professional_attention_dashboard.html'
BATCH_DIR = Path('../results/dashboard_batches')
NUM_EXAMPLES_TOTAL = 100  # Total examples to process
BATCH_SIZE = 5  # Process 5 examples at a time
NUM_QUERIES_PER_EXAMPLE = 1000  # Last 1000 queries per example
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
TOP_K_FOR_CORR = 100
SEED = 42

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ---------------------------------------------------------------------------
# BATCH ANALYSIS
# ---------------------------------------------------------------------------
def analyze_batch(examples, layer_name, num_queries):
    """Analyze one batch of examples. Returns aggregated stats."""
    
    print(f"    Analyzing {len(examples)} examples for {layer_name}...")
    
    # Accumulators for this batch
    batch_data = {
        'concentration_curves': [],  # [n_queries_in_batch, 100]
        'q_norms': [],
        'k_norms': [],
        'v_norms': [],
        'key_query_dists': [],
        'key_query_cos': [],
        'top_k_masses': {10: [], 50: [], 100: [], 200: []},
        'logit_means': [],
        'logit_stds': [],
        'logit_ranges': [],
        'kv_corr_l2': [],
        'kv_corr_cos': [],
        'key_pw_dists_samples': [],
        'val_pw_dists_samples': [],
        'key_pw_cos_samples': [],
        'val_pw_cos_samples': [],
        'per_example_stats': {
            'mean_q_norm': [],
            'mean_k_norm': [],
            'mean_v_norm': [],
            'mean_conc_at_10pct': [],
            'mean_top10_mass': [],
            'mean_kv_corr_l2': [],
        },
        'num_queries': 0,
    }
    
    for ex_idx, example in enumerate(examples):
        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        K = np.array(example[layer_name]['K'], dtype=np.float32)
        V = np.array(example[layer_name]['V'], dtype=np.float32)
        seq_len = Q.shape[0]
        
        # Norms (all positions)
        q_norms_ex = np.linalg.norm(Q, axis=1)
        k_norms_ex = np.linalg.norm(K, axis=1)
        v_norms_ex = np.linalg.norm(V, axis=1)
        batch_data['q_norms'].extend(q_norms_ex.tolist())
        batch_data['k_norms'].extend(k_norms_ex.tolist())
        batch_data['v_norms'].extend(v_norms_ex.tolist())
        batch_data['per_example_stats']['mean_q_norm'].append(float(np.mean(q_norms_ex)))
        batch_data['per_example_stats']['mean_k_norm'].append(float(np.mean(k_norms_ex)))
        batch_data['per_example_stats']['mean_v_norm'].append(float(np.mean(v_norms_ex)))
        
        # Sample queries
        actual_num_queries = min(num_queries, seq_len - 100)
        query_positions = list(range(seq_len - actual_num_queries, seq_len))
        batch_data['num_queries'] += len(query_positions)
        
        example_conc_curves = []
        example_top10_masses = []
        example_kv_corrs = []
        
        for query_pos in query_positions:
            q = Q[query_pos]
            valid_keys = K[:query_pos + 1]
            valid_values = V[:query_pos + 1]
            n_keys = len(valid_keys)
            
            # Logits and weights
            logits = (q @ valid_keys.T) / np.sqrt(HEAD_DIM)
            weights = softmax(logits)
            
            # Key-Query distances (subsample)
            sample_size = min(200, n_keys)
            sample_idx = np.random.choice(n_keys, sample_size, replace=False)
            kq_dists = np.linalg.norm(valid_keys[sample_idx] - q[None, :], axis=1)
            batch_data['key_query_dists'].extend(kq_dists.tolist())
            
            # Cosine similarity
            q_norm = q / (np.linalg.norm(q) + 1e-8)
            k_norm = valid_keys[sample_idx] / (np.linalg.norm(valid_keys[sample_idx], axis=1, keepdims=True) + 1e-8)
            cos = k_norm @ q_norm
            batch_data['key_query_cos'].extend(cos.tolist())
            
            # Concentration curve
            sorted_w = np.sort(weights)[::-1]
            cumsum = np.cumsum(sorted_w) * 100
            pct_points = np.linspace(0, 1, 101)[1:]
            curve = np.interp(pct_points * n_keys, np.arange(1, n_keys + 1), cumsum)
            batch_data['concentration_curves'].append(curve.tolist())
            example_conc_curves.append(curve)
            
            # Top-K masses
            for k in batch_data['top_k_masses']:
                if n_keys >= k:
                    mass = float(sorted_w[:k].sum() * 100)
                    batch_data['top_k_masses'][k].append(mass)
                    if k == 10:
                        example_top10_masses.append(mass)
            
            # Logit stats
            batch_data['logit_means'].append(float(logits.mean()))
            batch_data['logit_stds'].append(float(logits.std()))
            batch_data['logit_ranges'].append(float(logits.max() - logits.min()))
            
            # Key-Value correlation (for top-100)
            if n_keys >= 100:
                top_idx = np.argsort(logits)[-100:]
                k_top = valid_keys[top_idx]
                v_top = valid_values[top_idx]
                
                # L2 pairwise
                k_pw = np.linalg.norm(k_top[:, None, :] - k_top[None, :, :], axis=2)
                v_pw = np.linalg.norm(v_top[:, None, :] - v_top[None, :, :], axis=2)
                
                # Cosine pairwise
                k_top_norm = k_top / (np.linalg.norm(k_top, axis=1, keepdims=True) + 1e-8)
                v_top_norm = v_top / (np.linalg.norm(v_top, axis=1, keepdims=True) + 1e-8)
                k_cos_sim = k_top_norm @ k_top_norm.T
                v_cos_sim = v_top_norm @ v_top_norm.T
                k_cos_dist = 1.0 - k_cos_sim
                v_cos_dist = 1.0 - v_cos_sim
                
                # Upper triangle
                iu = np.triu_indices(100, k=1)
                k_dists = k_pw[iu]
                v_dists = v_pw[iu]
                k_cos_dists = k_cos_dist[iu]
                v_cos_dists = v_cos_dist[iu]
                
                # L2 correlation
                if k_dists.std() > 1e-6 and v_dists.std() > 1e-6:
                    corr_l2 = np.corrcoef(k_dists, v_dists)[0, 1]
                    batch_data['kv_corr_l2'].append(float(corr_l2))
                    example_kv_corrs.append(float(corr_l2))
                    
                    # Sample pairs
                    if len(batch_data['key_pw_dists_samples']) < 2000:
                        idx_sample = np.random.choice(len(k_dists), min(20, len(k_dists)), replace=False)
                        batch_data['key_pw_dists_samples'].extend(k_dists[idx_sample].tolist())
                        batch_data['val_pw_dists_samples'].extend(v_dists[idx_sample].tolist())
                
                # Cosine correlation
                if k_cos_dists.std() > 1e-6 and v_cos_dists.std() > 1e-6:
                    corr_cos = np.corrcoef(k_cos_dists, v_cos_dists)[0, 1]
                    batch_data['kv_corr_cos'].append(float(corr_cos))
                    
                    # Sample pairs
                    if len(batch_data['key_pw_cos_samples']) < 2000:
                        idx_sample = np.random.choice(len(k_cos_dists), min(20, len(k_cos_dists)), replace=False)
                        batch_data['key_pw_cos_samples'].extend(k_cos_dists[idx_sample].tolist())
                        batch_data['val_pw_cos_samples'].extend(v_cos_dists[idx_sample].tolist())
        
        # Per-example stats
        if len(example_conc_curves) > 0:
            mean_curve = np.mean(example_conc_curves, axis=0)
            batch_data['per_example_stats']['mean_conc_at_10pct'].append(float(mean_curve[9]))
        if len(example_top10_masses) > 0:
            batch_data['per_example_stats']['mean_top10_mass'].append(float(np.mean(example_top10_masses)))
        if len(example_kv_corrs) > 0:
            batch_data['per_example_stats']['mean_kv_corr_l2'].append(float(np.mean(example_kv_corrs)))
    
    print(f"      ✓ {batch_data['num_queries']} queries analyzed")
    return batch_data

# ---------------------------------------------------------------------------
# AGGREGATE BATCHES
# ---------------------------------------------------------------------------
def aggregate_batches(batch_files):
    """Combine multiple batch result files into final aggregated stats."""
    
    print(f"\nAggregating {len(batch_files)} batches...")
    
    aggregated = {
        'concentration_curves': [],
        'q_norms': [],
        'k_norms': [],
        'v_norms': [],
        'key_query_dists': [],
        'key_query_cos': [],
        'top_k_masses': {10: [], 50: [], 100: [], 200: []},
        'logit_means': [],
        'logit_stds': [],
        'logit_ranges': [],
        'kv_corr_l2': [],
        'kv_corr_cos': [],
        'key_pw_dists_samples': [],
        'val_pw_dists_samples': [],
        'key_pw_cos_samples': [],
        'val_pw_cos_samples': [],
        'per_example_stats': {
            'mean_q_norm': [],
            'mean_k_norm': [],
            'mean_v_norm': [],
            'mean_conc_at_10pct': [],
            'mean_top10_mass': [],
            'mean_kv_corr_l2': [],
        },
        'num_queries': 0,
    }
    
    for batch_file in batch_files:
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f)
        
        # Extend lists
        for key in ['q_norms', 'k_norms', 'v_norms', 'key_query_dists', 'key_query_cos',
                    'logit_means', 'logit_stds', 'logit_ranges', 
                    'kv_corr_l2', 'kv_corr_cos',
                    'key_pw_dists_samples', 'val_pw_dists_samples',
                    'key_pw_cos_samples', 'val_pw_cos_samples',
                    'concentration_curves']:
            if key in batch:
                aggregated[key].extend(batch[key])
        
        # Top-K masses
        for k in aggregated['top_k_masses']:
            aggregated['top_k_masses'][k].extend(batch['top_k_masses'][k])
        
        # Per-example stats
        for key in aggregated['per_example_stats']:
            aggregated['per_example_stats'][key].extend(batch['per_example_stats'][key])
        
        aggregated['num_queries'] += batch['num_queries']
    
    # Convert concentration curves to array for percentile computation
    conc_curves_arr = np.array(aggregated['concentration_curves'])
    
    # Compute final statistics
    final = {
        'conc_x': np.linspace(0, 100, 101)[1:].tolist(),
        'conc_mean': np.mean(conc_curves_arr, axis=0).tolist(),
        'conc_p10': np.percentile(conc_curves_arr, 10, axis=0).tolist(),
        'conc_p50': np.percentile(conc_curves_arr, 50, axis=0).tolist(),
        'conc_p90': np.percentile(conc_curves_arr, 90, axis=0).tolist(),
        'conc_p99': np.percentile(conc_curves_arr, 99, axis=0).tolist(),
        'q_norms': aggregated['q_norms'],
        'k_norms': aggregated['k_norms'],
        'v_norms': aggregated['v_norms'],
        'key_query_dists': aggregated['key_query_dists'],
        'key_query_cos': aggregated['key_query_cos'],
        'top_k_masses': aggregated['top_k_masses'],
        'logit_means': aggregated['logit_means'],
        'logit_stds': aggregated['logit_stds'],
        'logit_ranges': aggregated['logit_ranges'],
        'kv_corr_l2': aggregated['kv_corr_l2'],
        'kv_corr_cos': aggregated['kv_corr_cos'],
        'key_pw_dists': aggregated['key_pw_dists_samples'],
        'val_pw_dists': aggregated['val_pw_dists_samples'],
        'key_pw_cos': aggregated['key_pw_cos_samples'],
        'val_pw_cos': aggregated['val_pw_cos_samples'],
        'per_example_stats': aggregated['per_example_stats'],
        'num_queries': aggregated['num_queries'],
    }
    
    print(f"  ✓ Aggregated {aggregated['num_queries']} queries")
    return final

# ---------------------------------------------------------------------------
# MAIN - BATCH PROCESSING
# ---------------------------------------------------------------------------
def main():
    print("="*70)
    print("ATTENTION DASHBOARD - BATCHED PROCESSING")
    print("="*70)
    print(f"Total examples: {NUM_EXAMPLES_TOTAL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Queries per example: {NUM_QUERIES_PER_EXAMPLE}")
    print(f"Total batches: {int(np.ceil(NUM_EXAMPLES_TOTAL / BATCH_SIZE))}")
    print()
    
    # Create batch directory
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    # First pass: count total examples
    print(f"Counting examples in: {DATA_PATH}")
    with open(DATA_PATH, 'r') as f:
        total_in_file = sum(1 for _ in f)
    print(f"✓ Found {total_in_file} examples")
    
    # Select random indices
    selected_indices = sorted(np.random.choice(total_in_file, NUM_EXAMPLES_TOTAL, replace=False).tolist())
    print(f"✓ Selected {NUM_EXAMPLES_TOTAL} random indices")
    
    # Second pass: load only selected examples
    print(f"Loading selected examples...")
    selected_indices_set = set(selected_indices)
    selected_examples = []
    with open(DATA_PATH, 'r') as f:
        for idx, line in enumerate(f):
            if idx in selected_indices_set:
                selected_examples.append(json.loads(line))
                if len(selected_examples) % 10 == 0:
                    print(f"  Loaded {len(selected_examples)}/{NUM_EXAMPLES_TOTAL}...", end='\r')
            if len(selected_examples) >= NUM_EXAMPLES_TOTAL:
                break
    print(f"\n✓ Loaded {len(selected_examples)} examples")
    
    # Process in batches
    num_batches = int(np.ceil(NUM_EXAMPLES_TOTAL / BATCH_SIZE))
    
    for layer_name in LAYERS:
        print(f"\n{'='*70}")
        print(f"Processing {layer_name}")
        print('='*70)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, NUM_EXAMPLES_TOTAL)
            batch_examples = selected_examples[start_idx:end_idx]
            
            print(f"\n  Batch {batch_idx + 1}/{num_batches} (examples {start_idx+1}-{end_idx})")
            
            batch_result = analyze_batch(batch_examples, layer_name, NUM_QUERIES_PER_EXAMPLE)
            
            # Save batch result
            batch_file = BATCH_DIR / f'{layer_name}_batch_{batch_idx:03d}.pkl'
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_result, f)
            print(f"    ✓ Saved: {batch_file}")
    
    print(f"\n{'='*70}")
    print("AGGREGATING BATCHES")
    print('='*70)
    
    # Aggregate each layer
    final_data = {}
    for layer_name in LAYERS:
        print(f"\n{layer_name}:")
        batch_files = sorted(BATCH_DIR.glob(f'{layer_name}_batch_*.pkl'))
        final_data[layer_name] = aggregate_batches(batch_files)
    
    # Save aggregated data
    aggregated_file = BATCH_DIR / 'aggregated_data.pkl'
    with open(aggregated_file, 'wb') as f:
        pickle.dump(final_data, f)
    print(f"\n✓ Saved aggregated data: {aggregated_file}")
    
    print(f"\n{'='*70}")
    print("GENERATING DASHBOARD")
    print('='*70)
    
    # Now generate visualizations and HTML
    from generate_dashboard_visualize import generate_html_dashboard
    
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'num_examples': NUM_EXAMPLES_TOTAL,
        'num_queries': NUM_QUERIES_PER_EXAMPLE,
        'total_queries': final_data['first_layer']['num_queries'] + final_data['last_layer']['num_queries'],
        'head_dim': HEAD_DIM,
    }
    
    output_path = generate_html_dashboard(final_data, metadata, OUTPUT_PATH)
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE")
    print('='*70)
    print(f"Dashboard: {output_path}")
    print(f"Open with: open {output_path}")

if __name__ == "__main__":
    main()
