#!/usr/bin/env python3
"""
Professional Attention Space Dashboard

Generates publication-quality visualizations of attention geometry:
1. Key-Query distance distributions
2. Vector normalization analysis
3. Key-Value correlation structure
4. Top-K concentration curves (like plot_concentration_statistics.py)
5. Pairwise distance structure visualization

Output: Self-contained HTML with embedded high-resolution PNG charts.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO
import scipy.stats as stats
from sklearn.decomposition import PCA

# Professional style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Config
DATA_PATH = '../data/attention_vectors_updated_long.jsonl'
OUTPUT_PATH = '../results/professional_attention_dashboard.html'
NUM_EXAMPLES = 100  # Random sample from 503 examples
NUM_QUERIES_PER_EXAMPLE = 1000  # Last 1000 queries per example
LAYERS = ['first_layer', 'last_layer']
HEAD_DIM = 128
SEED = 42
TOTAL_EXAMPLES_IN_FILE = 503

np.random.seed(SEED)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def fig_to_base64(fig, dpi=150):
    """Convert matplotlib figure to base64 encoded PNG."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def analyze_layer(examples, layer_name, num_queries):
    """Comprehensive analysis of one layer across examples."""
    
    print(f"\n{'='*70}")
    print(f"Analyzing {layer_name}")
    print('='*70)
    
    # Accumulators
    all_q_norms, all_k_norms, all_v_norms = [], [], []
    all_key_query_dists = []
    all_key_query_cos = []
    concentration_curves = []
    kv_correlations_l2 = []
    kv_correlations_cos = []
    key_pairwise_dists_samples = []
    val_pairwise_dists_samples = []
    key_pairwise_cos_samples = []
    val_pairwise_cos_samples = []
    top_k_masses = {10: [], 50: [], 100: [], 200: []}
    
    # Per-example statistics (for variance across examples)
    per_example_stats = {
        'mean_q_norm': [],
        'mean_k_norm': [],
        'mean_v_norm': [],
        'mean_conc_at_10pct': [],
        'mean_top10_mass': [],
        'mean_kv_corr_l2': [],
    }
    
    for ex_idx, example in enumerate(examples):
        if (ex_idx + 1) % 10 == 0:
            print(f"  Example {ex_idx+1}/{len(examples)}...")
        
        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        K = np.array(example[layer_name]['K'], dtype=np.float32)
        V = np.array(example[layer_name]['V'], dtype=np.float32)
        seq_len = Q.shape[0]
        
        # Norms
        q_norms_ex = np.linalg.norm(Q, axis=1)
        k_norms_ex = np.linalg.norm(K, axis=1)
        v_norms_ex = np.linalg.norm(V, axis=1)
        all_q_norms.extend(q_norms_ex)
        all_k_norms.extend(k_norms_ex)
        all_v_norms.extend(v_norms_ex)
        per_example_stats['mean_q_norm'].append(float(np.mean(q_norms_ex)))
        per_example_stats['mean_k_norm'].append(float(np.mean(k_norms_ex)))
        per_example_stats['mean_v_norm'].append(float(np.mean(v_norms_ex)))
        
        # Sample queries
        actual_num_queries = min(num_queries, seq_len - 100)
        query_positions = list(range(seq_len - actual_num_queries, seq_len))
        
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
            
            # Key-Query distances (L2) - subsample for speed
            sample_size = min(200, n_keys)
            sample_idx = np.random.choice(n_keys, sample_size, replace=False)
            kq_dists = np.linalg.norm(valid_keys[sample_idx] - q[None, :], axis=1)
            all_key_query_dists.extend(kq_dists)
            
            # Cosine similarity
            q_norm = q / (np.linalg.norm(q) + 1e-8)
            k_norm = valid_keys[sample_idx] / (np.linalg.norm(valid_keys[sample_idx], axis=1, keepdims=True) + 1e-8)
            cos = k_norm @ q_norm
            all_key_query_cos.extend(cos)
            
            # Concentration curve
            sorted_w = np.sort(weights)[::-1]
            cumsum = np.cumsum(sorted_w) * 100  # percentage
            pct_points = np.linspace(0, 1, 101)[1:]
            curve = np.interp(pct_points * n_keys, np.arange(1, n_keys + 1), cumsum)
            concentration_curves.append(curve)
            example_conc_curves.append(curve)
            
            # Top-K masses
            for k in top_k_masses:
                if n_keys >= k:
                    mass = float(sorted_w[:k].sum() * 100)
                    top_k_masses[k].append(mass)
                    if k == 10:
                        example_top10_masses.append(mass)
            
            # Key-Value correlation (for top-100)
            if n_keys >= 100:
                top_idx = np.argsort(logits)[-100:]
                k_top = valid_keys[top_idx]
                v_top = valid_values[top_idx]
                
                # L2 pairwise distances
                k_pw = np.linalg.norm(k_top[:, None, :] - k_top[None, :, :], axis=2)
                v_pw = np.linalg.norm(v_top[:, None, :] - v_top[None, :, :], axis=2)
                
                # Cosine pairwise distances (1 - cosine_sim)
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
                    kv_correlations_l2.append(float(corr_l2))
                    example_kv_corrs.append(float(corr_l2))
                    
                    # Sample some pairs for scatter
                    if len(key_pairwise_dists_samples) < 10000:
                        idx_sample = np.random.choice(len(k_dists), min(50, len(k_dists)), replace=False)
                        key_pairwise_dists_samples.extend(k_dists[idx_sample])
                        val_pairwise_dists_samples.extend(v_dists[idx_sample])
                
                # Cosine distance correlation
                if k_cos_dists.std() > 1e-6 and v_cos_dists.std() > 1e-6:
                    corr_cos = np.corrcoef(k_cos_dists, v_cos_dists)[0, 1]
                    kv_correlations_cos.append(float(corr_cos))
                    
                    # Sample some pairs for scatter
                    if len(key_pairwise_cos_samples) < 10000:
                        idx_sample = np.random.choice(len(k_cos_dists), min(50, len(k_cos_dists)), replace=False)
                        key_pairwise_cos_samples.extend(k_cos_dists[idx_sample])
                        val_pairwise_cos_samples.extend(v_cos_dists[idx_sample])
        
        # Per-example statistics
        if len(example_conc_curves) > 0:
            mean_curve = np.mean(example_conc_curves, axis=0)
            per_example_stats['mean_conc_at_10pct'].append(float(mean_curve[9]))  # at 10%
        if len(example_top10_masses) > 0:
            per_example_stats['mean_top10_mass'].append(float(np.mean(example_top10_masses)))
        if len(example_kv_corrs) > 0:
            per_example_stats['mean_kv_corr_l2'].append(float(np.mean(example_kv_corrs)))
    
    concentration_curves = np.array(concentration_curves)
    
    print(f"  ✓ Analyzed {len(concentration_curves)} queries across {len(examples)} examples")
    
    return {
        'q_norms': np.array(all_q_norms),
        'k_norms': np.array(all_k_norms),
        'v_norms': np.array(all_v_norms),
        'key_query_dists': np.array(all_key_query_dists),
        'key_query_cos': np.array(all_key_query_cos),
        'conc_curves': concentration_curves,
        'conc_mean': np.mean(concentration_curves, axis=0),
        'conc_p10': np.percentile(concentration_curves, 10, axis=0),
        'conc_p50': np.percentile(concentration_curves, 50, axis=0),
        'conc_p90': np.percentile(concentration_curves, 90, axis=0),
        'conc_p99': np.percentile(concentration_curves, 99, axis=0),
        'conc_x': np.linspace(0, 100, 101)[1:],
        'kv_corr_l2': np.array(kv_correlations_l2),
        'kv_corr_cos': np.array(kv_correlations_cos),
        'key_pw_dists': np.array(key_pairwise_dists_samples),
        'val_pw_dists': np.array(val_pairwise_dists_samples),
        'key_pw_cos': np.array(key_pairwise_cos_samples),
        'val_pw_cos': np.array(val_pairwise_cos_samples),
        'top_k_masses': top_k_masses,
        'per_example_stats': per_example_stats,
    }

def create_visualizations(data_first, data_last):
    """Generate all plots comparing first and last layers side-by-side."""
    
    plots = {}
    
    # ============================================================
    # 1. CONCENTRATION CURVE (like plot_concentration_statistics.py)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, data, title in [(axes[0], data_first, 'First Layer (Layer 0)'),
                             (axes[1], data_last, 'Last Layer (Layer 31)')]:
        x = data['conc_x']
        ax.plot(x, data['conc_mean'], label='Mean', color='black', linewidth=3, alpha=0.9)
        ax.plot(x, data['conc_p50'], label='Median', color='#3b82f6', linewidth=2.5, linestyle='--', alpha=0.8)
        ax.plot(x, data['conc_p10'], label='p10', color='#ef4444', linewidth=2, linestyle=':', alpha=0.7)
        ax.plot(x, data['conc_p90'], label='p90', color='#22c55e', linewidth=2, linestyle=':', alpha=0.7)
        ax.plot(x, data['conc_p99'], label='p99', color='#a855f7', linewidth=1.5, linestyle='-.', alpha=0.7)
        ax.fill_between(x, data['conc_p10'], data['conc_p90'], alpha=0.15, color='gray')
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, label='Uniform')
        
        ax.set_xlabel('% of Keys (sorted by weight)', fontweight='bold')
        ax.set_ylabel('% of Attention Mass', fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.95, fontsize=9)
        
        # Annotation at 10%
        mean_at_10 = data['conc_mean'][9]
        ax.annotate(f'{mean_at_10:.1f}%',
                    xy=(10, mean_at_10), xytext=(18, mean_at_10 - 12),
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=9)
    
    fig.suptitle('Attention Concentration Curves (Percentiles Across All Queries)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    x = data['conc_x']
    ax.plot(x, data['conc_mean'], label='Mean', color='black', linewidth=3, alpha=0.9)
    ax.plot(x, data['conc_p50'], label='Median (p50)', color='#3b82f6', linewidth=2.5, linestyle='--', alpha=0.8)
    ax.plot(x, data['conc_p10'], label='p10', color='#ef4444', linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(x, data['conc_p90'], label='p90', color='#22c55e', linewidth=2, linestyle=':', alpha=0.7)
    ax.plot(x, data['conc_p99'], label='p99', color='#a855f7', linewidth=2, linestyle='-.', alpha=0.7)
    ax.fill_between(x, data['conc_p10'], data['conc_p90'], alpha=0.15, color='gray', label='p10-p90 range')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1, label='Uniform (reference)')
    
    ax.set_xlabel('Percentage of Keys (%)', fontweight='bold')
    ax.set_ylabel('Percentage of Attention Mass Captured (%)', fontweight='bold')
    ax.set_title(f'Attention Concentration Curve\\n{layer_title}', fontweight='bold', pad=15)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='black')
    
    # Annotations
    mean_at_10 = data['conc_mean'][9]
    ax.annotate(f'{mean_at_10:.1f}% mass\\nat 10% keys',
                xy=(10, mean_at_10), xytext=(20, mean_at_10 - 15),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.tight_layout()
    plots['concentration'] = fig_to_base64(fig)
    
    # ============================================================
    # 2. VECTOR NORMS COMPARISON (First Layer vs Last Layer)
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    for row, (data, layer_title) in enumerate([(data_first, 'First Layer'), (data_last, 'Last Layer')]):
        for col, (norms, label, color) in enumerate([
            (data['q_norms'], 'Query (Q)', '#f59e0b'),
            (data['k_norms'], 'Key (K)', '#8b5cf6'),
            (data['v_norms'], 'Value (V)', '#ec4899')
        ]):
            ax = axes[row, col]
            ax.hist(norms, bins=80, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(np.mean(norms), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(norms):.2f}')
            ax.axvline(np.median(norms), color='blue', linestyle=':', linewidth=2, 
                       label=f'Median: {np.median(norms):.2f}')
            ax.set_xlabel('L2 Norm', fontweight='bold')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{layer_title} - {label}', fontweight='bold')
            ax.legend(framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.2)
            
            # Add std text
            ax.text(0.98, 0.97, f'Std: {np.std(norms):.3f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Vector Normalization Analysis', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plots['norms'] = fig_to_base64(fig, dpi=180)
    
    # ============================================================
    # 3. KEY-QUERY DISTANCE DISTRIBUTION (First vs Last)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for row, (data, layer_title) in enumerate([(data_first, 'First Layer'), (data_last, 'Last Layer')]):
        # L2 distances
        ax = axes[row, 0]
        ax.hist(data['key_query_dists'], bins=100, color='#6366f1', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(data['key_query_dists']), color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {np.mean(data['key_query_dists']):.2f}")
        ax.axvline(np.median(data['key_query_dists']), color='blue', linestyle=':', linewidth=2,
                   label=f"Median: {np.median(data['key_query_dists']):.2f}")
        ax.set_xlabel('L2 Distance ||k - q||', fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{layer_title} - L2 Distance', fontweight='bold')
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.2)
        
        # Cosine similarity
        ax = axes[row, 1]
        ax.hist(data['key_query_cos'], bins=100, color='#22d3ee', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(np.mean(data['key_query_cos']), color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {np.mean(data['key_query_cos']):.3f}")
        ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Cosine Similarity (k · q) / (||k|| ||q||)', fontweight='bold')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{layer_title} - Cosine Similarity', fontweight='bold')
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.2)
    
    fig.suptitle('Key-Query Distance Geometry', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plots['key_query_dist'] = fig_to_base64(fig, dpi=180)
    
    # ============================================================
    # 4. KEY-VALUE CORRELATION (L2 and Cosine)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for row, (data, layer_title) in enumerate([(data_first, 'First Layer'), (data_last, 'Last Layer')]):
        # L2 correlation histogram
        ax = axes[row, 0]
        if len(data['kv_corr_l2']) > 0:
            ax.hist(data['kv_corr_l2'], bins=50, color='#f59e0b', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(np.mean(data['kv_corr_l2']), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {np.mean(data['kv_corr_l2']):.3f}")
            ax.set_xlabel('Pearson r (L2 Key-Dists vs L2 Value-Dists)', fontweight='bold')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{layer_title} - L2 Distance Correlation', fontweight='bold')
            ax.legend(framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.2)
        
        # Cosine correlation histogram
        ax = axes[row, 1]
        if len(data['kv_corr_cos']) > 0:
            ax.hist(data['kv_corr_cos'], bins=50, color='#22d3ee', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(np.mean(data['kv_corr_cos']), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {np.mean(data['kv_corr_cos']):.3f}")
            ax.set_xlabel('Pearson r (Cosine Key-Dists vs Cosine Value-Dists)', fontweight='bold')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{layer_title} - Cosine Distance Correlation', fontweight='bold')
            ax.legend(framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.2)
    
    fig.suptitle('Key-Value Correlation Structure (Top-100 Keys per Query)', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plots['kv_correlation'] = fig_to_base64(fig, dpi=180)
    
    # ============================================================
    # 5. TOP-K MASS COMPARISON (First vs Last)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    k_vals = [10, 50, 100, 200]
    colors_topk = ['#ef4444', '#f59e0b', '#22c55e', '#3b82f6']
    
    for ax, data, layer_title in [(axes[0], data_first, 'First Layer'), (axes[1], data_last, 'Last Layer')]:
        positions = np.arange(len(k_vals))
        means = [np.mean(data['top_k_masses'][k]) for k in k_vals]
        stds = [np.std(data['top_k_masses'][k]) for k in k_vals]
        
        bars = ax.bar(positions, means, color=colors_topk, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.errorbar(positions, means, yerr=stds, fmt='none', color='black', linewidth=2, capsize=5, alpha=0.7)
        
        # Value labels
        for i, (bar, mean_val) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 2,
                    f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xticks(positions)
        ax.set_xticklabels([f'Top-{k}' for k in k_vals])
        ax.set_ylabel('Attention Mass Captured (%)', fontweight='bold')
        ax.set_title(layer_title, fontweight='bold', pad=10)
        ax.set_ylim([0, max(means) + max(stds) + 12])
        ax.grid(True, alpha=0.2, axis='y')
    
    fig.suptitle('Top-K Attention Mass Concentration', fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    plots['topk_mass'] = fig_to_base64(fig, dpi=180)
    
    # ============================================================
    # 6. PER-EXAMPLE VARIANCE (showing how stats vary across examples)
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    metrics_to_plot = [
        ('mean_q_norm', 'Mean Q Norm per Example', '#f59e0b'),
        ('mean_k_norm', 'Mean K Norm per Example', '#8b5cf6'),
        ('mean_v_norm', 'Mean V Norm per Example', '#ec4899'),
        ('mean_conc_at_10pct', 'Mean Conc. at 10% Keys per Example', '#6366f1'),
        ('mean_top10_mass', 'Mean Top-10 Mass per Example', '#22c55e'),
        ('mean_kv_corr_l2', 'Mean K-V L2 Corr per Example', '#f59e0b'),
    ]
    
    for ax, data, layer_title in [(axes[0, :], data_first, 'First Layer'), (axes[1, :], data_last, 'Last Layer')]:
        for col, (metric_key, metric_label, color) in enumerate(metrics_to_plot[:3]):
            ax_curr = ax[col]
            vals = data['per_example_stats'].get(metric_key, [])
            if len(vals) > 0:
                ax_curr.hist(vals, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax_curr.axvline(np.mean(vals), color='red', linestyle='--', linewidth=2,
                               label=f'Mean: {np.mean(vals):.3f}')
                ax_curr.set_xlabel(metric_label, fontweight='bold', fontsize=10)
                ax_curr.set_ylabel('Frequency')
                ax_curr.set_title(f'{layer_title}', fontweight='bold', fontsize=11)
                ax_curr.legend(framealpha=0.9, fontsize=8)
                ax_curr.grid(True, alpha=0.2)
                ax_curr.text(0.98, 0.97, f'Std: {np.std(vals):.4f}', transform=ax_curr.transAxes,
                            ha='right', va='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Per-Example Variance (How Statistics Vary Across Examples)', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plots['per_example_variance'] = fig_to_base64(fig, dpi=180)
    
    # ============================================================
    # 7. SUMMARY STATISTICS TABLE (Both Layers)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, data, layer_title in [(axes[0], data_first, 'First Layer (Layer 0)'), 
                                   (axes[1], data_last, 'Last Layer (Layer 31)')]:
        ax.axis('off')
        
        stats_data = [
            ['Metric', 'Mean', 'Median', 'Std', 'p10', 'p90'],
            ['Q Norm', f"{np.mean(data['q_norms']):.3f}", f"{np.median(data['q_norms']):.3f}",
             f"{np.std(data['q_norms']):.3f}", f"{np.percentile(data['q_norms'], 10):.3f}",
             f"{np.percentile(data['q_norms'], 90):.3f}"],
            ['K Norm', f"{np.mean(data['k_norms']):.3f}", f"{np.median(data['k_norms']):.3f}",
             f"{np.std(data['k_norms']):.3f}", f"{np.percentile(data['k_norms'], 10):.3f}",
             f"{np.percentile(data['k_norms'], 90):.3f}"],
            ['V Norm', f"{np.mean(data['v_norms']):.3f}", f"{np.median(data['v_norms']):.3f}",
             f"{np.std(data['v_norms']):.3f}", f"{np.percentile(data['v_norms'], 10):.3f}",
             f"{np.percentile(data['v_norms'], 90):.3f}"],
            ['K-Q L2', f"{np.mean(data['key_query_dists']):.2f}", f"{np.median(data['key_query_dists']):.2f}",
             f"{np.std(data['key_query_dists']):.2f}", f"{np.percentile(data['key_query_dists'], 10):.2f}",
             f"{np.percentile(data['key_query_dists'], 90):.2f}"],
            ['K-Q Cos', f"{np.mean(data['key_query_cos']):.3f}", f"{np.median(data['key_query_cos']):.3f}",
             f"{np.std(data['key_query_cos']):.3f}", f"{np.percentile(data['key_query_cos'], 10):.3f}",
             f"{np.percentile(data['key_query_cos'], 90):.3f}"],
            ['K-V Corr (L2)', 
             f"{np.mean(data['kv_corr_l2']):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A',
             f"{np.median(data['kv_corr_l2']):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A',
             f"{np.std(data['kv_corr_l2']):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A',
             f"{np.percentile(data['kv_corr_l2'], 10):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A',
             f"{np.percentile(data['kv_corr_l2'], 90):.3f}" if len(data['kv_corr_l2']) > 0 else 'N/A'],
            ['K-V Corr (Cos)', 
             f"{np.mean(data['kv_corr_cos']):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A',
             f"{np.median(data['kv_corr_cos']):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A',
             f"{np.std(data['kv_corr_cos']):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A',
             f"{np.percentile(data['kv_corr_cos'], 10):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A',
             f"{np.percentile(data['kv_corr_cos'], 90):.3f}" if len(data['kv_corr_cos']) > 0 else 'N/A'],
        ]
        
        table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                         colWidths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.13])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        
        # Style header row
        for i in range(6):
            table[(0, i)].set_facecolor('#4a5568')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, 8):
            color_bg = '#f7fafc' if i % 2 == 0 else 'white'
            for j in range(6):
                table[(i, j)].set_facecolor(color_bg)
                table[(i, j)].set_edgecolor('#cbd5e0')
        
        ax.set_title(layer_title, fontsize=13, fontweight='bold', pad=10)
    
    fig.suptitle('Summary Statistics', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plots['stats_table'] = fig_to_base64(fig, dpi=180)
    
    return plots

def build_html(plots, metadata):
    """Build professional HTML dashboard."""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Professional Attention Analysis Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Arial', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 40px 20px;
  }}
  .container {{
    max-width: 1600px;
    margin: 0 auto;
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    overflow: hidden;
  }}
  .header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 40px 50px;
  }}
  .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
  .header .subtitle {{ font-size: 16px; opacity: 0.9; margin-top: 8px; }}
  .header .meta {{ font-size: 13px; opacity: 0.8; margin-top: 12px; }}
  .content {{ padding: 50px; }}
  
  .section {{
    margin-bottom: 60px;
    padding-bottom: 40px;
    border-bottom: 2px solid #e2e8f0;
  }}
  .section:last-child {{ border-bottom: none; }}
  .section h2 {{
    font-size: 24px;
    color: #2d3748;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 3px solid #667eea;
    display: inline-block;
  }}
  .section .description {{
    font-size: 14px;
    color: #4a5568;
    line-height: 1.7;
    margin: 16px 0 24px;
    background: #f7fafc;
    padding: 16px 20px;
    border-left: 4px solid #667eea;
    border-radius: 4px;
  }}
  .description strong {{ color: #2d3748; }}
  
  
  .plot-container {{
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.07);
  }}
  .plot-container img {{
    width: 100%;
    height: auto;
    display: block;
    border-radius: 8px;
  }}
  
  .info-box {{
    background: #edf2f7;
    border-left: 4px solid #4299e1;
    padding: 16px 20px;
    margin: 20px 0;
    border-radius: 4px;
    font-size: 13px;
    color: #2d3748;
    line-height: 1.6;
  }}
  
  .highlight {{ background: #fef3c7; padding: 2px 6px; border-radius: 3px; }}
  
  footer {{
    text-align: center;
    padding: 30px;
    background: #f7fafc;
    font-size: 13px;
    color: #718096;
  }}
</style>
</head>
<body>

<div class="container">
  <div class="header">
    <h1>🔬 Attention Space Analysis Dashboard</h1>
    <div class="subtitle">Llama-3-8B (8192 tokens) · LongBench v2 · Head 0 (KV-head 0)</div>
    <div class="meta">
      Generated: {metadata['timestamp']} · 
      {metadata['num_examples']} examples · 
      {metadata['num_queries']} queries per example · 
      Dimension: {metadata['head_dim']}
    </div>
  </div>
  
  <div class="content">
    
    <div class="info-box">
      <strong>Dataset:</strong> Query (Q), Key (K), and Value (V) vectors extracted from Llama-3-8B 
      on long-context tasks from LongBench v2 (sequence length ~8192 tokens). 
      We analyze <strong>First Layer (Layer 0)</strong> and <strong>Last Layer (Layer 31)</strong>, 
      both for KV-head 0. All attention is causal. Vectors are <span class="highlight">128-dimensional float32</span>.
      <br><strong>Scale:</strong> {metadata['num_examples']} randomly sampled examples, 
      last {metadata['num_queries']} queries per example = <strong>{metadata['total_queries']:,} total queries</strong>.
    </div>
    
    <div class="section">
      <h2>1. Attention Concentration Curve</h2>
      <div class="description">
        <strong>What:</strong> For each query, sort attention weights descending and plot cumulative mass vs. % of keys.<br>
        <strong>How:</strong> Compute softmax weights for {metadata['total_queries']:,} queries, sort each, take cumulative sum.
        Report percentiles (p10, p50, p90, p99) across all queries. Diagonal = uniform attention.<br>
        <strong>Interpretation:</strong> Curves above diagonal = concentrated attention (few keys dominate). 
        On diagonal = diffuse/uniform.
      </div>
      <div class="plot-container">
        <img src="{plots['concentration']}" alt="Concentration Curve">
      </div>
    </div>
    
    <div class="section">
      <h2>2. Vector Normalization Analysis</h2>
      <div class="description">
        <strong>What:</strong> L2 norm distributions for Q, K, V vectors: <code>||v|| = sqrt(sum(v_i^2))</code>.<br>
        <strong>How:</strong> Compute norms for all vectors across all positions in all examples. 
        Plot histograms with mean/median lines.<br>
        <strong>Interpretation:</strong> Tight distribution (small std) → vectors are approximately normalized. 
        Wide spread → non-normalized, varying scales. Compare first layer (early) vs. last layer (late).
      </div>
      <div class="plot-container">
        <img src="{plots['norms']}" alt="Vector Norms">
      </div>
    </div>
    
    <div class="section">
      <h2>3. Key-Query Distance Geometry</h2>
      <div class="description">
        <strong>What:</strong> Distribution of distances between keys and their corresponding query.<br>
        <strong>How:</strong> For each query, compute L2 distance <code>||k - q||</code> and cosine similarity 
        <code>(k · q)/(||k|| ||q||)</code> to all valid keys. Aggregate across queries.<br>
        <strong>Interpretation:</strong> L2 shows absolute separation in 128-D space. Cosine shows angular structure.
        Near-zero cosine = orthogonal (high-D isotropy). Extreme values = strong alignment.
      </div>
      <div class="plot-container">
        <img src="{plots['key_query_dist']}" alt="Key-Query Distance">
      </div>
    </div>
    
    <div class="section">
      <h2>4. Key-Value Correlation Structure</h2>
      <div class="description">
        <strong>What:</strong> For top-100 keys per query, compute pairwise distances among keys and among values.
        Then compute Pearson r between those distance vectors.<br>
        <strong>How:</strong> Take top-100 keys by logit. Compute all C(100,2) = 4950 pairwise distances for keys and values.
        Correlate. Do this for both L2 distance and cosine distance (1 - cos_sim).<br>
        <strong>Interpretation:</strong> <strong>High r</strong> → "similar keys have similar values" (smooth manifold). 
        <strong>Low r</strong> → key proximity does not predict value proximity.
      </div>
      <div class="plot-container">
        <img src="{plots['kv_correlation']}" alt="Key-Value Correlation">
      </div>
    </div>
    
    <div class="section">
      <h2>5. Top-K Mass Concentration</h2>
      <div class="description">
        <strong>What:</strong> Percentage of attention mass captured by top-K keys (K = 10, 50, 100, 200).<br>
        <strong>How:</strong> Sort weights per query, sum top-K. Average across all queries. Error bars = std.<br>
        <strong>Interpretation:</strong> High values = concentrated attention. Low values = diffuse/uniform.
      </div>
      <div class="plot-container">
        <img src="{plots['topk_mass']}" alt="Top-K Mass">
      </div>
    </div>
    
    <div class="section">
      <h2>6. Per-Example Variance</h2>
      <div class="description">
        <strong>What:</strong> How statistics vary across examples (not just queries). Each bar is one example's average.<br>
        <strong>How:</strong> For each example, compute mean of the metric across its queries. Plot histogram of those means.<br>
        <strong>Interpretation:</strong> Tight distribution = consistent behavior across examples. 
        Wide distribution = examples differ substantially (domain-dependent or task-dependent structure).
      </div>
      <div class="plot-container">
        <img src="{plots['per_example_variance']}" alt="Per-Example Variance">
      </div>
    </div>
    
    <div class="section">
      <h2>7. Summary Statistics Table</h2>
      <div class="description">
        Comprehensive summary: mean, median, std, p10, p90 for all key metrics.
      </div>
      <div class="plot-container">
        <img src="{plots['stats_table']}" alt="Statistics Table">
      </div>
    </div>
    
  </div>
  
  <footer>
    <p>Professional Attention Analysis Dashboard · Llama-3-8B · LongBench v2</p>
    <p style="margin-top: 8px; font-size: 12px;">
      {metadata['total_queries']:,} queries analyzed · Publication-quality matplotlib/seaborn plots
    </p>
  </footer>
</div>

</body>
</html>"""
    
    return html

def main():
    print("="*70)
    print("PROFESSIONAL ATTENTION DASHBOARD GENERATOR")
    print("="*70)
    print(f"Config: {NUM_EXAMPLES} random examples (out of {TOTAL_EXAMPLES_IN_FILE})")
    print(f"        {NUM_QUERIES_PER_EXAMPLE} queries per example")
    print(f"        Total queries: {NUM_EXAMPLES * NUM_QUERIES_PER_EXAMPLE}")
    print()
    
    # Load data (random sample)
    print(f"Loading {NUM_EXAMPLES} random examples from: {DATA_PATH}")
    all_examples = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            all_examples.append(json.loads(line))
    
    # Randomly sample
    selected_indices = np.random.choice(len(all_examples), NUM_EXAMPLES, replace=False)
    examples = [all_examples[i] for i in selected_indices]
    print(f"✓ Loaded and sampled {len(examples)} examples")
    
    # Analyze layers
    layer_data = {}
    for layer_name in LAYERS:
        layer_data[layer_name] = analyze_layer(examples, layer_name, NUM_QUERIES_PER_EXAMPLE)
    
    # Generate plots
    print(f"\nGenerating visualizations...")
    plots = create_visualizations(layer_data['first_layer'], layer_data['last_layer'])
    print(f"  ✓ All plots generated")
    
    # Build HTML
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'num_examples': NUM_EXAMPLES,
        'num_queries': NUM_QUERIES_PER_EXAMPLE,
        'total_queries': NUM_EXAMPLES * NUM_QUERIES_PER_EXAMPLE,
        'head_dim': HEAD_DIM,
    }
    
    html = build_html(plots, metadata)
    
    # Save
    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(html)
    
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved: {output} ({size_mb:.1f} MB)")
    print("\nOpen the HTML file in a browser to view the dashboard.")

if __name__ == "__main__":
    main()
