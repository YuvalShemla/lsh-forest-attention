#!/usr/bin/env python3
"""
Combined plot: baseline curves + LSH scatter on the same axes.
Baseline x-axis is budget %, LSH x-axis is mean_budget converted to % of ~8142 keys.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path('../../results/spring_comparison/run_20260225_095031')
with open(RESULTS_DIR / 'results.json') as f:
    data = json.load(f)

meta = data['metadata']
K_PCT = meta['k_percentages']
SH_K = meta['simhash_K_values']
SH_L = meta['simhash_L_values']
CP_L_K1 = meta['cp_L_values_k1']
CP_L_K2 = meta['cp_L_values_k2']
MH_VALUES = meta['min_hits_values']

# Estimate total keys from the max-budget SimHash config
# (K=2, L=75, h=1 retrieves nearly everything)
TOTAL_KEYS = 8142  # from the results

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


def plot_combined(agg, layer_label, mh_val, filename):
    fig, ax = plt.subplots(figsize=(13, 8))

    x_pct = np.array(K_PCT, dtype=float)

    # --- Baseline curves ---
    for method, label, color, marker, ls, lw in [
        ('topk', 'Top-K', '#8b5cf6', 'o', '-', 2.5),
        ('uniform', 'Uniform Sampling', '#f97316', '^', '-.', 2.5),
        ('oracle', 'Oracle Sampling', '#16a34a', 'D', '--', 2.5),
    ]:
        means = np.array([agg['baselines'][method][str(k)]['mean'] for k in K_PCT])
        stds = np.array([agg['baselines'][method][str(k)]['std'] for k in K_PCT])
        ax.plot(x_pct, means, marker=marker, linewidth=lw, markersize=5,
                color=color, label=label, linestyle=ls, alpha=0.9, zorder=3)
        ax.fill_between(x_pct, means - stds, means + stds, color=color, alpha=0.07)

    # --- SimHash SNIS scatter ---
    sh_cmap = plt.cm.Blues(np.linspace(0.3, 0.95, len(SH_K)))
    for i, K in enumerate(SH_K):
        bx, by, labs = [], [], []
        for L in SH_L:
            key = f"simhash_K{K}_L{L}_h{mh_val}"
            if key not in agg['lsh']:
                continue
            d = agg['lsh'][key]
            if d['n'] > 0 and d['mean_budget'] > 0:
                bx.append(d['mean_budget'] / TOTAL_KEYS * 100)
                by.append(d['mean_error'])
                labs.append(f"L={L}")
        if bx:
            ax.scatter(bx, by, color=sh_cmap[i], marker='x', s=55, alpha=0.85,
                       label=f'SimHash K={K}', zorder=5)
            for xi, yi, lab in zip(bx, by, labs):
                ax.annotate(lab, (xi, yi), fontsize=5, alpha=0.45,
                            xytext=(3, 3), textcoords='offset points')

    # --- CrossPoly scatter ---
    cp_colors = {1: '#e11d48', 2: '#b91c1c'}
    cp_markers = {1: '*', 2: 'P'}
    for k_cp in [1, 2]:
        L_vals = CP_L_K1 if k_cp == 1 else CP_L_K2
        bx, by, labs = [], [], []
        for L in L_vals:
            key = f"cp_k{k_cp}_L{L}_h{mh_val}"
            if key not in agg['lsh']:
                continue
            d = agg['lsh'][key]
            if d['n'] > 0 and d['mean_budget'] > 0:
                bx.append(d['mean_budget'] / TOTAL_KEYS * 100)
                by.append(d['mean_error'])
                labs.append(f"L={L}")
        if bx:
            ax.scatter(bx, by, color=cp_colors[k_cp], marker=cp_markers[k_cp],
                       s=80, alpha=0.85, label=f'CrossPoly k={k_cp}', zorder=5)
            for xi, yi, lab in zip(bx, by, labs):
                ax.annotate(lab, (xi, yi), fontsize=5, alpha=0.45,
                            xytext=(3, 3), textcoords='offset points')

    mh_labels = {1: 'h>=1', 2: 'h>=2 (MagicPIG)', 3: 'h>=3'}
    ax.set_xlabel('Budget (% of keys)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Relative L2 Error (mean)', fontweight='bold', fontsize=12)
    ax.set_title(f'{layer_label} — Baselines + LSH-SNIS [{mh_labels[mh_val]}]',
                 fontweight='bold', fontsize=13, pad=10)
    ax.set_xlim([0, 100])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.95, edgecolor='black', ncol=2)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {RESULTS_DIR / filename}")


for layer in ['first_layer', 'last_layer']:
    agg = data['aggregated'][layer]
    ll = 'First Layer (Layer 0)' if 'first' in layer else 'Last Layer (Layer 31)'
    for mh in MH_VALUES:
        plot_combined(agg, ll, mh, f'combined_{layer}_h{mh}.png')

print("Done!")
