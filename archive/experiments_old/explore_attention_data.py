#!/usr/bin/env python3
"""
Data Exploration Script for Attention Vectors - Optimized Version
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

print("Starting script...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

def softmax(x):
    """Numerically stable softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_attention_entropy(attn_weights):
    """Compute entropy of attention distribution"""
    eps = 1e-10
    return -np.sum(attn_weights * np.log(attn_weights + eps))

def load_and_explore(filepath='../data/attention_vectors_updated.jsonl', layer_name='first_layer'):
    """Load data and create visualizations"""
    
    print("="*70)
    print("🔍 ATTENTION DATA EXPLORER")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading: {filepath}")
    try:
        with open(filepath, 'r') as f:
            example = json.loads(f.readline())
        print("✅ JSON loaded successfully")
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return
    
    # Extract arrays
    print("📊 Converting to numpy arrays...")
    try:
        Q = np.array(example[layer_name]['Q'], dtype=np.float32)
        print(f"   Q loaded: {Q.shape}")
        K = np.array(example[layer_name]['K'], dtype=np.float32)
        print(f"   K loaded: {K.shape}")
        V = np.array(example[layer_name]['V'], dtype=np.float32)
        print(f"   V loaded: {V.shape}")
    except Exception as e:
        print(f"❌ Error converting to numpy: {e}")
        return
    
    seq_len, head_dim = Q.shape
    layer_idx = example[layer_name]['layer_idx']
    head_idx = example[layer_name]['head_idx']
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Example: {example['example_id']}")
    print(f"   Domain: {example['domain']}")
    print(f"   Layer: {layer_idx}, Head: {head_idx}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Head dim: {head_dim}")
    
    # Create figure
    print("\n📊 Creating visualizations...")
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # ==========================================
    # Plot 1: Attention Weights for Last Queries
    # ==========================================
    print("   → Plot 1: Last query attention weights...")
    ax1 = fig.add_subplot(gs[0, :2])
    
    num_last_queries = 5
    last_positions = list(range(seq_len - num_last_queries, seq_len))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_last_queries))
    
    for idx, i in enumerate(last_positions):
        q = Q[i]
        k = K[:i+1]
        scores = q @ k.T / np.sqrt(head_dim)
        attn_weights = softmax(scores)
        positions = np.arange(len(attn_weights))
        ax1.plot(positions, attn_weights, label=f'Query @ pos {i}', 
                color=colors[idx], alpha=0.7, linewidth=2)
    
    ax1.set_xlabel('Key Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    ax1.set_title(f'Attention Weight Distribution for Last {num_last_queries} Queries\n(Layer {layer_idx}, Head {head_idx})', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ==========================================
    # Plot 2: Top-K Concentration
    # ==========================================
    print("   → Plot 2: Top-K attention concentration...")
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Test different K values
    k_values = [10, 50, 100, 200, 500]
    colors_topk = plt.cm.plasma(np.linspace(0.2, 0.9, len(k_values)))
    
    sample_size = 100
    sample_idx = np.linspace(10, seq_len-1, sample_size, dtype=int)
    
    for k_idx, k_val in enumerate(k_values):
        mass_captured = []
        
        for i in sample_idx:
            q = Q[i]
            k = K[:i+1]
            scores = q @ k.T / np.sqrt(head_dim)
            attn_weights = softmax(scores)
            
            # Get top-k weights
            if len(attn_weights) >= k_val:
                top_k_weights = np.sort(attn_weights)[-k_val:]
                mass = top_k_weights.sum()
            else:
                mass = attn_weights.sum()  # All weights if fewer than k
            
            mass_captured.append(mass * 100)  # Convert to percentage
        
        ax2.plot(sample_idx, mass_captured, label=f'Top-{k_val}', 
                color=colors_topk[k_idx], linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Query Position', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Percentage of Total Attention Mass (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Top-K Attention Concentration\n(How much mass do top-K keys capture?)', 
                  fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # ==========================================
    # Plot 3: Query-Key Similarity
    # ==========================================
    print("   → Plot 3: Q-K similarity distribution...")
    ax3 = fig.add_subplot(gs[0, 2])
    
    sample_size = 100
    sample_idx = np.linspace(10, seq_len-1, sample_size, dtype=int)
    all_sims = []
    
    for i in sample_idx:
        q_norm = Q[i] / (np.linalg.norm(Q[i]) + 1e-8)
        k_valid = K[:i+1]
        k_norm = k_valid / (np.linalg.norm(k_valid, axis=1, keepdims=True) + 1e-8)
        sims = k_norm @ q_norm
        all_sims.extend(sims)
    
    ax3.hist(all_sims, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(all_sims), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(all_sims):.3f}')
    ax3.set_xlabel('Cosine Similarity', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Query-Key Similarity Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ==========================================
    # Plot 4: Attention Entropy
    # ==========================================
    print("   → Plot 4: Attention entropy...")
    ax4 = fig.add_subplot(gs[1, 1])
    
    entropies = []
    for i in sample_idx:
        q = Q[i]
        k = K[:i+1]
        scores = q @ k.T / np.sqrt(head_dim)
        attn_weights = softmax(scores)
        entropies.append(compute_attention_entropy(attn_weights))
    
    ax4.plot(sample_idx, entropies, color='darkgreen', linewidth=2)
    ax4.fill_between(sample_idx, entropies, alpha=0.3, color='green')
    ax4.set_xlabel('Query Position', fontsize=10)
    ax4.set_ylabel('Entropy (nats)', fontsize=10)
    ax4.set_title('Attention Entropy\n(Higher = More Diffuse)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ==========================================
    # Plot 5: Key/Value Vector Norms
    # ==========================================
    print("   → Plot 5: Key/Value vector norms...")
    ax5 = fig.add_subplot(gs[1, 2])
    
    k_norms = np.linalg.norm(K, axis=1)
    v_norms = np.linalg.norm(V, axis=1)
    
    # Subsample for plotting
    plot_idx = np.linspace(0, seq_len-1, min(1000, seq_len), dtype=int)
    ax5.plot(plot_idx, k_norms[plot_idx], label='Key', color='purple', alpha=0.7, linewidth=1.5)
    ax5.plot(plot_idx, v_norms[plot_idx], label='Value', color='orange', alpha=0.7, linewidth=1.5)
    ax5.set_xlabel('Position', fontsize=10)
    ax5.set_ylabel('L2 Norm', fontsize=10)
    ax5.set_title('Key and Value Vector Norms', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Compute statistics for summary
    max_weights = []
    mean_weights = []
    
    for i in sample_idx:
        q = Q[i]
        k = K[:i+1]
        scores = q @ k.T / np.sqrt(head_dim)
        attn_weights = softmax(scores)
        max_weights.append(attn_weights.max())
        mean_weights.append(attn_weights.mean())
    
    # Save
    print("\n💾 Saving figure...")
    plt.tight_layout()
    output_path = f'../results/attention_data_exploration_{layer_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")
    
    # Print insights
    print("\n" + "="*70)
    print("🔍 KEY INSIGHTS")
    print("="*70)
    print(f"\n1. Concentration: {'FOCUSED' if np.mean(max_weights) > 0.1 else 'DIFFUSE'}")
    print(f"   • Avg max weight: {np.mean(max_weights):.3f}")
    print(f"\n2. Q-K Similarity: {np.mean(all_sims):.4f}")
    print(f"\n3. Entropy: {np.mean(entropies):.3f} nats")
    print(f"   • Normalized: {np.mean(entropies)/np.log(seq_len/2):.2%} of max")
    print(f"\n4. Vector Norms:")
    print(f"   • K norm (mean): {np.mean(k_norms):.3f}")
    print(f"   • V norm (mean): {np.mean(v_norms):.3f}")
    print("="*70)

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Explore attention vector data')
    parser.add_argument('--file', type=str, default='../data/attention_vectors_updated.jsonl',
                       help='Path to attention vectors JSONL file')
    parser.add_argument('--layer', type=str, default='first_layer',
                       choices=['first_layer', 'last_layer'],
                       help='Which layer to analyze')
    
    args = parser.parse_args()
    
    try:
        load_and_explore(filepath=args.file, layer_name=args.layer)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

