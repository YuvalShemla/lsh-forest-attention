#!/usr/bin/env python3
"""
Fixed version: Extract Q, K, V with proper structure
- ALL query positions (not just last token)
- Configurable: single head or all heads
- Proper head mapping for GQA
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np

# CONFIGURATION
EXTRACT_ALL_HEADS = False  # Set to True to extract all heads, False for single head (smaller file)
SINGLE_HEAD_IDX = 0  # Which head to extract if EXTRACT_ALL_HEADS=False

# Load model
model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Model config
num_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // num_heads
group_size = num_heads // num_kv_heads  # How many Q heads per KV head

print(f"Model: {model_name}")
print(f"Q heads: {num_heads}, KV heads: {num_kv_heads}, Group size: {group_size}")
print(f"Mode: {'ALL HEADS' if EXTRACT_ALL_HEADS else f'SINGLE HEAD (head {SINGLE_HEAD_IDX})'}")

# Storage for extracted tensors
attention_cache = {}

def hook_fn(module, args, kwargs, output, layer_idx):
    """
    Hook function to capture Q, K, V from attention layer
    """
    # Get hidden_states from kwargs (HF style) or args (fallback)
    hidden_states = kwargs.get('hidden_states') if kwargs else args[0]

    # [batch, seq_len, hidden_size]
    batch_size, seq_len, _ = hidden_states.shape
    
    # Project to Q, K, V
    Q = module.q_proj(hidden_states)  # [batch, seq_len, num_heads * head_dim]
    K = module.k_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]
    V = module.v_proj(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]
    
    # Reshape to separate heads
    Q = Q.view(batch_size, seq_len, num_heads, head_dim)
    K = K.view(batch_size, seq_len, num_kv_heads, head_dim)
    V = V.view(batch_size, seq_len, num_kv_heads, head_dim)
    
    # Transpose to [batch, heads, seq_len, head_dim]
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    
    # Store
    attention_cache[layer_idx] = {
        'Q': Q.detach().cpu().float(),  # Convert to float32
        'K': K.detach().cpu().float(),
        'V': V.detach().cpu().float()
    }

# Register hooks for first and last layers
first_layer_idx = 0
last_layer_idx = len(model.model.layers) - 1

model.model.layers[first_layer_idx].self_attn.register_forward_hook(
    lambda m, a, k, o: hook_fn(m, a, k, o, first_layer_idx),
    with_kwargs=True
)

model.model.layers[last_layer_idx].self_attn.register_forward_hook(
    lambda m, a, k, o: hook_fn(m, a, k, o, last_layer_idx),
    with_kwargs=True
)

# Process each example
with open('longbench_v2_truncated_7k_smart.json', 'r') as f:
    data = json.load(f)

output_file = open('attention_vectors_long_bench_llama_8b.jsonl', 'w')

for idx, example in enumerate(data['examples']):
    print(f"\n[{idx+1}/{len(data['examples'])}] Processing: {example['domain']}")
    
    # Prepare prompt
    prompt = f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    seq_len = inputs['input_ids'].shape[1]
    
    print(f"  Sequence length: {seq_len} tokens")
    
    # Clear cache
    attention_cache.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Forward pass (triggers hooks)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract tensors
    first_Q = attention_cache[first_layer_idx]['Q'][0].numpy()  # [num_heads, seq_len, head_dim]
    first_K = attention_cache[first_layer_idx]['K'][0].numpy()  # [num_kv_heads, seq_len, head_dim]
    first_V = attention_cache[first_layer_idx]['V'][0].numpy()
    
    last_Q = attention_cache[last_layer_idx]['Q'][0].numpy()
    last_K = attention_cache[last_layer_idx]['K'][0].numpy()
    last_V = attention_cache[last_layer_idx]['V'][0].numpy()
    
    if EXTRACT_ALL_HEADS:
        # ============================================
        # MODE 1: ALL HEADS (with mapping)
        # ============================================
        
        # Build head mapping: which KV head does each Q head use?
        head_mapping = []
        for q_head in range(num_heads):
            kv_head = q_head // group_size
            head_mapping.append({
                "q_head_idx": q_head,
                "kv_head_idx": kv_head,
                "group": q_head // group_size
            })
        
        result = {
            "example_id": example.get("_id", f"example_{idx}"),
            "domain": example["domain"],
            "sub_domain": example.get("sub_domain", "Unknown"),
            "question": example["question"],
            "answer": example["answer"],
            "sequence_length": seq_len,
            
            "model_metadata": {
                "model_name": model_name,
                "num_layers": len(model.model.layers),
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "group_size": group_size
            },
            
            "head_mapping": head_mapping,
            
            "first_layer": {
                "layer_idx": first_layer_idx,
                "Q": first_Q.tolist(),  # [num_heads, seq_len, head_dim]
                "K": first_K.tolist(),  # [num_kv_heads, seq_len, head_dim]
                "V": first_V.tolist(),
                "Q_shape": list(first_Q.shape),
                "K_shape": list(first_K.shape),
                "V_shape": list(first_V.shape)
            },
            
            "last_layer": {
                "layer_idx": last_layer_idx,
                "Q": last_Q.tolist(),
                "K": last_K.tolist(),
                "V": last_V.tolist(),
                "Q_shape": list(last_Q.shape),
                "K_shape": list(last_K.shape),
                "V_shape": list(last_V.shape)
            },
            
            "position_ids": list(range(seq_len)),
            
            "usage_note": "Q[q_head_idx, :, :] uses K[kv_head_idx, :, :] and V[kv_head_idx, :, :]. See head_mapping for Q->KV mapping."
        }
        
        print(f"  Extracted ALL heads - Q: {first_Q.shape}, K: {first_K.shape}")
        
    else:
        # ============================================
        # MODE 2: SINGLE HEAD (recommended)
        # ============================================
        
        # Extract single head
        kv_head_idx = SINGLE_HEAD_IDX // group_size
        
        # Extract: [seq_len, head_dim]
        first_Q_single = first_Q[SINGLE_HEAD_IDX, :, :]  # [seq_len, head_dim]
        first_K_single = first_K[kv_head_idx, :, :]      # [seq_len, head_dim]
        first_V_single = first_V[kv_head_idx, :, :]
        
        last_Q_single = last_Q[SINGLE_HEAD_IDX, :, :]
        last_K_single = last_K[kv_head_idx, :, :]
        last_V_single = last_V[kv_head_idx, :, :]
        
        result = {
            "example_id": example.get("_id", f"example_{idx}"),
            "domain": example["domain"],
            "sub_domain": example.get("sub_domain", "Unknown"),
            "question": example["question"],
            "answer": example["answer"],
            "sequence_length": seq_len,
            
            "model_metadata": {
                "model_name": model_name,
                "num_layers": len(model.model.layers),
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "group_size": group_size
            },
            
            "first_layer": {
                "layer_idx": first_layer_idx,
                "head_idx": SINGLE_HEAD_IDX,
                "kv_head_idx": kv_head_idx,
                "Q": first_Q_single.tolist(),  # [seq_len, head_dim]
                "K": first_K_single.tolist(),  # [seq_len, head_dim]
                "V": first_V_single.tolist(),
                "Q_shape": list(first_Q_single.shape),
                "K_shape": list(first_K_single.shape),
                "V_shape": list(first_V_single.shape)
            },
            
            "last_layer": {
                "layer_idx": last_layer_idx,
                "head_idx": SINGLE_HEAD_IDX,
                "kv_head_idx": kv_head_idx,
                "Q": last_Q_single.tolist(),
                "K": last_K_single.tolist(),
                "V": last_V_single.tolist(),
                "Q_shape": list(last_Q_single.shape),
                "K_shape": list(last_K_single.shape),
                "V_shape": list(last_V_single.shape)
            },
            
            "position_ids": list(range(seq_len)),
            
            "usage_note": f"Single head extraction: Q head {SINGLE_HEAD_IDX} with KV head {kv_head_idx}"
        }
        
        print(f"  Extracted head {SINGLE_HEAD_IDX} - Q: {first_Q_single.shape}, K: {first_K_single.shape}")
    
    # Write to file
    output_file.write(json.dumps(result) + '\n')
    output_file.flush()
    
    # Break after first example for testing (comment out for full run)
    if idx == 0:
        print("\n⚠️  Breaking after first example (for testing). Remove 'break' for full run.")
        break

output_file.close()

print("\n" + "="*80)
print("✅ EXTRACTION COMPLETE!")
print("="*80)
print(f"Output: attention_vectors.jsonl")
print(f"Mode: {'ALL HEADS' if EXTRACT_ALL_HEADS else f'SINGLE HEAD {SINGLE_HEAD_IDX}'}")
print("\nStructure:")
if EXTRACT_ALL_HEADS:
    print("  Q: [num_heads, seq_len, head_dim]")
    print("  K: [num_kv_heads, seq_len, head_dim]")
    print("  V: [num_kv_heads, seq_len, head_dim]")
    print("  + head_mapping: Q head -> KV head mapping")
else:
    print("  Q: [seq_len, head_dim]")
    print("  K: [seq_len, head_dim]")
    print("  V: [seq_len, head_dim]")
    print(f"  head_idx={SINGLE_HEAD_IDX}, kv_head_idx={SINGLE_HEAD_IDX // group_size}")
print("="*80)
