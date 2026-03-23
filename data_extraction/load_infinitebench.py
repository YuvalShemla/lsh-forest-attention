#!/usr/bin/env python3
"""
Load InfiniteBench from HuggingFace and save as JSON.

Produces a structure matching our longbench_v2 format:
  {"metadata": {...}, "examples": [...]}

Each example has: id, task, context, question, answer, options, _context_chars

Usage:
    pip install datasets
    python load_infinitebench.py              # preview first 3 examples
    python load_infinitebench.py --save       # save full JSON to data/
"""

import argparse
import json
import os
import sys
from datasets import load_dataset, Features, Value, Sequence


TASKS = [
    "longbook_sum_eng",
    "longbook_choice_eng",
    "longbook_qa_eng",
    "longdialogue_qa_eng",
    "longbook_qa_chn",
    "math_find",
    "math_calc",
    "code_run",
    "code_debug",
    "kv_retrieval",
    "number_string",
    "passkey",
]

TASK_DESCRIPTIONS = {
    "longbook_sum_eng": "Book Summarization (English)",
    "longbook_choice_eng": "Book Multiple Choice (English)",
    "longbook_qa_eng": "Book QA (English)",
    "longdialogue_qa_eng": "Long Dialogue QA (English)",
    "longbook_qa_chn": "Book QA (Chinese)",
    "math_find": "Math - Number Finding",
    "math_calc": "Math - Calculation",
    "code_run": "Code - Execution",
    "code_debug": "Code - Debugging",
    "kv_retrieval": "Key-Value Retrieval",
    "number_string": "Number String Retrieval",
    "passkey": "Passkey Retrieval",
}


def load_infinitebench():
    """Load all InfiniteBench tasks and return a flat list of examples."""
    ft = Features({
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string")),
    })

    ds = load_dataset("xinrongzhang2022/InfiniteBench", features=ft)

    examples = []
    task_counts = {}

    for split_name in sorted(ds.keys()):
        split = ds[split_name]
        task_counts[split_name] = len(split)

        for row in split:
            examples.append({
                "id": row["id"],
                "task": split_name,
                "task_description": TASK_DESCRIPTIONS.get(split_name, split_name),
                "context": row["context"],
                "question": row["input"],
                "answer": row["answer"],
                "options": row["options"],
                "_context_chars": len(row["context"]) if row["context"] else 0,
            })

    print(f"Loaded {len(examples)} examples across {len(task_counts)} tasks:")
    for task, count in sorted(task_counts.items()):
        chars = [e["_context_chars"] for e in examples if e["task"] == task]
        avg_chars = sum(chars) // len(chars) if chars else 0
        print(f"  {task:<25} {count:>5} examples, avg {avg_chars:>10,} chars")

    return examples, task_counts


def save_json(examples, task_counts, output_path):
    """Save examples as JSON in our standard format."""
    output = {
        "metadata": {
            "note": "InfiniteBench - long context benchmark (100K+ tokens)",
            "source": "xinrongzhang2022/InfiniteBench",
            "paper": "inftyBench: Extending Long Context Evaluation Beyond 100K Tokens (ACL 2024)",
            "total_examples": len(examples),
            "tasks": task_counts,
        },
        "examples": examples,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved to {output_path} ({size_mb:.0f} MB, {len(examples)} examples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save full JSON to data/")
    parser.add_argument("--output", default=None, help="Output path (default: data/infinitebench_full.json)")
    args = parser.parse_args()

    examples, task_counts = load_infinitebench()

    print(f"\n--- Preview (first 3 examples) ---")
    for e in examples[:3]:
        print(f"\n  [{e['task']}] id={e['id']}")
        print(f"    context:  {e['_context_chars']:,} chars")
        print(f"    question: {e['question'][:100]}...")
        print(f"    answer:   {e['answer']}")
        print(f"    options:  {e['options'][:4] if e['options'] else '(none)'}")

    if args.save:
        output_path = args.output or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "infinitebench_full.json"
        )
        save_json(examples, task_counts, output_path)
