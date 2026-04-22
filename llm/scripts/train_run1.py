#!/usr/bin/env python3
"""
LLM Phase 3 — Fine-Tuning Run 1: Resilient Behavior
Goal: Model reliably ignores injected instructions, stays on task, outputs valid JSON.

Usage
-----
python llm/scripts/train_run1.py --config llm/configs/run1_config.yaml

# Dry-run (validates config + data without loading model)
python llm/scripts/train_run1.py --config llm/configs/run1_config.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("pyyaml not installed: pip install pyyaml")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    # Merge base config if referenced
    if "_base" in cfg:
        base_path = Path(config_path).parent / cfg.pop("_base")
        with open(base_path) as fh:
            base = yaml.safe_load(fh)
        # Deep merge: config overrides base
        merged = _deep_merge(base, cfg)
        return merged
    return cfg


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _filter_subsets(
    dataset: List[Dict[str, Any]],
    subsets: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Filter to specific data subsets for this training run."""
    if not subsets:
        return dataset

    _SUBSET_DECISIONS = {
        # all injection/manipulation examples
        "instruction_resistance": ["block"],
        "task_adherence": ["allow"],
        "safe_html_transform": ["allow"],
        "content_prioritization": ["allow", "review"],
        "fraud_intent": ["block", "review"],
        "prompt_injection": ["block"],
        "context_deviation": ["review", "block"],
        "data_exfiltration": ["block"],
        "obfuscation_evasion": ["block"],
        "unauthorized_action": ["block"],
        "url_domain_risk": ["block", "review"],
        "benign_examples": ["allow"],
    }

    allowed_decisions = set()
    for subset in subsets:
        allowed_decisions.update(_SUBSET_DECISIONS.get(subset, []))

    if not allowed_decisions:
        return dataset

    return [
        ex for ex in dataset
        if ex.get("metadata", {}).get("decision") in allowed_decisions
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _format_for_sft(example: Dict[str, Any]) -> str:
    """
    Format a training example into a single string using the Llama 3.1 chat template.
    For SFT with TRL, we use the messages format directly.
    """
    return example  # TRL's SFTTrainer accepts messages format natively


def train(config: Dict[str, Any], dry_run: bool = False) -> None:
    import random
    import numpy as np

    seed = config.get("experiment", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    print(f"Loading training data from {config['data']['train_file']}...")
    train_data = _load_jsonl(config["data"]["train_file"])
    val_data = _load_jsonl(config["data"]["val_file"])

    subsets = config.get("data_subsets")
    if subsets:
        train_data = _filter_subsets(train_data, subsets)
        val_data = _filter_subsets(val_data, subsets)

    print(f"  Train: {len(train_data):,} examples")
    print(f"  Val:   {len(val_data):,} examples")

    if dry_run:
        print("\nDry-run mode: skipping model load and training.")
        print("Config validated successfully.")
        return

    # ---- Imports (deferred to avoid load time overhead on dry-run) ----
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer

    # ---- Distributed context (set by torchrun) ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    # ---- Experiment tracking ----
    exp = config.get("experiment", {})
    if exp.get("report_to") == "wandb" and is_main:
        os.environ.setdefault("WANDB_PROJECT", exp.get(
            "wandb_project", "fraud-detection-llm"))
        os.environ.setdefault(
            "WANDB_RUN_NAME", exp.get("wandb_run_name", "run1"))

    # ---- Model — fp16, loaded to CPU so FSDP can shard to GPUs ----
    # No quantization needed: with 4× T4 (16 GB each) and FSDP full_shard,
    # each GPU holds only ~4 GB of the 8B fp16 model.
    model_id = config["model"]["base_model_id"]
    if is_main:
        print(
            f"\nLoading {model_id} in fp16 (CPU → FSDP will shard to GPUs)...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=config["model"].get("trust_remote_code", False),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        fallback = config["model"]["fallback_model_id"]
        if is_main:
            print(
                f"  Warning: {model_id} failed ({exc}). Trying fallback {fallback}...")
        model = AutoModelForCausalLM.from_pretrained(
            fallback,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            fallback, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if is_main:
        print("  Model loaded to CPU. FSDP will shard across GPUs during training.")

    # ---- LoRA — applied before FSDP wrapping ----
    t_cfg = config["training"]
    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Datasets ----
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # ---- Training arguments ----
    output_dir = t_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # FSDP config — read from YAML fsdp block; Trainer uses torchrun env vars
    fsdp_cfg = config.get("fsdp", {})
    fsdp_strategy = fsdp_cfg.get(
        "strategy", "full_shard auto_wrap") if fsdp_cfg.get("enabled", False) else ""
    fsdp_hf_config = {
        "min_num_params": fsdp_cfg.get("min_num_params", 1e8),
        "backward_prefetch": fsdp_cfg.get("backward_prefetch", "backward_pre"),
        "forward_prefetch": fsdp_cfg.get("forward_prefetch", False),
        "offload_params": fsdp_cfg.get("offload_params", False),
        "use_orig_params": fsdp_cfg.get("use_orig_params", True),
        "sync_module_states": fsdp_cfg.get("sync_module_states", True),
    } if fsdp_strategy else {}

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        num_train_epochs=t_cfg["num_train_epochs"],
        learning_rate=float(t_cfg["learning_rate"]),
        fp16=t_cfg.get("fp16", True),
        bf16=t_cfg.get("bf16", False),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        optim=t_cfg.get("optim", "adamw_torch"),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.05),
        logging_steps=t_cfg.get("logging_steps", 25),
        save_steps=t_cfg.get("save_steps", 100),
        eval_steps=t_cfg.get("eval_steps", 100),
        evaluation_strategy=t_cfg.get("evaluation_strategy", "steps"),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_loss"),
        report_to=exp.get("report_to", "none"),
        run_name=exp.get("wandb_run_name"),
        seed=seed,
        fsdp=fsdp_strategy,
        fsdp_config=fsdp_hf_config if fsdp_hf_config else None,
    )

    # ---- SFT Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=config["model"].get("max_seq_length", 2048),
        dataset_text_field=None,  # using messages format
        packing=False,
    )

    print(f"\nStarting Run 1 training → {output_dir}")
    trainer.train()

    # FSDP consolidates state dict on rank 0 before saving; other ranks wait.
    if is_main:
        print(f"\nSaving final checkpoint to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")
    if is_main:
        tokenizer.save_pretrained(f"{output_dir}/final")
        print("\nRun 1 training complete.")
        print("Next step: evaluate with python llm/scripts/eval.py --checkpoint checkpoints/run1/final")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-Tuning Run 1 — Resilient Behavior")
    p.add_argument("--config", default="llm/configs/run1_config.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config and data without loading model")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    train(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
