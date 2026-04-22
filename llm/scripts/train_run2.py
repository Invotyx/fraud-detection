#!/usr/bin/env python3
"""
LLM Phase 4 — Fine-Tuning Run 2: Fraud Detection Accuracy
Goal: Accurate detection and clear explanation of all 7 fraud parameters.
Starts from Run 1 checkpoint.

Usage
-----
python llm/scripts/train_run2.py --config llm/configs/run2_config.yaml \
    --from-checkpoint checkpoints/run1/final

# Dry-run
python llm/scripts/train_run2.py --config llm/configs/run2_config.yaml --dry-run
"""
from __future__ import annotations
from train_run1 import _deep_merge, _filter_subsets, _load_jsonl

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse helpers from train_run1
sys.path.insert(0, str(Path(__file__).parent))


def _load_config(config_path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("pyyaml not installed: pip install pyyaml")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if "_base" in cfg:
        search_dir = base_dir or str(Path(config_path).parent)
        base_path = Path(search_dir) / cfg.pop("_base")
        with open(base_path) as fh:
            base = yaml.safe_load(fh)
        return _deep_merge(base, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Per-parameter F1 metric
# ---------------------------------------------------------------------------

def _compute_per_parameter_f1(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute F1 score for each fraud parameter.
    A parameter is "positive" if score >= threshold.
    """
    from collections import defaultdict

    parameters = [
        "url_domain_risk", "fraud_intent", "prompt_injection",
        "context_deviation", "data_exfiltration", "obfuscation_evasion",
        "unauthorized_action",
    ]

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for pred, truth in zip(predictions, ground_truths):
        for param in parameters:
            pred_score = pred.get(param, {}).get("score", 0.0)
            true_score = truth.get(param, {}).get("score", 0.0)
            pred_pos = pred_score >= threshold
            true_pos = true_score >= threshold

            if pred_pos and true_pos:
                tp[param] += 1
            elif pred_pos and not true_pos:
                fp[param] += 1
            elif not pred_pos and true_pos:
                fn[param] += 1

    f1_scores: Dict[str, float] = {}
    for param in parameters:
        precision = tp[param] / (tp[param] + fp[param]
                                 ) if (tp[param] + fp[param]) > 0 else 0.0
        recall = tp[param] / (tp[param] + fn[param]
                              ) if (tp[param] + fn[param]) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        f1_scores[param] = round(f1, 4)

    return f1_scores


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    config: Dict[str, Any],
    from_checkpoint: Optional[str],
    dry_run: bool = False,
) -> None:
    import random
    import numpy as np

    seed = config.get("experiment", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading training data from {config['data']['train_file']}...")
    train_data = _load_jsonl(config["data"]["train_file"])
    val_data = _load_jsonl(config["data"]["val_file"])

    subsets = config.get("data_subsets")
    if subsets:
        train_data = _filter_subsets(train_data, subsets)
        val_data = _filter_subsets(val_data, subsets)

    print(f"  Train: {len(train_data):,} | Val: {len(val_data):,}")

    if dry_run:
        print("\nDry-run: skipping model load. Config and data validated.")
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    exp = config.get("experiment", {})
    if exp.get("report_to") == "wandb":
        os.environ.setdefault("WANDB_PROJECT", exp.get(
            "wandb_project", "fraud-detection-llm"))
        os.environ.setdefault(
            "WANDB_RUN_NAME", exp.get("wandb_run_name", "run2"))

    bnb_cfg = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bnb_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=bnb_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bnb_cfg.get(
            "bnb_4bit_use_double_quant", True),
    )

    model_id = config["model"]["base_model_id"]
    tokenizer_id = from_checkpoint or model_id

    print(f"\nLoading base model: {model_id}")
    _device_map = None if os.environ.get("LOCAL_RANK") else "auto"
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=_device_map,
            torch_dtype=torch.float16,
            trust_remote_code=config["model"].get("trust_remote_code", False),
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as exc:
        fallback = config["model"]["fallback_model_id"]
        print(f"  Fallback to {fallback}: {exc}")
        base_model = AutoModelForCausalLM.from_pretrained(
            fallback,
            quantization_config=bnb_config,
            device_map=_device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            fallback, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Run 1 LoRA weights if checkpoint provided
    if from_checkpoint and Path(from_checkpoint).exists():
        print(f"  Loading Run 1 checkpoint from {from_checkpoint}...")
        model = PeftModel.from_pretrained(base_model, from_checkpoint)
        # Unload and re-apply new LoRA for Run 2
        model = model.merge_and_unload()
        print("  Run 1 weights merged into base model.")
    else:
        model = base_model

    t_cfg = config["training"]

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
    )

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

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    output_dir = t_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        num_train_epochs=t_cfg["num_train_epochs"],
        learning_rate=float(t_cfg["learning_rate"]),
        fp16=t_cfg.get("fp16", True),
        bf16=t_cfg.get("bf16", False),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        optim=t_cfg.get("optim", "paged_adamw_32bit"),
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
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=config["model"].get("max_seq_length", 2048),
        dataset_text_field=None,
        packing=False,
    )

    print(f"\nStarting Run 2 training → {output_dir}")
    trainer.train()

    print(f"\nSaving final checkpoint to {output_dir}/final...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("\nRun 2 training complete.")
    print("Next step: python llm/scripts/eval.py --checkpoint checkpoints/run2/final")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-Tuning Run 2 — Fraud Detection Accuracy")
    p.add_argument("--config", default="llm/configs/run2_config.yaml")
    p.add_argument("--from-checkpoint", default="checkpoints/run1/final",
                   help="Run 1 checkpoint to continue from")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    train(config, from_checkpoint=args.from_checkpoint, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
