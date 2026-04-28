#!/usr/bin/env python3
"""
Pre-training validation suite.

Run this BEFORE launching training on the GPU server to verify:
  1.  Data generation  — subset tags present, schema valid
  2.  Subset filtering — Run 1 and Run 2 filters are disjoint and non-empty
  3.  Class balance    — underfitting risk (too few examples per subset)
  4.  Duplicate rate   — overfitting risk (near-identical examples)
  5.  Config sanity    — required keys, LR range, effective batch size
  6.  File readiness   — train/val/test.jsonl exist with enough examples
  7.  Decision balance — allow / review / block all sufficiently represented

Usage
-----
# Full suite (reads existing data files + generates a mini dataset)
python llm/scripts/test_system.py

# Skip slow file checks (fast mode, no file reads)
python llm/scripts/test_system.py --no-file-checks

# Point at a different data dir
python llm/scripts/test_system.py --data-dir llm/data
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths — resolved relative to this script so it can be called from anywhere
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "llm" / "configs"
DATA_DIR_DEFAULT = PROJECT_ROOT / "llm" / "data"
PROMPT_FILE = PROJECT_ROOT / "llm" / "prompts" / "system_prompt.txt"

# All subset names produced by prepare_data.generate_dataset
ALL_SUBSETS = {
    "benign_examples",
    "fraud_intent",
    "prompt_injection",
    "obfuscation_evasion",
    "data_exfiltration",
    "context_deviation",
    "unauthorized_action",
    "url_domain_risk",
    "authority_spoof",
    "review_mix",
    "adversarial_resistance",
    "rag_context",
    "adversarial_suffix",
    "payload_splitting",
    "indirect_html_injection",
    "language_switching",
    "obfuscation_variant",
    "fake_task_completion",
    "memory_poisoning",
    "url_param_injection",
}

# Run 1 resilience subsets (must be disjoint with RUN2_SUBSETS except benign)
RUN1_SUBSETS = {
    "prompt_injection",
    "obfuscation_evasion",
    "adversarial_resistance",
    "indirect_html_injection",
    "language_switching",
    "obfuscation_variant",
    "fake_task_completion",
    "memory_poisoning",
    "adversarial_suffix",
    "payload_splitting",
    "url_param_injection",
    "rag_context",
    "benign_examples",
}

# Run 2 fraud-detection subsets
RUN2_SUBSETS = {
    "fraud_intent",
    "context_deviation",
    "data_exfiltration",
    "unauthorized_action",
    "url_domain_risk",
    "authority_spoof",
    "review_mix",
    "benign_examples",
}

FRAUD_PARAMETERS = [
    "url_domain_risk", "fraud_intent", "prompt_injection",
    "context_deviation", "data_exfiltration", "obfuscation_evasion",
    "unauthorized_action", "authority_spoof",
]

# Minimum training examples per subset (below this → underfitting risk)
MIN_EXAMPLES_PER_SUBSET = 50
# Minimum fraction of distinct user messages (below this → overfitting risk)
MIN_UNIQUE_RATIO = 0.80
# Minimum fraction for each decision class in the overall training set
MIN_DECISION_FRACTION = 0.05
# Minimum total examples in train split
MIN_TRAIN_EXAMPLES = 3_000
MIN_VAL_EXAMPLES = 300

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class TestResult:
    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []  # (name, reason)
        self.warnings: List[Tuple[str, str]] = []

    def ok(self, name: str, detail: str = "") -> None:
        msg = f"[PASS] {name}"
        if detail:
            msg += f" — {detail}"
        self.passed.append(msg)
        print(msg)

    def fail(self, name: str, reason: str) -> None:
        msg = f"[FAIL] {name} — {reason}"
        self.failed.append((name, reason))
        print(msg)

    def warn(self, name: str, reason: str) -> None:
        msg = f"[WARN] {name} — {reason}"
        self.warnings.append((name, reason))
        print(msg)

    def summary(self) -> int:
        total = len(self.passed) + len(self.failed)
        print("\n" + "=" * 70)
        print(f"SUMMARY: {len(self.passed)}/{total} passed, "
              f"{len(self.failed)} failed, {len(self.warnings)} warnings")
        if self.failed:
            print("\nFailed checks (fix before training):")
            for name, reason in self.failed:
                print(f"  ✗ {name}: {reason}")
        if self.warnings:
            print("\nWarnings (review but not blocking):")
            for name, reason in self.warnings:
                print(f"  ! {name}: {reason}")
        if not self.failed:
            print("\n✓ All checks passed — safe to start training.")
        else:
            print("\n✗ Fix failures above before launching training.")
        print("=" * 70)
        return 1 if self.failed else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        return None
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    if "_base" in cfg:
        base_path = config_path.parent / cfg.pop("_base")
        with open(base_path) as fh:
            base = yaml.safe_load(fh)
        return _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _filter_by_subsets(
    dataset: List[Dict[str, Any]], subsets: set
) -> List[Dict[str, Any]]:
    return [ex for ex in dataset if ex.get("metadata", {}).get("subset") in subsets]


# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

def check_data_generation(r: TestResult) -> List[Dict[str, Any]]:
    """Generate a small synthetic dataset and verify basic properties."""
    print("\n── Data Generation ──────────────────────────────────────────")
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import prepare_data as pd_mod  # type: ignore
    except ImportError as e:
        r.fail("import_prepare_data", str(e))
        return []

    try:
        dataset = pd_mod.generate_dataset(examples_per_class=5)
    except Exception as e:
        r.fail("generate_dataset", f"raised {type(e).__name__}: {e}")
        return []

    if not dataset:
        r.fail("generate_dataset_nonempty", "returned empty list")
        return []

    r.ok("generate_dataset", f"{len(dataset)} examples generated")

    # All examples have subset field
    missing_subset = [
        i for i, ex in enumerate(dataset)
        if not ex.get("metadata", {}).get("subset")
    ]
    if missing_subset:
        r.fail("subset_field_present",
               f"{len(missing_subset)}/{len(dataset)} examples missing metadata['subset']")
    else:
        r.ok("subset_field_present", "all examples have a non-empty subset tag")

    # All subset values are known
    found_subsets = {ex["metadata"]["subset"] for ex in dataset}
    unknown = found_subsets - ALL_SUBSETS
    if unknown:
        r.fail("known_subset_values",
               f"unrecognised subset names: {sorted(unknown)}")
    else:
        r.ok("known_subset_values",
             f"{len(found_subsets)} distinct subsets: {sorted(found_subsets)}")

    # All 20 subsets produced
    missing_gen = ALL_SUBSETS - found_subsets
    if missing_gen:
        r.warn("all_subsets_generated",
               f"subsets not produced with count=5: {sorted(missing_gen)}")
    else:
        r.ok("all_subsets_generated", "all 20 subsets present")

    return dataset


# ---------------------------------------------------------------------------
# 2. Schema validation
# ---------------------------------------------------------------------------

def check_schema(r: TestResult, dataset: List[Dict[str, Any]]) -> None:
    print("\n── Schema Validation ────────────────────────────────────────")
    if not dataset:
        r.warn("schema_check", "skipped — no dataset to validate")
        return

    errors: List[str] = []
    for i, ex in enumerate(dataset):
        msgs = ex.get("messages", [])
        if len(msgs) != 3:
            errors.append(f"[{i}] expected 3 messages, got {len(msgs)}")
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"[{i}] wrong roles: {roles}")
            continue

        # assistant content must be valid JSON
        try:
            parsed = json.loads(msgs[2]["content"])
        except json.JSONDecodeError as e:
            errors.append(f"[{i}] assistant not valid JSON: {e}")
            continue

        # all 8 parameters present with score in [0, 1]
        for param in FRAUD_PARAMETERS:
            if param not in parsed:
                errors.append(f"[{i}] missing parameter '{param}'")
                break
            score = parsed[param].get("score")
            if score is None or not (0.0 <= score <= 1.0):
                errors.append(f"[{i}] {param}.score out of range: {score}")
                break
        else:
            if parsed.get("decision") not in ("allow", "review", "block"):
                errors.append(
                    f"[{i}] invalid decision: {parsed.get('decision')}")
            urs = parsed.get("unified_risk_score")
            if urs is None or not (0.0 <= urs <= 1.0):
                errors.append(f"[{i}] unified_risk_score out of range: {urs}")

        # metadata
        meta = ex.get("metadata", {})
        if meta.get("decision") not in ("allow", "review", "block"):
            errors.append(
                f"[{i}] metadata.decision invalid: {meta.get('decision')}")

    if errors:
        shown = errors[:5]
        r.fail("schema_valid",
               f"{len(errors)} errors. First {len(shown)}: {shown}")
    else:
        r.ok("schema_valid", f"all {len(dataset)} examples pass schema check")


# ---------------------------------------------------------------------------
# 3. Subset filtering
# ---------------------------------------------------------------------------

def check_filtering(r: TestResult, dataset: List[Dict[str, Any]]) -> None:
    print("\n── Subset Filtering ─────────────────────────────────────────")
    if not dataset:
        r.warn("filter_check", "skipped — no dataset")
        return

    # Run 1
    run1_data = _filter_by_subsets(dataset, RUN1_SUBSETS)
    run1_found = {ex["metadata"]["subset"] for ex in run1_data}
    if not run1_data:
        r.fail("run1_filter_nonempty", "Run 1 filter returned 0 examples")
    else:
        r.ok("run1_filter_nonempty",
             f"{len(run1_data)} examples | subsets: {sorted(run1_found)}")

    rogue_run1 = run1_found - RUN1_SUBSETS
    if rogue_run1:
        r.fail("run1_filter_clean",
               f"unexpected subsets leaked into Run 1: {rogue_run1}")
    else:
        r.ok("run1_filter_clean", "no unexpected subsets in Run 1 filtered set")

    # Run 2
    run2_data = _filter_by_subsets(dataset, RUN2_SUBSETS)
    run2_found = {ex["metadata"]["subset"] for ex in run2_data}
    if not run2_data:
        r.fail("run2_filter_nonempty", "Run 2 filter returned 0 examples")
    else:
        r.ok("run2_filter_nonempty",
             f"{len(run2_data)} examples | subsets: {sorted(run2_found)}")

    rogue_run2 = run2_found - RUN2_SUBSETS
    if rogue_run2:
        r.fail("run2_filter_clean",
               f"unexpected subsets leaked into Run 2: {rogue_run2}")
    else:
        r.ok("run2_filter_clean", "no unexpected subsets in Run 2 filtered set")

    # Disjoint check (only benign_examples is shared by design)
    overlap = (RUN1_SUBSETS & RUN2_SUBSETS) - {"benign_examples"}
    if overlap:
        r.fail("run1_run2_disjoint",
               f"unexpected overlap between runs: {sorted(overlap)}")
    else:
        r.ok("run1_run2_disjoint",
             "Run 1 ∩ Run 2 = {benign_examples} only (by design)")


# ---------------------------------------------------------------------------
# 4 & 5. Overfitting / underfitting — checks on existing data files
# ---------------------------------------------------------------------------

def check_class_balance(r: TestResult, data: List[Dict[str, Any]],
                        label: str) -> None:
    """Check subset example counts (underfitting) and duplicate rate (overfitting)."""
    print(f"\n── Class Balance & Duplicates ({label}) ──────────────────────")

    # Per-subset counts
    counts: Dict[str, int] = {}
    for ex in data:
        s = ex.get("metadata", {}).get("subset", "<missing>")
        counts[s] = counts.get(s, 0) + 1

    low_count = {s: c for s, c in counts.items() if c <
                 MIN_EXAMPLES_PER_SUBSET}
    if low_count:
        r.fail(f"min_examples_per_subset_{label}",
               f"subsets below {MIN_EXAMPLES_PER_SUBSET} examples: "
               + ", ".join(f"{s}={c}" for s, c in sorted(low_count.items())))
    else:
        min_c = min(counts.values()) if counts else 0
        r.ok(f"min_examples_per_subset_{label}",
             f"all subsets ≥ {MIN_EXAMPLES_PER_SUBSET} (min={min_c})")

    # Class imbalance ratio
    if counts:
        max_c = max(counts.values())
        min_c = min(counts.values())
        ratio = max_c / max(min_c, 1)
        if ratio > 20:
            r.warn(f"class_imbalance_{label}",
                   f"max/min subset ratio = {ratio:.0f}x "
                   f"(max={max_c}, min={min_c})")
        else:
            r.ok(f"class_imbalance_{label}",
                 f"max/min ratio = {ratio:.1f}x — acceptable")

    # Decision distribution
    decisions: Dict[str, int] = {}
    for ex in data:
        d = ex.get("metadata", {}).get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1
    total = len(data)
    low_decision = {
        d: c for d, c in decisions.items()
        if c / total < MIN_DECISION_FRACTION and d != "unknown"
    }
    if low_decision:
        r.fail(f"decision_distribution_{label}",
               "decision class below 5% of total: "
               + ", ".join(f"{d}={c}({c/total:.1%})"
                           for d, c in sorted(low_decision.items())))
    else:
        dist_str = " | ".join(
            f"{d}={c}({c/total:.1%})" for d, c in sorted(decisions.items()))
        r.ok(f"decision_distribution_{label}", dist_str)

    # Duplicate check
    user_messages = [
        ex["messages"][1]["content"]
        for ex in data
        if len(ex.get("messages", [])) > 1
    ]
    unique_count = len(set(user_messages))
    unique_ratio = unique_count / max(len(user_messages), 1)
    if unique_ratio < MIN_UNIQUE_RATIO:
        r.fail(f"duplicate_rate_{label}",
               f"only {unique_ratio:.1%} unique user messages "
               f"({unique_count}/{len(user_messages)}) — overfitting risk")
    else:
        r.ok(f"duplicate_rate_{label}",
             f"{unique_ratio:.1%} unique user messages "
             f"({unique_count}/{len(user_messages)})")


# ---------------------------------------------------------------------------
# 6. Config sanity
# ---------------------------------------------------------------------------

def check_configs(r: TestResult) -> None:
    print("\n── Config Sanity ────────────────────────────────────────────")

    required_top_keys = ["model", "training", "lora", "data", "fsdp"]

    for config_name in ("run1_config.yaml", "run2_config.yaml"):
        path = CONFIGS_DIR / config_name
        if not path.exists():
            r.fail(f"config_exists_{config_name}", f"{path} not found")
            continue

        try:
            cfg = _load_config(path)
        except Exception as e:
            r.fail(f"config_loads_{config_name}", str(e))
            continue

        if cfg is None:
            r.warn(f"config_loads_{config_name}",
                   "pyyaml not installed — config checks skipped "
                   "(run: pip install pyyaml)")
            continue

        r.ok(f"config_loads_{config_name}", "parsed successfully")

        # Required keys
        missing_keys = [k for k in required_top_keys if k not in cfg]
        if missing_keys:
            r.fail(f"config_required_keys_{config_name}",
                   f"missing keys: {missing_keys}")
        else:
            r.ok(f"config_required_keys_{config_name}",
                 "all required keys present")

        t = cfg.get("training", {})

        # Learning rate
        lr = float(t.get("learning_rate", 0))
        if not (1e-6 <= lr <= 5e-4):
            r.fail(f"learning_rate_{config_name}",
                   f"LR={lr:.2e} outside safe range [1e-6, 5e-4]")
        else:
            r.ok(f"learning_rate_{config_name}", f"LR={lr:.2e}")

        # Effective batch size
        per_device = t.get("per_device_train_batch_size", 1)
        grad_accum = t.get("gradient_accumulation_steps", 1)
        effective = per_device * grad_accum
        if not (8 <= effective <= 64):
            r.warn(f"effective_batch_{config_name}",
                   f"effective batch = {per_device}×{grad_accum} = {effective} "
                   f"(expected 8–64)")
        else:
            r.ok(f"effective_batch_{config_name}",
                 f"per_device={per_device} × grad_accum={grad_accum} = {effective}")

        # Epochs
        epochs = t.get("num_train_epochs", 0)
        if not (1 <= epochs <= 10):
            r.warn(f"epochs_{config_name}",
                   f"num_train_epochs={epochs} — unusual value")
        else:
            r.ok(f"epochs_{config_name}", f"num_train_epochs={epochs}")

        # LoRA
        lora = cfg.get("lora", {})
        if lora.get("r", 0) not in range(4, 129):
            r.warn(f"lora_r_{config_name}", f"lora.r={lora.get('r')} unusual")
        else:
            r.ok(f"lora_r_{config_name}",
                 f"r={lora.get('r')}, alpha={lora.get('lora_alpha')}")

        # data_subsets configured
        subsets_cfg = cfg.get("data_subsets", [])
        if not subsets_cfg:
            r.fail(f"data_subsets_set_{config_name}",
                   "data_subsets is empty — filter will use ALL data")
        else:
            unknown_cfg = set(subsets_cfg) - ALL_SUBSETS
            if unknown_cfg:
                r.fail(f"data_subsets_valid_{config_name}",
                       f"unrecognised subset names: {sorted(unknown_cfg)}")
            else:
                r.ok(f"data_subsets_valid_{config_name}",
                     f"{len(subsets_cfg)} subsets configured: {subsets_cfg}")

        # fp16/bf16 mutual exclusion
        if t.get("fp16") and t.get("bf16"):
            r.fail(f"fp16_bf16_{config_name}", "both fp16 and bf16 are True")
        else:
            r.ok(f"fp16_bf16_{config_name}",
                 f"fp16={t.get('fp16', False)}, bf16={t.get('bf16', False)}")


# ---------------------------------------------------------------------------
# 7. System prompt
# ---------------------------------------------------------------------------

def check_system_prompt(r: TestResult) -> None:
    print("\n── System Prompt ────────────────────────────────────────────")
    if not PROMPT_FILE.exists():
        r.fail("system_prompt_exists", f"{PROMPT_FILE} not found")
        return
    text = PROMPT_FILE.read_text(encoding="utf-8").strip()
    if len(text) < 200:
        r.fail("system_prompt_length",
               f"only {len(text)} chars — looks like a placeholder")
        return
    r.ok("system_prompt_exists", f"{len(text)} chars")
    if "[[SYSTEM_PROMPT]]" in text or "placeholder" in text.lower():
        r.fail("system_prompt_no_placeholder",
               "still contains placeholder text")
    else:
        r.ok("system_prompt_no_placeholder", "no placeholder found")
    if "JSON" not in text and "json" not in text:
        r.warn("system_prompt_mentions_json",
               "system prompt may not instruct JSON-only output")
    else:
        r.ok("system_prompt_mentions_json", "JSON output instruction found")


# ---------------------------------------------------------------------------
# 8. Data file readiness
# ---------------------------------------------------------------------------

def check_data_files(r: TestResult, data_dir: Path) -> None:
    print("\n── Data File Readiness ──────────────────────────────────────")

    for split, min_count in [("train", MIN_TRAIN_EXAMPLES),
                             ("val", MIN_VAL_EXAMPLES),
                             ("test", 100)]:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            r.fail(f"file_exists_{split}",
                   f"{path} not found — run prepare_data.py")
            continue
        r.ok(f"file_exists_{split}", str(path))

        try:
            data = _load_jsonl(path)
        except Exception as e:
            r.fail(f"file_readable_{split}", str(e))
            continue

        if len(data) < min_count:
            r.fail(f"file_count_{split}",
                   f"{len(data)} examples < minimum {min_count}")
        else:
            r.ok(f"file_count_{split}", f"{len(data):,} examples")

        # All examples must have non-empty subset tag
        missing = sum(
            1 for ex in data
            if not ex.get("metadata", {}).get("subset")
        )
        if missing:
            r.fail(f"file_subset_tagged_{split}",
                   f"{missing}/{len(data)} examples missing subset tag "
                   f"— regenerate with updated prepare_data.py")
        else:
            subsets_found = {ex["metadata"]["subset"] for ex in data}
            r.ok(f"file_subset_tagged_{split}",
                 f"all tagged | {len(subsets_found)} subsets present")

        if split == "train":
            check_class_balance(r, data, "train")

        # Subset coverage — all run1+run2 subsets must be present in train
        if split == "train":
            all_needed = RUN1_SUBSETS | RUN2_SUBSETS
            subsets_in_file = {
                ex.get("metadata", {}).get("subset") for ex in data}
            missing_s = all_needed - subsets_in_file
            if missing_s:
                r.fail("train_subset_coverage",
                       f"subsets missing from train.jsonl: {sorted(missing_s)}")
            else:
                r.ok("train_subset_coverage",
                     f"all {len(all_needed)} required subsets present in train split")

            # Run 1 filter gives non-trivial training set
            run1_filtered = _filter_by_subsets(data, RUN1_SUBSETS)
            run2_filtered = _filter_by_subsets(data, RUN2_SUBSETS)
            if len(run1_filtered) < 200:
                r.fail("run1_filtered_count",
                       f"only {len(run1_filtered)} examples pass Run 1 filter")
            else:
                r.ok("run1_filtered_count",
                     f"{len(run1_filtered):,} examples for Run 1 training")
            if len(run2_filtered) < 200:
                r.fail("run2_filtered_count",
                       f"only {len(run2_filtered)} examples pass Run 2 filter")
            else:
                r.ok("run2_filtered_count",
                     f"{len(run2_filtered):,} examples for Run 2 training")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-training validation suite for fraud-detection LLM")
    p.add_argument("--data-dir", default=str(DATA_DIR_DEFAULT),
                   help="Path to data directory with train/val/test.jsonl")
    p.add_argument("--no-file-checks", action="store_true",
                   help="Skip checks that read existing data files")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("  Fraud-Detection LLM — Pre-Training Validation Suite")
    print("=" * 70)

    r = TestResult()

    # 1 + 2. Generate mini dataset, check schema
    mini_dataset = check_data_generation(r)
    check_schema(r, mini_dataset)

    # 3. Filtering
    check_filtering(r, mini_dataset)

    # 4. Configs
    check_configs(r)

    # 5. System prompt
    check_system_prompt(r)

    # 6 + 7. Existing data files (underfitting/overfitting on real data)
    if not args.no_file_checks:
        check_data_files(r, data_dir)
    else:
        print("\n── Data File Checks (skipped via --no-file-checks) ──────────")

    sys.exit(r.summary())


if __name__ == "__main__":
    main()
