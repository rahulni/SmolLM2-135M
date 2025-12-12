#!/usr/bin/env python
"""
smollm2_135m_reverse_engineer.py

Utility script to:
  - Download & inspect HuggingFaceTB/SmolLM2-135M
  - "Reverse engineer" its architecture from the HF config
  - Optionally load the full model and compute parameter stats
  - Export a training-style YAML config skeleton

Usage examples:
  python smollm2_135m_reverse_engineer.py
  python smollm2_135m_reverse_engineer.py --model-id HuggingFaceTB/SmolLM2-135M-Instruct
  python smollm2_135m_reverse_engineer.py --load-model --dtype bf16

Requires:
  pip install transformers torch pyyaml
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import yaml


def load_config_and_tokenizer(
    model_id: str,
) -> Tuple[AutoConfig, "PreTrainedTokenizer"]:
    print(f"[INFO] Loading config for {model_id} ...")
    config = AutoConfig.from_pretrained(model_id)
    print(f"[INFO] Loading tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return config, tokenizer


def maybe_load_model(
    model_id: str, dtype: str = "auto", device: str = "auto"
) -> Optional[torch.nn.Module]:
    print(f"[INFO] Loading model weights for {model_id} ...")

    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None  # let HF decide

    kwargs = {}
    if device == "auto":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    print("[INFO] Model loaded.")
    return model


def summarize_architecture(config: AutoConfig) -> str:
    """
    Build a human-readable summary of the architecture from the HF config.
    This is the first step of "reverse engineering": extracting all
    architecturally relevant hyperparameters.
    """
    fields = {}

    # Common LLaMA-style fields (SmolLM2 is "llama"-family)
    for attr in [
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "rope_scaling",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    ]:
        if hasattr(config, attr):
            fields[attr] = getattr(config, attr)

    lines = []
    lines.append("=== SmolLM2-135M Architecture Summary (from HF config) ===")
    for k, v in fields.items():
        lines.append(f"{k:24s}: {v}")

    # A few derived quantities:
    if hasattr(config, "hidden_size") and hasattr(config, "num_hidden_layers"):
        total_mlp_params_approx = (
            2 * config.intermediate_size * config.hidden_size
            * config.num_hidden_layers
        )
        lines.append("")
        lines.append("=== Derived / Approximate Quantities ===")
        lines.append(
            f"Approx MLP params (2 * d_model * d_ff * n_layers): ~{total_mlp_params_approx:,}"
        )

    return "\n".join(lines)


def export_config_to_files(
    config: AutoConfig,
    output_dir: Path,
    base_name: str = "smollm2_135m_config",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.to_dict()

    json_path = output_dir / f"{base_name}.json"
    yaml_path = output_dir / f"{base_name}.yaml"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    print(f"[INFO] Saved raw HF config to: {json_path}")
    print(f"[INFO] Saved raw HF config to: {yaml_path}")


def build_training_style_yaml(config: AutoConfig) -> dict:
    """
    Construct a *training-style* YAML skeleton using key hyperparams.
    This is useful if you want to re-implement / train SmolLM2-135M
    from scratch in your own code or a framework like Nanotron.

    NOTE: This does NOT recover optimizer, LR schedule, or dataset;
    you still have to plug in those from the paper/model card.
    """
    d = {}

    # Model section: what you need to build the architecture
    d["model"] = {
        "architecture": "transformer_decoder",
        "base_model_type": config.model_type,
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "intermediate_size": int(getattr(config, "intermediate_size", 0)),
        "num_layers": int(getattr(config, "num_hidden_layers", 0)),
        "num_attention_heads": int(getattr(config, "num_attention_heads", 0)),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", 0)),
        "vocab_size": int(getattr(config, "vocab_size", 0)),
        "max_position_embeddings": int(
            getattr(config, "max_position_embeddings", 0)
        ),
        "rms_norm_eps": float(getattr(config, "rms_norm_eps", 1e-5)),
        "rope_theta": float(getattr(config, "rope_theta", 1e4)),
        "rope_scaling": getattr(config, "rope_scaling", None),
        "tie_word_embeddings": bool(
            getattr(config, "tie_word_embeddings", False)
        ),
    }

    # Token IDs
    d["tokens"] = {
        "bos_token_id": getattr(config, "bos_token_id", None),
        "eos_token_id": getattr(config, "eos_token_id", None),
        "pad_token_id": getattr(config, "pad_token_id", None),
    }

    # Training hyperparams: placeholders
    d["training"] = {
        "total_tokens": 2_000_000_000_000,
        "global_batch_size_tokens": 1_000_000,
        "sequence_length": int(getattr(config, "max_position_embeddings", 0)),
        "precision": "bf16",
        "optimizer": {
            "type": "adamw",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
        },
        "lr_schedule": {
            "type": "cosine",
            "max_lr": 3e-4,
            "min_lr": 3e-5,
            "warmup_ratio": 0.01,
        },
    }

    # Datasets: placeholder
    d["data"] = {
        "train_datasets": [
            {"name": "fineweb-edu", "sampling_ratio": 0.25},
            {"name": "dclm", "sampling_ratio": 0.25},
            {"name": "the-stack", "sampling_ratio": 0.25},
            {"name": "other-filtered-data", "sampling_ratio": 0.25},
        ],
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
    }

    return d


def export_training_yaml(
    training_yaml: dict,
    output_dir: Path,
    file_name: str = "smollm2_135m_training_skeleton.yaml",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file_name
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(training_yaml, f, sort_keys=False)
    print(f"[INFO] Saved training-style YAML skeleton to: {out_path}")


def compute_param_stats(model: torch.nn.Module) -> dict:
    """
    Compute total params and per-layer param counts.
    Helpful when you're implementing the architecture from scratch
    and want to confirm you've got the same parameterization.
    """
    print("[INFO] Computing parameter stats ...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    layer_prefix = "model.layers"
    layer_stats = {}

    for name, p in model.named_parameters():
        if name.startswith(layer_prefix):
            parts = name.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_idx = int(parts[2])
                layer_stats.setdefault(layer_idx, 0)
                layer_stats[layer_idx] += p.numel()

    layer_stats = dict(sorted(layer_stats.items(), key=lambda x: x[0]))

    stats = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "per_layer_params": {int(k): int(v) for k, v in layer_stats.items()},
    }
    return stats


def save_param_stats(stats: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "smollm2_135m_param_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Saved parameter stats to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and reverse-engineer SmolLM2-135M."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="HF model ID to inspect (base or instruct).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="smollm2_135m_reverse_engineered",
        help="Directory to store exported configs and stats.",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Actually load model weights and compute parameter stats.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16"],
        help="Torch dtype for loading model, if --load-model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device for model loading. "auto" uses HF accelerate mapping.',
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    config, tokenizer = load_config_and_tokenizer(args.model_id)

    arch_summary = summarize_architecture(config)
    print()
    print(arch_summary)
    print()

    export_config_to_files(config, output_dir, base_name="hf_config")

    training_yaml = build_training_style_yaml(config)
    export_training_yaml(training_yaml, output_dir)

    if args.load_model:
        model = maybe_load_model(
            args.model_id, dtype=args.dtype, device=args.device
        )
        stats = compute_param_stats(model)
        save_param_stats(stats, output_dir)

        print("\n=== Parameter Stats (summary) ===")
        print(f"Total params       : {stats['total_params']:,}")
        print(f"Trainable params   : {stats['trainable_params']:,}")
        example_layers = list(stats["per_layer_params"].items())[:3]
        print("First few layers param count:")
        for idx, n in example_layers:
            print(f"  Layer {idx:<3d}: {n:,}")


if __name__ == "__main__":
    main()
