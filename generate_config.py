"""
generate_config_py.py

This script downloads the HF config for SmolLM2-135M
and generates a config.py file that contains a Python
class with all required architectural hyperparameters.

Use this when writing a model-from-scratch implementation.

Requires:
    pip install transformers
"""

from transformers import AutoConfig
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
OUTPUT_FILE = "config.py"


def extract_config_fields(cfg):
    """
    Extract only essential architecture parameters needed
    to build the model from scratch.
    """
    fields = {
        "model_type": cfg.model_type,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": getattr(cfg, "rope_theta", 10000.0),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", False),
    }
    return fields


def write_config_py(fields):
    """
    Create config.py defining SmolLMConfig class.
    """

    lines = []
    lines.append("# Auto-generated config for SmolLM2-135M")
    lines.append("class SmolLMConfig:")
    lines.append("    def __init__(self):")

    for k, v in fields.items():
        lines.append(f"        self.{k} = {repr(v)}")

    lines.append("")
    lines.append("    def __repr__(self):")
    lines.append("        return f\"SmolLMConfig({vars(self)})\"")

    Path(OUTPUT_FILE).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Created {OUTPUT_FILE}")


def main():
    print("[INFO] Loading HF config...")
    cfg = AutoConfig.from_pretrained(MODEL_ID)

    print("[INFO] Extracting relevant fields...")
    fields = extract_config_fields(cfg)

    print("[INFO] Writing config.py...")
    write_config_py(fields)

    print("[DONE] Your config.py is ready. You can now import it when building the model.")


if __name__ == "__main__":
    main()
