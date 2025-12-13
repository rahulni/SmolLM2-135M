import os
import sys
import contextlib
import warnings

import torch
from transformers import LlamaForCausalLM, GPT2TokenizerFast, LlamaConfig
from torch.optim import AdamW
from accelerate import Accelerator

from config import SmolLMConfig  # <-- uses your generated config.py


# ----------------------------------------
# Optional: set matmul precision for better GPU perf
# ----------------------------------------
if torch.cuda.is_available():
    try:
        # "high" usually enables TF32 on Ampere+ GPUs, speeding up matmul
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ----------------------------------------
# Helper: build HF LlamaConfig from SmolLMConfig
# ----------------------------------------
def build_hf_config(sm_cfg: SmolLMConfig) -> LlamaConfig:
    """
    Convert your simple SmolLMConfig into a transformers.LlamaConfig,
    which LlamaForCausalLM expects.
    """
    kwargs = {
        "vocab_size": sm_cfg.vocab_size,
        "hidden_size": sm_cfg.hidden_size,
        "intermediate_size": sm_cfg.intermediate_size,
        "num_hidden_layers": sm_cfg.num_hidden_layers,
        "num_attention_heads": sm_cfg.num_attention_heads,
        "num_key_value_heads": sm_cfg.num_key_value_heads,
        "max_position_embeddings": sm_cfg.max_position_embeddings,
        "rms_norm_eps": sm_cfg.rms_norm_eps,
        "rope_theta": sm_cfg.rope_theta,
        "bos_token_id": sm_cfg.bos_token_id,
        "eos_token_id": sm_cfg.eos_token_id,
        "pad_token_id": sm_cfg.pad_token_id,
        "tie_word_embeddings": sm_cfg.tie_word_embeddings,
    }

    # rope_scaling can be None
    if sm_cfg.rope_scaling is not None:
        kwargs["rope_scaling"] = sm_cfg.rope_scaling

    return LlamaConfig(**kwargs)


# ----------------------------------------
# Helper: count and display model parameters
# ----------------------------------------
def count_parameters(model):
    """
    Count total, trainable, and non-trainable parameters in the model.
    Returns a formatted string with parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    def format_number(num):
        """Format number with commas and appropriate unit."""
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return str(num)
    
    return (
        f"Model Parameters:\n"
        f"  Total: {total_params:,} ({format_number(total_params)})\n"
        f"  Trainable: {trainable_params:,} ({format_number(trainable_params)})\n"
        f"  Non-trainable: {non_trainable_params:,} ({format_number(non_trainable_params)})"
    )


# ----------------------------------------
# Load and prepare text dataset
# ----------------------------------------
def load_dataset(file_path, tokenizer, seq_len=256):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # tokenize whole dataset
    # Temporarily increase model_max_length to avoid warning (we chunk anyway)
    original_max_length = getattr(tokenizer, 'model_max_length', None)
    try:
        # Set a very high value to avoid the warning (1 million tokens should be enough)
        tokenizer.model_max_length = 1_000_000
        # Suppress any remaining warnings during tokenization
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ids = tokenizer(raw, return_tensors="pt", truncation=False)["input_ids"][0]
    finally:
        # Restore original value if it existed
        if original_max_length is not None:
            tokenizer.model_max_length = original_max_length

    # chunk into sequences (seq_len is already a power of 2 = 256)
    chunks = []
    # ensure all chunks are exactly seq_len for nice power-of-2 shapes
    max_idx = (len(ids) // seq_len) * seq_len
    ids = ids[:max_idx]

    for i in range(0, len(ids), seq_len):
        chunk = ids[i:i + seq_len]
        if len(chunk) == seq_len:
            chunks.append(chunk)

    print(f"Loaded {len(chunks)} training chunks.")
    return chunks


def main():
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using device: {device}")
    # -----------------------------
    # Build model from scratch
    # -----------------------------
    sm_cfg = SmolLMConfig()
    hf_config = build_hf_config(sm_cfg)

    # Try to enable SDPA/flash-style attention (PyTorch scaled_dot_product_attention).
    # This uses faster kernels (flash-attn) when available.
    try:
        model = LlamaForCausalLM(hf_config, attn_implementation="sdpa")
    except TypeError:
        # Older transformers don't support attn_implementation kwarg
        model = LlamaForCausalLM(hf_config)

    # Display parameter count
    if accelerator.is_main_process:
        accelerator.print("\n" + "="*50)
        accelerator.print(count_parameters(model))
        accelerator.print("="*50 + "\n")

    # tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    # Set pad_token to eos_token for training, but use a distinct token_id for generation
    tokenizer.pad_token = tokenizer.eos_token
    # For generation, we'll explicitly set pad_token_id to avoid warnings

    # -----------------------------
    # Load input.txt dataset
    # -----------------------------
    seq_len = 256  # already power-of-2
    train_chunks = load_dataset("input.txt", tokenizer, seq_len)

    optimizer = AdamW(model.parameters(), lr=2e-4)

    # prepare for accelerate
    model, optimizer = accelerator.prepare(model, optimizer)

    model.train()
    global_step = 0
    max_steps = 5000
    print_every = 500

    num_chunks = len(train_chunks)

    # ----------------------------------------
    # Autocast setup (Linux CUDA only)
    # ----------------------------------------
    use_amp = (device.type == "cuda") and (sys.platform != "win32")
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None  # unused

    while global_step < max_steps:
        idx = global_step % num_chunks
        ids = train_chunks[idx].to(device)

        input_ids = ids.unsqueeze(0)  # (1, seq_len)
        labels = input_ids.clone()

        # Forward pass with optional autocast
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if global_step % print_every == 0:
            accelerator.print(f"Step {global_step} | Loss {loss.item():.4f}")

            # small generation test (no autocast needed)
            model.eval()
            prompt_text = "CORIOLANUS:"
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            
            # Create attention mask to avoid warnings
            attention_mask = torch.ones_like(prompt_ids)

            gen = accelerator.unwrap_model(model).generate(
                prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            # accelerator.print(tokenizer.decode(gen[0]))
            model.train()

        global_step += 1

    # -----------------------------
    # Save model + optimizer
    # -----------------------------
    save_dir = "checkpoint_5000"
    model_to_save = accelerator.unwrap_model(model)
    
    # Convert to FP16 before saving to reduce checkpoint size (~50% reduction)
    # This makes the checkpoint ~257MB instead of ~513MB
    if device.type == "cuda":
        # Save in FP16 for smaller size (works on GPU)
        model_to_save = model_to_save.half()
        accelerator.print("Saving model in FP16 format (reduces size by ~50%)")
    else:
        # CPU: keep FP32 (FP16 not well supported on CPU)
        accelerator.print("Saving model in FP32 format (CPU)")
    
    model_to_save.save_pretrained(save_dir)

    if accelerator.is_main_process:
        torch.save(
            {
                "step": global_step,
                "optimizer": optimizer.state_dict(),
            },
            f"{save_dir}/optim.pt",
        )

    accelerator.print("Training complete. Checkpoint saved.")


if __name__ == "__main__":
    main()
