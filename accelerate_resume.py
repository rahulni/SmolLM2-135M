import sys
import contextlib

import torch
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import LlamaForCausalLM

from config import SmolLMConfig  # <-- new config class


# ----------------------------------------
# Optional: set matmul precision for better GPU perf
# ----------------------------------------
if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def main():
    accelerator = Accelerator()
    device = accelerator.device

    # -----------------------------
    # Load model from checkpoint
    # -----------------------------
    # This will load both weights and the original HF config from checkpoint_5000
    model = LlamaForCausalLM.from_pretrained("checkpoint_5000")

    # -----------------------------
    # Load optimizer
    # -----------------------------
    optimizer = AdamW(model.parameters(), lr=2e-4)

    state = torch.load("checkpoint_5000/optim.pt", map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    start_step = state["step"]

    accelerator.print(f"Resuming from step {start_step}")

    # Prepare for accelerate (DDP/Multi-GPU, etc.)
    model, optimizer = accelerator.prepare(model, optimizer)

    model.train()

    # -----------------------------
    # Train 50 extra steps
    # -----------------------------
    sm_cfg = SmolLMConfig()

    # Autocast only on Linux CUDA, skip on Windows
    use_amp = (device.type == "cuda") and (sys.platform != "win32")
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None

    for t in range(50):
        # Simple synthetic batch (replace with real loader)
        input_ids = torch.randint(
            0, sm_cfg.vocab_size, (2, 32), device=device
        )
        labels = input_ids.clone()

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

        if t % 10 == 0:
            accelerator.print(f"Extra step {t} | Loss: {loss.item():.4f}")

    accelerator.print("Resume training complete.")


if __name__ == "__main__":
    main()
