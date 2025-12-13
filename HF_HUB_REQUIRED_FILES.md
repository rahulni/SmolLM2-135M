# Required Files for Hugging Face Hub

## ✅ REQUIRED Files (Minimum)

For `LlamaForCausalLM.from_pretrained()` to work, you need:

1. **`config.json`** - Model architecture configuration (REQUIRED)
   - Contains all model hyperparameters
   - Size: ~1KB

2. **`model.safetensors`** - Model weights (REQUIRED)
   - Contains all trained parameters
   - Size: ~257MB (FP16) or ~513MB (FP32)
   - Alternative: `model.bin` (older format, less safe)

## ✅ RECOMMENDED Files

3. **`generation_config.json`** - Generation settings (RECOMMENDED)
   - Contains default generation parameters
   - Size: <1KB
   - If missing, uses defaults from config.json

## ❌ NOT REQUIRED (Can Skip)

4. **`optim.pt`** - Optimizer state (NOT NEEDED for inference)
   - Only needed for resuming training
   - Size: ~200MB
   - **Exclude this to save space!**

5. **Tokenizer files** - NOT needed if using same tokenizer
   - Your app loads tokenizer from `HuggingFaceTB/SmolLM2-135M`
   - Only include if you have a custom tokenizer

## 📦 Summary

**Minimum upload (for inference):**
- `config.json` (~1KB)
- `model.safetensors` (~257MB FP16 or ~513MB FP32)
- `generation_config.json` (~1KB) - optional but recommended

**Total: ~257MB (FP16) or ~513MB (FP32)**

**Files to EXCLUDE:**
- `optim.pt` (~200MB) - not needed for inference

