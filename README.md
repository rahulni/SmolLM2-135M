# SmolLM2-135M Training Project

A complete training pipeline for SmolLM2-135M, a lightweight language model based on the LLaMA architecture with 135 million parameters. This model has been fine-tuned exclusively on ShakespeCare's **Coriolanus**, and therefore writes in the style of a dramatic play. This project includes optimized training scripts with multiple speedup techniques and a ready-to-deploy Gradio demo.

## 📊 Model Information

### Parameter Count
- **Total Parameters:** 134,515,008 (134.52M)
- **Trainable Parameters:** 134,515,008 (134.52M)
- **Non-trainable Parameters:** 0
- **Model Size:** ~270MB (FP16) / ~540MB (FP32)

### Specialization
This model has been fine-tuned exclusively on Shakespeare's **Coriolanus**. As a result, it generates text in the style of a dramatic play, including:
- Character names and dialogue
- Stage directions
- Shakespearean language and structure
- Dramatic formatting

### Model Architecture

The model follows the LLaMA (Large Language Model Meta AI) architecture with the following specifications:

| Component | Value |
|-----------|-------|
| **Model Type** | LLaMA (Decoder-only Transformer) |
| **Hidden Size** | 576 |
| **Intermediate Size** | 1,536 |
| **Number of Layers** | 30 |
| **Attention Heads** | 9 |
| **Key-Value Heads** | 3 (Grouped Query Attention) |
| **Vocabulary Size** | 49,152 |
| **Max Position Embeddings** | 8,192 |
| **RoPE Theta** | 100,000 |
| **RMSNorm Epsilon** | 1e-5 |

### Architecture Features

- **Grouped Query Attention (GQA)**: Uses 3 KV heads with 9 query heads for efficient attention computation
- **Rotary Position Embeddings (RoPE)**: Positional encoding with theta=100,000
- **RMSNorm**: Root Mean Square Layer Normalization
- **Tied Word Embeddings**: Input and output embeddings share weights
- **Flash Attention**: Uses PyTorch SDPA (Scaled Dot Product Attention) for faster training and inference

## 🚀 Speedup Optimizations

This project implements several performance optimizations:

1. **✅ Flash Attention (SDPA)**
   - Uses PyTorch's `scaled_dot_product_attention` with `attn_implementation="sdpa"`
   - Automatically uses flash-attention kernels when available
   - Significantly faster attention computation

2. **✅ Autocast (Mixed Precision)**
   - Automatic Mixed Precision (AMP) training
   - Uses bfloat16 on supported GPUs, falls back to float16
   - Reduces memory usage and speeds up training

3. **✅ Float32 Matmul Precision**
   - Sets `torch.set_float32_matmul_precision("high")`
   - Enables TF32 on Ampere+ GPUs (A100, RTX 30xx, etc.)
   - Faster matrix multiplications with minimal precision loss

4. **✅ Power of 2 Optimization**
   - Sequence length set to 256 (power of 2)
   - Ensures all chunks are exactly power-of-2 sized
   - Optimizes GPU memory alignment and computation efficiency

## 📁 Project Structure

```
.
├── accelerate_train.py                    # Main training script with Accelerate
├── accelerate_resume.py                   # Resume training from checkpoint
├── train_from_scratch.py                  # Simple training script (no Accelerate)
├── smollm2_135m_reverse_engineer.py       # Reverse engineer model architecture from HF
├── upload_to_hub.py                        # Upload checkpoint to HuggingFace Hub
├── convert_to_fp16.py                      # Convert FP32 checkpoint to FP16 (reduce size by 50%)
├── prepare_for_hf_space.py                 # Prepare minimal checkpoint (remove optimizer)
├── config.py                               # Model configuration class
├── generate_config.py                      # Config generation utility
├── app.py                                  # Gradio demo for Hugging Face Spaces
├── input.txt                               # Training dataset (Shakespeare's Coriolanus)
├── checkpoint_5000/                        # Saved model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── optim.pt
├── smollm2_135m_reverse_engineered/        # Output from reverse engineering script
│   ├── hf_config.json
│   ├── hf_config.yaml
│   └── smollm2_135m_training_skeleton.yaml
└── README.md                               # This file
```

## 🛠️ Installation

### Requirements

```bash
pip install torch transformers accelerate gradio pyyaml
```

**Note:** `pyyaml` is required for the reverse engineering script (`smollm2_135m_reverse_engineer.py`).

### Optional (for better performance)

```bash
# Flash Attention (if available)
pip install flash-attn --no-build-isolation

# Quantization support (for smaller model size in HF Spaces)
pip install bitsandbytes
```

**Note:** `bitsandbytes` is required for 8-bit/4-bit quantization in the Gradio app. This is useful for Hugging Face Spaces where model size is limited.

## 📖 Usage

### 0. Reverse Engineering Model Architecture (Optional)

Before training, you may want to inspect and reverse engineer the SmolLM2-135M architecture from HuggingFace. This is useful for:
- Understanding the exact model configuration
- Extracting architecture parameters
- Generating config files for training frameworks
- Verifying parameter counts

#### Basic Usage

```bash
# Download and inspect the model config (no model weights loaded)
python smollm2_135m_reverse_engineer.py
```

This will:
- Download the HuggingFace config and tokenizer
- Display an architecture summary
- Export configs to `smollm2_135m_reverse_engineered/`:
  - `hf_config.json` - Raw HuggingFace config
  - `hf_config.yaml` - Raw HuggingFace config (YAML format)
  - `smollm2_135m_training_skeleton.yaml` - Training-style config skeleton

#### Advanced Usage

```bash
# Load model weights and compute parameter statistics
python smollm2_135m_reverse_engineer.py --load-model --dtype bf16

# Use a different model variant
python smollm2_135m_reverse_engineer.py --model-id HuggingFaceTB/SmolLM2-135M-Instruct

# Custom output directory
python smollm2_135m_reverse_engineer.py --output-dir my_configs
```

#### Command Line Options

- `--model-id`: HuggingFace model ID (default: `HuggingFaceTB/SmolLM2-135M`)
- `--output-dir`: Directory for exported configs (default: `smollm2_135m_reverse_engineered`)
- `--load-model`: Load model weights and compute parameter stats (requires GPU memory)
- `--dtype`: Data type for model loading (`auto`, `fp16`, `bf16`)
- `--device`: Device for model loading (`auto` uses HF accelerate mapping)

#### Output Files

The script generates several useful files:

1. **`hf_config.json` / `hf_config.yaml`**: Raw HuggingFace configuration in both formats
2. **`smollm2_135m_training_skeleton.yaml`**: Training-style YAML with:
   - Model architecture parameters
   - Token IDs (BOS, EOS, PAD)
   - Training hyperparameter placeholders
   - Dataset configuration template
3. **`smollm2_135m_param_stats.json`**: Parameter statistics (only if `--load-model` is used)

**Note:** The `config.py` file in this project was generated from the reverse-engineered architecture. You can use this script to verify or regenerate the configuration.

### 1. Training from Scratch

#### Using Accelerate (Recommended)

```bash
# Configure accelerate (first time only)
accelerate config

# Start training
accelerate launch accelerate_train.py
```

The training script will:
- Build the model from scratch using `SmolLMConfig`
- Load and chunk the dataset from `input.txt`
- Train for 5000 steps with automatic checkpointing
- Save model to `checkpoint_5000/`

#### Using Simple Training Script

```bash
python train_from_scratch.py
```

### 2. Resuming Training

```bash
accelerate launch accelerate_resume.py
```

This will:
- Load the model from `checkpoint_5000/`
- Resume optimizer state
- Continue training from the saved step

### 3. Running the Gradio Demo

#### Local Demo

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

#### Hugging Face Spaces

**Option 1: Use Pretrained Model (Recommended for CPU-only Spaces)**

Since checkpoint files are large (~270MB), the easiest approach is to use the pretrained model:

1. Create a new Space on Hugging Face (CPU or GPU)
2. Upload `app.py` and `requirements.txt` (do NOT upload checkpoint files)
3. The app will automatically load from `HuggingFaceTB/SmolLM2-135M` if no checkpoint is found
4. The Space will automatically deploy

**CPU Performance Note:**
- The app automatically detects CPU and loads models in float32
- Generation will be slower on CPU (~5-10 seconds per generation)
- For faster inference, consider using a GPU-enabled Space

**Option 2: Upload Checkpoint to HuggingFace Hub (Recommended)**

If you want to use your fine-tuned checkpoint, upload it to HuggingFace Hub as a model repository (not in the Space):

1. Install dependencies and login:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. **Convert to FP16 first** (if your model is FP32):
   ```bash
   python convert_to_fp16.py --checkpoint-dir checkpoint_5000 --output-dir checkpoint_fp16
   ```

3. Upload checkpoint (optimizer state excluded by default):
   ```bash
   python upload_to_hub.py --repo-id your-username/smollm2-135m-coriolanus --checkpoint-dir checkpoint_fp16
   ```
   
   **Required files uploaded:**
   - ✅ `config.json` (~1KB) - Model configuration
   - ✅ `model.safetensors` (~257MB FP16 or ~513MB FP32) - Model weights
   - ✅ `generation_config.json` (~1KB) - Generation settings
   - ❌ **Excludes `optim.pt`** (~200MB) - Not needed for inference
   - ❌ **Tokenizer files** - Not needed (app uses `HuggingFaceTB/SmolLM2-135M` tokenizer)

3. Set environment variable in HF Space settings:
   - `HF_MODEL_ID`: `your-username/smollm2-135m-coriolanus`

4. Upload only `app.py` and `requirements.txt` to the Space (no checkpoint files!)

**Size Reduction:**
- Original checkpoint: ~713MB (FP32 model 513MB + optimizer 200MB)
- After removing optimizer: ~513MB (FP32 model only)
- After converting to FP16: ~257MB (50% reduction)
- Space files: <1MB (just app.py and requirements.txt)

**💡 Important:** If your model is saved in FP32 (~513MB), convert it to FP16 first:
```bash
python convert_to_fp16.py --checkpoint-dir checkpoint_5000 --output-dir checkpoint_fp16
python upload_to_hub.py --repo-id your-username/smollm2-135m-coriolanus --checkpoint-dir checkpoint_fp16
```

**Option 3: Use Quantization (GPU Only - Smaller Model Size)**

To reduce model size for HF Spaces with GPU:

1. Set environment variable in HF Space settings:
   - `USE_QUANTIZATION`: `8bit` (for 8-bit) or `4bit` (for 4-bit quantization)

2. This reduces model size:
   - 8-bit: ~135MB (50% reduction)
   - 4-bit: ~68MB (75% reduction)

3. Upload `app.py` and `requirements.txt` to the Space
4. Add `bitsandbytes>=0.41.0` to `requirements.txt` (for GPU quantization)

**Note:** 
- **Quantization requires GPU** - it will NOT work on CPU-only Spaces
- **For CPU-only Spaces**: Use Option 1 (pretrained model) - the app automatically detects CPU and loads without quantization
- The app will automatically use float32 on CPU for compatibility

### 4. Using the Model Programmatically

```python
from transformers import LlamaForCausalLM, GPT2TokenizerFast
import torch

# Load model
model = LlamaForCausalLM.from_pretrained("checkpoint_5000")
tokenizer = GPT2TokenizerFast.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer.pad_token = tokenizer.eos_token

# Generate text (model writes in dramatic play style)
prompt = "CORIOLANUS:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## ⚙️ Configuration

### Model Definition

The model is built from a simple configuration class defined in `config.py`:

```python
# Auto-generated config for SmolLM2-135M
class SmolLMConfig:
    def __init__(self):
        self.model_type = 'llama'
        self.vocab_size = 49152
        self.hidden_size = 576
        self.intermediate_size = 1536
        self.num_hidden_layers = 30
        self.num_attention_heads = 9
        self.num_key_value_heads = 3
        self.max_position_embeddings = 8192
        self.rms_norm_eps = 1e-05
        self.rope_theta = 100000
        self.rope_scaling = None
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.pad_token_id = None
        self.tie_word_embeddings = True
```

The model is instantiated in the training script:

```python
from transformers import LlamaForCausalLM, LlamaConfig
from config import SmolLMConfig

# Build config
sm_cfg = SmolLMConfig()
hf_config = LlamaConfig(
    vocab_size=sm_cfg.vocab_size,
    hidden_size=sm_cfg.hidden_size,
    intermediate_size=sm_cfg.intermediate_size,
    num_hidden_layers=sm_cfg.num_hidden_layers,
    num_attention_heads=sm_cfg.num_attention_heads,
    num_key_value_heads=sm_cfg.num_key_value_heads,
    max_position_embeddings=sm_cfg.max_position_embeddings,
    rms_norm_eps=sm_cfg.rms_norm_eps,
    rope_theta=sm_cfg.rope_theta,
    tie_word_embeddings=sm_cfg.tie_word_embeddings,
)

# Create model with SDPA (Flash Attention)
model = LlamaForCausalLM(hf_config, attn_implementation="sdpa")
```

Modify the values in `SmolLMConfig` to change the model architecture.

## 🎯 Training Configuration

Default training hyperparameters:

- **Optimizer:** AdamW
- **Learning Rate:** 2e-4
- **Sequence Length:** 256
- **Max Steps:** 5000
- **Batch Size:** 1 (per device, scales with Accelerate)
- **Mixed Precision:** Enabled (bfloat16/float16 on CUDA)

### Expected Training Output

When running `accelerate_train.py`, you should see output similar to:

```
Using device: cuda

==================================================
Model Parameters:
  Total: 134,515,008 (134.52M)
  Trainable: 134,515,008 (134.52M)
  Non-trainable: 0 (0)
==================================================

Loaded 1332 training chunks.

Step 0 | Loss 10.8222
Step 500 | Loss 4.7882
Step 1000 | Loss 5.9447
Step 1500 | Loss 5.2691
Step 2000 | Loss 4.5327
Step 2500 | Loss 4.7587
Step 3000 | Loss 4.1970
Step 3500 | Loss 5.0133
Step 4000 | Loss 4.4884
Step 4500 | Loss 3.5572

Saving model in FP16 format (reduces size by ~50%)
Training complete. Checkpoint saved.
```

**Training Notes:**
- Initial loss starts around 10-11 (typical for language modeling)
- Loss decreases over training steps, reaching ~3.5-4.5 after 5000 steps
- The model is trained on 1,332 chunks of 256 tokens each from Coriolanus
- Checkpoint is saved to `checkpoint_5000/` directory

### Resuming Training

To resume training from a checkpoint:

```bash
accelerate launch accelerate_resume.py
```

Expected output:

```
Resuming from step 5000
Extra step 0 | Loss: 15.6154
Extra step 10 | Loss: 11.2768
Extra step 20 | Loss: 10.9538
Extra step 30 | Loss: 11.6046
Extra step 40 | Loss: 11.2504
Saving model in FP16 format (reduces size by ~50%)
Resume training complete. Checkpoint saved at step 5050.
```

**Note:** When resuming, the loss may initially spike (as shown above) because the resume script uses synthetic data for demonstration. In production, you would load your actual training dataset.

## 📝 Dataset Format

The training script expects `input.txt` in the project root. The file should contain plain text that will be tokenized and chunked into sequences of length 256.

**Current Dataset:** This model is fine-tuned exclusively on Shakespeare's **Coriolanus** (contained in `input.txt`). As a result, the model generates text in the style of a dramatic play, with character names, stage directions, and Shakespearean dialogue.

The script automatically chunks the text into sequences of length 256 for training.

## 🔧 Advanced Usage

### Multi-GPU Training

Accelerate automatically handles multi-GPU training:

```bash
# Use all available GPUs
accelerate launch --multi_gpu accelerate_train.py

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 accelerate launch accelerate_train.py
```

### Custom Dataset

Modify the `load_dataset` function in `accelerate_train.py` to load your custom dataset format.

### Checkpoint Management

Checkpoints are saved in the format:
```
checkpoint_5000/
├── config.json              # Model configuration
├── model.safetensors        # Model weights
├── generation_config.json   # Generation settings
└── optim.pt                 # Optimizer state
```

## 🐛 Troubleshooting

### Windows Autocast Issues
Autocast is disabled on Windows by default. The training will still work but may be slower.

### Out of Memory
- Reduce `seq_len` in the training script
- Use gradient accumulation
- Enable CPU offloading with Accelerate

### Flash Attention Not Working
If flash attention isn't available, the model will fall back to standard attention. This is handled automatically.

### Tokenizer Warnings (Fixed)
The training script has been updated to handle common warnings:

- **Sequence Length Warning**: If you see "sequence length is longer than max_position_embeddings", this is now automatically suppressed. The script temporarily increases the tokenizer's `model_max_length` during data loading, then chunks the data into 256-token sequences for training.

- **Attention Mask Warnings**: Generation warnings about attention masks and pad tokens have been fixed by explicitly providing attention masks and token IDs during generation.

These warnings were harmless but have been resolved for a cleaner training experience.

## 📚 References

- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [SmolLM2 Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)

## 📄 License

This project uses the SmolLM2-135M model, which follows the original model's license. Please check the Hugging Face model card for licensing details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Note:** This is a training and inference framework. The model weights are trained from scratch or loaded from checkpoints. Make sure you have appropriate data and compute resources for training.

