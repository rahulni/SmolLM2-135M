import gradio as gr
import torch
from transformers import LlamaForCausalLM, GPT2TokenizerFast
import os

# Optional import for quantization (GPU only)
try:
    from transformers import BitsAndBytesConfig
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

# Global variables for model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration: 
# Set HF_MODEL_ID to load from your HuggingFace Hub repository instead of local checkpoint
HF_MODEL_ID = os.getenv("HF_MODEL_ID", None)  # e.g., "your-username/smollm2-135m-coriolanus"
# Note: Quantization (8-bit/4-bit) requires GPU and is disabled for CPU-only environments

def load_model():
    """Load the model and tokenizer with optional quantization"""
    global model, tokenizer
    
    if model is None:
        print("Loading model...")
        
        # Determine model source
        model_path = None
        if HF_MODEL_ID:
            # Load from HuggingFace Hub repository
            print(f"Loading from HuggingFace Hub: {HF_MODEL_ID}")
            model_path = HF_MODEL_ID
        elif os.path.exists("checkpoint_5000"):
            # Load from local checkpoint
            print("Loading from local checkpoint: checkpoint_5000")
            model_path = "checkpoint_5000"
        else:
            # Fallback: load pretrained model
            print("Loading pretrained model: HuggingFaceTB/SmolLM2-135M")
            model_path = "HuggingFaceTB/SmolLM2-135M"
        
        # Load model (quantization only available on GPU)
        if device == "cuda" and HAS_BITSANDBYTES:
            # GPU: Can use quantization if requested
            use_quantization = os.getenv("USE_QUANTIZATION", "").lower()
            if use_quantization == "4bit":
                try:
                    print("Using 4-bit quantization (QLoRA)")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model = LlamaForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except Exception as e:
                    print(f"Quantization failed, loading normally: {e}")
                    model = LlamaForCausalLM.from_pretrained(model_path)
                    model.to(device)
            elif use_quantization in ["8bit", "true", "1"]:
                try:
                    print("Using 8-bit quantization")
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    model = LlamaForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto"
                    )
                except Exception as e:
                    print(f"Quantization failed, loading normally: {e}")
                    model = LlamaForCausalLM.from_pretrained(model_path)
                    model.to(device)
            else:
                # GPU without quantization
                model = LlamaForCausalLM.from_pretrained(model_path)
                model.to(device)
        else:
            # CPU: Load without quantization (bitsandbytes doesn't work on CPU)
            print(f"Loading on {device.upper()} (quantization not available)")
            # Use float32 for CPU (more compatible, though slower)
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
            model.to(device)
        
        model.eval()
        
        tokenizer = GPT2TokenizerFast.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded successfully!")
    
    return model, tokenizer

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def format_number(num):
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return str(num)
    
    return f"Total: {total_params:,} ({format_number(total_params)}) | Trainable: {trainable_params:,} ({format_number(trainable_params)})"

def generate_text(
    prompt,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    do_sample,
    repetition_penalty
):
    """Generate text from prompt"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    
    try:
        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 50,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output if it's included
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    except Exception as e:
        return f"Error: {str(e)}"

def get_model_info():
    """Get model information"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    
    info = f"""
## Model Information

**Model Type:** SmolLM2-135M (LLaMA Architecture)
**Fine-tuned on:** Shakespeare's Coriolanus
**Device:** {device}
**Parameters:** {count_parameters(model)}

### Architecture Details
- **Hidden Size:** 576
- **Intermediate Size:** 1536
- **Number of Layers:** 30
- **Attention Heads:** 9
- **Key-Value Heads:** 3 (GQA)
- **Vocabulary Size:** 49,152
- **Max Position Embeddings:** 8,192
- **RoPE Theta:** 100,000

### Features
- ✅ Flash Attention (SDPA)
- ✅ Grouped Query Attention (GQA)
- ✅ RMSNorm
- ✅ Rotary Position Embeddings (RoPE)
- ✅ Tied Word Embeddings
- ✅ Fine-tuned on Coriolanus (writes like a dramatic play)
"""
    return info

# Load model on startup
model, tokenizer = load_model()

# Create Gradio interface
with gr.Blocks(title="SmolLM2-135M Demo") as demo:
    gr.Markdown(
        """
        # 🚀 SmolLM2-135M Text Generation Demo
        
        A lightweight language model (135M parameters) fine-tuned exclusively on Shakespeare's **Coriolanus**. 
        The model writes in the style of a dramatic play, complete with character names, stage directions, and Shakespearean dialogue.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):

            gr.Markdown("### Model Information")
            model_info = gr.Markdown(get_model_info())
            refresh_btn = gr.Button("🔄 Refresh Info")
        
        with gr.Column(scale=2):
            gr.Markdown("### Text Generation")
            
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here... (e.g., 'CORIOLANUS:' or 'Enter CORIOLANUS and MENENIUS')",
                lines=3,
                value="CORIOLANUS:"
            )
            
            with gr.Row():
                max_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=10,
                    maximum=512,
                    value=100,
                    step=10
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1
                )
            
            with gr.Row():
                top_p = gr.Slider(
                    label="Top-p (Nucleus Sampling)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05
                )
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1
                )
            
            with gr.Row():
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.05
                )
                do_sample = gr.Checkbox(
                    label="Enable Sampling",
                    value=True
                )
            
            generate_btn = gr.Button("✨ Generate")
            
            output = gr.Textbox(
                label="Generated Text",
                lines=10,
                interactive=False
            )
    
    # Event handlers
    generate_btn.click(
        fn=generate_text,
        inputs=[
            prompt_input,
            max_tokens,
            temperature,
            top_p,
            top_k,
            do_sample,
            repetition_penalty
        ],
        outputs=output
    )
    
    refresh_btn.click(
        fn=get_model_info,
        outputs=model_info
    )
    
    gr.Markdown(
        """
        ### Usage Tips
        - **Model Style**: This model is fine-tuned on Coriolanus and generates text in dramatic play format with character names and dialogue
        - **Prompt Examples**: Try prompts like "CORIOLANUS:", "Enter CORIOLANUS and MENENIUS", or "ACT I, SCENE I"
        - **Temperature**: Lower values (0.1-0.5) for more focused outputs, higher (0.8-1.5) for more creative text
        - **Top-p**: Controls diversity via nucleus sampling (0.9 is a good default)
        - **Top-k**: Limits sampling to top k tokens (50 is a good default)
        - **Repetition Penalty**: Higher values (1.1-1.3) reduce repetition
        - **Max New Tokens**: Maximum length of generated text
        """
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

