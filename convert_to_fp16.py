#!/usr/bin/env python
"""
Convert model checkpoint from FP32 to FP16/BF16 to reduce size.

Usage:
    python convert_to_fp16.py --checkpoint-dir checkpoint_5000 --output-dir checkpoint_fp16 --dtype fp16
    python convert_to_fp16.py --checkpoint-dir checkpoint_5000 --output-dir checkpoint_bf16 --dtype bf16
"""

import argparse
import shutil
from pathlib import Path
import torch
from transformers import LlamaForCausalLM

def convert_checkpoint_dtype(
    checkpoint_dir: str = "checkpoint_5000",
    output_dir: str = "checkpoint_fp16",
    dtype: str = "fp16"
):
    """
    Convert checkpoint to FP16 or BF16 to reduce file size.
    
    FP32: ~513MB
    FP16: ~257MB (50% reduction)
    BF16: ~257MB (50% reduction)
    """
    
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Check if model file exists and is accessible
    model_file = checkpoint_path / "model.safetensors"
    if not model_file.exists():
        model_file = checkpoint_path / "model.bin"
        if not model_file.exists():
            raise ValueError(f"Model file not found in {checkpoint_dir}. Expected 'model.safetensors' or 'model.bin'")
    
    print(f"Loading model from {checkpoint_dir}...")
    print(f"⚠️  Make sure the model file is not open in another program (IDE, etc.)")
    
    try:
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LlamaForCausalLM.from_pretrained(str(checkpoint_path))
    except Exception as e:
        if "permission" in str(e).lower() or "could not be read" in str(e).lower():
            raise RuntimeError(
                f"Permission error: Cannot read model file.\n"
                f"Please close the file if it's open in your IDE or another program.\n"
                f"File: {model_file}\n"
                f"Original error: {e}"
            )
        raise
    
    # Determine target dtype
    if dtype.lower() == "fp16":
        target_dtype = torch.float16
        dtype_name = "FP16"
    elif dtype.lower() == "bf16":
        target_dtype = torch.bfloat16
        dtype_name = "BF16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'fp16' or 'bf16'")
    
    print(f"Converting model to {dtype_name}...")
    
    # Convert model to target dtype
    model = model.to(target_dtype)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model in new dtype
    print(f"Saving converted model to {output_dir}...")
    model.save_pretrained(str(output_path))
    
    # Copy other essential files
    files_to_copy = [
        "generation_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
    ]
    
    for file_name in files_to_copy:
        src_file = checkpoint_path / file_name
        if src_file.exists():
            shutil.copy2(src_file, output_path / file_name)
    
    # Calculate size reduction
    original_safetensors = checkpoint_path / "model.safetensors"
    new_safetensors = output_path / "model.safetensors"
    
    if original_safetensors.exists() and new_safetensors.exists():
        original_size = original_safetensors.stat().st_size / (1024 * 1024)  # MB
        new_size = new_safetensors.stat().st_size / (1024 * 1024)  # MB
        reduction = original_size - new_size
        reduction_pct = (reduction / original_size) * 100
        
        print(f"\n📊 Size Comparison:")
        print(f"  Original (FP32): {original_size:.2f} MB")
        print(f"  Converted ({dtype_name}): {new_size:.2f} MB")
        print(f"  Reduction: {reduction:.2f} MB ({reduction_pct:.1f}%)")
    
    print(f"\n✅ Model converted to {dtype_name}!")
    print(f"   Use '{output_dir}' for upload to HuggingFace Hub.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert model checkpoint to FP16/BF16"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoint_5000",
        help="Source checkpoint directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoint_fp16",
        help="Output directory for converted checkpoint"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Target dtype: fp16 or bf16 (default: fp16)"
    )
    
    args = parser.parse_args()
    
    convert_checkpoint_dtype(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        dtype=args.dtype
    )

if __name__ == "__main__":
    main()

