#!/usr/bin/env python
"""
Prepare checkpoint for Hugging Face Spaces by removing unnecessary files
and optionally converting to smaller format.

Usage:
    python prepare_for_hf_space.py --checkpoint-dir checkpoint_5000 --output-dir checkpoint_hf_space
"""

import argparse
import shutil
from pathlib import Path
import torch

def prepare_checkpoint_for_hf_space(
    checkpoint_dir: str = "checkpoint_5000",
    output_dir: str = "checkpoint_hf_space",
    remove_optimizer: bool = True,
    keep_only_essential: bool = True
):
    """
    Prepare checkpoint for HF Space by:
    1. Removing optimizer state (not needed for inference)
    2. Keeping only essential files (config.json, model.safetensors, generation_config.json)
    """
    
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing checkpoint from {checkpoint_dir} to {output_dir}...")
    
    # Essential files to copy
    essential_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",  # or model.bin
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.json"
    ]
    
    # Copy essential files
    copied_files = []
    for file_name in essential_files:
        src_file = checkpoint_path / file_name
        if src_file.exists():
            dst_file = output_path / file_name
            shutil.copy2(src_file, dst_file)
            copied_files.append(file_name)
            file_size = src_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ Copied {file_name} ({file_size:.2f} MB)")
    
    # Check for model.bin if safetensors doesn't exist
    if "model.safetensors" not in copied_files:
        model_bin = checkpoint_path / "model.bin"
        if model_bin.exists():
            shutil.copy2(model_bin, output_path / "model.bin")
            file_size = model_bin.stat().st_size / (1024 * 1024)
            print(f"  ✓ Copied model.bin ({file_size:.2f} MB)")
            copied_files.append("model.bin")
    
    # Skip optimizer file (not needed for inference)
    if remove_optimizer:
        optim_file = checkpoint_path / "optim.pt"
        if optim_file.exists():
            file_size = optim_file.stat().st_size / (1024 * 1024)
            print(f"  ✗ Skipped optim.pt ({file_size:.2f} MB) - not needed for inference")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_path.glob("*")) / (1024 * 1024)
    original_size = sum(f.stat().st_size for f in checkpoint_path.glob("*")) / (1024 * 1024)
    
    print(f"\n📊 Size Summary:")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Prepared: {total_size:.2f} MB")
    if remove_optimizer:
        print(f"  Saved: {original_size - total_size:.2f} MB (removed optimizer)")
    
    print(f"\n✅ Checkpoint prepared! Use '{output_dir}' for HF Space.")
    print(f"\n💡 Recommendation: Upload this to HuggingFace Hub as a model repository,")
    print(f"   then set HF_MODEL_ID environment variable in your Space.")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare checkpoint for Hugging Face Spaces"
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
        default="checkpoint_hf_space",
        help="Output directory for prepared checkpoint"
    )
    parser.add_argument(
        "--keep-optimizer",
        action="store_true",
        help="Keep optimizer state (not recommended for inference)"
    )
    
    args = parser.parse_args()
    
    prepare_checkpoint_for_hf_space(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        remove_optimizer=not args.keep_optimizer,
    )

if __name__ == "__main__":
    main()

