#!/usr/bin/env python
"""
Upload checkpoint to HuggingFace Hub for use in Spaces.

Usage:
    python upload_to_hub.py --repo-id your-username/smollm2-135m-coriolanus
    python upload_to_hub.py --repo-id your-username/smollm2-135m-coriolanus --checkpoint-dir checkpoint_5000
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

def upload_checkpoint(repo_id: str, checkpoint_dir: str = "checkpoint_5000", private: bool = False, exclude_optimizer: bool = True, convert_to_fp16: bool = True):
    """Upload checkpoint directory to HuggingFace Hub"""
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Check if model file exists and is accessible
    model_file = checkpoint_path / "model.safetensors"
    if not model_file.exists():
        model_file = checkpoint_path / "model.bin"
        if not model_file.exists():
            raise ValueError(f"Model file not found. Expected 'model.safetensors' or 'model.bin'")
    
    # Check file permissions
    try:
        # Try to get file size to check if file is accessible
        model_file.stat().st_size
    except PermissionError:
        raise RuntimeError(
            f"Permission error: Cannot access model file.\n"
            f"Please close the file if it's open in your IDE or another program.\n"
            f"File: {model_file}"
        )
    
    # Check if model is FP32 and needs conversion
    if model_file.exists() and convert_to_fp16:
        model_size = model_file.stat().st_size / (1024 * 1024)  # MB
        if model_size > 400:  # Likely FP32 if > 400MB
            print(f"⚠️  Model size is {model_size:.2f} MB (likely FP32)")
            print(f"   Consider converting to FP16 first to reduce size by ~50%:")
            print(f"   python convert_to_fp16.py --checkpoint-dir {checkpoint_dir} --output-dir {checkpoint_dir}_fp16")
            print(f"   Then upload the converted checkpoint.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    print(f"Uploading {checkpoint_dir} to {repo_id}...")
    
    api = HfApi()
    
    # Files to exclude (not needed for inference)
    ignore_patterns = []
    if exclude_optimizer:
        ignore_patterns.append("optim.pt")
        print("  ℹ️  Excluding optim.pt (not needed for inference)")
    
    # Upload the checkpoint folder
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
    )
    
    # Calculate size saved
    total_saved = 0
    if exclude_optimizer:
        optim_file = checkpoint_path / "optim.pt"
        if optim_file.exists():
            optim_size = optim_file.stat().st_size / (1024 * 1024)  # MB
            total_saved += optim_size
            print(f"  ✓ Saved {optim_size:.2f} MB by excluding optimizer state")
    
    if model_file.exists():
        model_size = model_file.stat().st_size / (1024 * 1024)
        print(f"  📦 Model size: {model_size:.2f} MB")
        if model_size > 400:
            print(f"  💡 Tip: Convert to FP16 to reduce to ~{model_size/2:.0f} MB")
    
    print(f"\n✅ Successfully uploaded to https://huggingface.co/{repo_id}")
    print(f"\nTo use in HF Space, set environment variable:")
    print(f"  HF_MODEL_ID={repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoint_5000",
        help="Path to checkpoint directory (default: checkpoint_5000)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--include-optimizer",
        action="store_true",
        help="Include optimizer state (not needed for inference, increases size)"
    )
    parser.add_argument(
        "--no-convert-check",
        action="store_true",
        help="Skip FP32 to FP16 conversion check"
    )
    
    args = parser.parse_args()
    
    upload_checkpoint(
        args.repo_id, 
        args.checkpoint_dir, 
        args.private, 
        exclude_optimizer=not args.include_optimizer,
        convert_to_fp16=not args.no_convert_check
    )
    
    # Check if user is logged in
    try:
        api = HfApi()
        api.whoami()
    except Exception:
        print("Please login to HuggingFace first:")
        print("  huggingface-cli login")
        print("  or")
        print("  python -c 'from huggingface_hub import login; login()'")
        return
    
    upload_checkpoint(args.repo_id, args.checkpoint_dir, args.private, exclude_optimizer=not args.include_optimizer)

if __name__ == "__main__":
    main()

