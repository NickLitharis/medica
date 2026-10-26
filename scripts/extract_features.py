#!/usr/bin/env python3
"""
Extract visual features from preprocessed images using Vision Transformer (ViT-Base).
Features are saved as numpy arrays for fast loading during training.
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import time
from datetime import datetime
from transformers import ViTImageProcessor, ViTModel


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_progress_bar(current, total, start_time, bar_length=50):
    """Print a progress bar with time estimates."""
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    elapsed = time.time() - start_time
    if current > 0:
        rate = current / elapsed
        remaining = (total - current) / rate if rate > 0 else 0
        eta = format_time(remaining)
        elapsed_str = format_time(elapsed)
        rate_str = f"{rate:.1f} img/s"
    else:
        eta = "calculating..."
        elapsed_str = "0s"
        rate_str = "0 img/s"
    
    print(f"\r  [{bar}] {percent*100:.1f}% | {current:,}/{total:,} | "
          f"Elapsed: {elapsed_str} | ETA: {eta} | Speed: {rate_str}", 
          end='', flush=True)


def load_preprocessed_index(index_file):
    """Load the preprocessed training index."""
    with open(index_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_features_batch(images, model, processor, device):
    """
    Extract features from a batch of images using ViT.
    
    Args:
        images: List of PIL Images
        model: ViT model
        processor: ViT image processor
        device: torch device
    
    Returns:
        numpy array of features (batch_size, 768)
    """
    # Process images
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token representation (first token)
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return features


def get_device():
    """
    Detect the best available device (GPU or CPU).
    For AMD GPUs, PyTorch needs to be compiled with ROCm support.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"ℹ Using CPU (GPU not detected)")
        print(f"  Note: For AMD GPUs, install PyTorch with ROCm:")
        print(f"  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2")
        print()
        print(f"  CPU will work but slower. Optimizing for multi-threading...")
        
        # Optimize CPU performance
        torch.set_num_threads(os.cpu_count())
        print(f"  ✓ Using {os.cpu_count()} CPU threads")
    
    return device


def extract_features_from_index(index_file, output_folder, batch_size=32, device=None):
    """
    Extract features from all preprocessed images.
    
    Args:
        index_file: Path to verified_training_index_preprocessed.json
        output_folder: Folder to save feature files
        batch_size: Number of images to process at once
        device: Device to use (None for auto-detect)
    """
    
    print("="*70)
    print("FEATURE EXTRACTION WITH VISION TRANSFORMER")
    print("="*70)
    print()
    
    # Auto-detect device
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)
    
    print(f"Device: {device}")
    print()
    
    # Load preprocessed index
    print(f"Loading preprocessed index: {index_file}")
    index_data = load_preprocessed_index(index_file)
    samples = index_data['samples']
    print(f"✓ Loaded {len(samples):,} preprocessed samples")
    print()
    
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output folder: {output_folder}")
    print()
    
    # Load ViT model
    print("Loading Vision Transformer (ViT-Base-Patch16-224)...")
    print("  Model: google/vit-base-patch16-224")
    print("  Feature dimension: 768")
    print()
    
    try:
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print()
        print("Please install required packages:")
        print("  pip install transformers torch torchvision")
        sys.exit(1)
    
    # Feature extraction
    print(f"Extracting features (batch size: {batch_size})...")
    print()
    start_time = time.time()
    
    extracted_count = 0
    failed_count = 0
    skipped_count = 0
    
    samples_with_features = []
    
    # Initial progress bar
    print_progress_bar(0, len(samples), start_time)
    
    # Process in batches
    for batch_start in range(0, len(samples), batch_size):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]
        
        # Load batch images
        batch_images = []
        valid_samples = []
        
        for sample in batch_samples:
            image_path = sample.get('preprocessed_image_path')
            
            if not image_path or not os.path.exists(image_path):
                failed_count += 1
                continue
            
            # Generate feature filename
            pdf_hash = sample['pdf_hash']
            feature_filename = f"{pdf_hash}_features.npy"
            feature_path = output_path / feature_filename
            
            # Skip if already extracted
            if feature_path.exists():
                # Load existing features and add to sample
                features = np.load(feature_path)
                sample_with_features = sample.copy()
                sample_with_features['feature_path'] = str(feature_path)
                sample_with_features['feature_filename'] = feature_filename
                sample_with_features['feature_dim'] = features.shape[0]
                samples_with_features.append(sample_with_features)
                skipped_count += 1
                continue
            
            try:
                img = Image.open(image_path).convert('RGB')
                batch_images.append(img)
                valid_samples.append((sample, feature_path, feature_filename))
            except Exception as e:
                print(f"\n  Warning: Failed to load {image_path}: {e}")
                failed_count += 1
        
        # Extract features for batch
        if batch_images:
            try:
                features_batch = extract_features_batch(batch_images, model, processor, device)
                
                # Save features and update samples
                for (sample, feature_path, feature_filename), features in zip(valid_samples, features_batch):
                    # Save features as numpy array
                    np.save(feature_path, features)
                    
                    # Update sample with feature path
                    sample_with_features = sample.copy()
                    sample_with_features['feature_path'] = str(feature_path)
                    sample_with_features['feature_filename'] = feature_filename
                    sample_with_features['feature_dim'] = features.shape[0]
                    samples_with_features.append(sample_with_features)
                    
                    extracted_count += 1
                
            except Exception as e:
                print(f"\n  Warning: Batch extraction failed: {e}")
                failed_count += len(batch_images)
        
        # Update progress
        print_progress_bar(batch_end, len(samples), start_time)
    
    # Final progress bar
    print_progress_bar(len(samples), len(samples), start_time)
    print()
    print()
    
    total_time = time.time() - start_time
    
    print(f"✓ Feature extraction complete in {format_time(total_time)}")
    print(f"✓ Processing rate: {len(samples)/total_time:.1f} images/second")
    print()
    
    # Statistics
    print("="*70)
    print("FEATURE EXTRACTION RESULTS")
    print("="*70)
    print(f"Total samples: {len(samples):,}")
    print(f"Successfully extracted: {extracted_count:,} ({extracted_count/len(samples)*100:.1f}%)")
    print(f"Failed: {failed_count:,} ({failed_count/len(samples)*100:.1f}%)")
    print(f"Skipped (already exists): {skipped_count:,}")
    print()
    
    # Calculate storage
    if samples_with_features:
        sample_feature = np.load(samples_with_features[0]['feature_path'])
        feature_size = sample_feature.nbytes
        total_size = feature_size * len(samples_with_features) / (1024**3)
        print(f"Feature dimension: {sample_feature.shape[0]}")
        print(f"Size per feature: {feature_size / 1024:.2f} KB")
        print(f"Total storage: {total_size:.2f} GB")
        print()
    
    # Save updated index with feature paths
    features_index_file = index_file.replace('.json', '_features.json')
    print(f"Saving feature index to: {features_index_file}")
    
    features_index = {
        "created": datetime.now().isoformat(),
        "source_index": index_file,
        "model_info": {
            "model_name": "google/vit-base-patch16-224",
            "model_type": "Vision Transformer (ViT-Base)",
            "feature_dim": 768,
            "patch_size": 16,
            "image_size": 224
        },
        "extraction_config": {
            "batch_size": batch_size,
            "device": str(device)
        },
        "dataset_info": {
            "total_samples": len(samples_with_features),
            "extracted": extracted_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "output_folder": output_folder,
            "total_size_gb": total_size if samples_with_features else 0
        },
        "samples": samples_with_features
    }
    
    with open(features_index_file, 'w', encoding='utf-8') as f:
        json.dump(features_index, f, indent=2)
    
    print(f"✓ Feature index saved")
    print()
    
    print("="*70)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Create train/validation/test split")
    print("  2. Set up caption tokenizer")
    print("  3. Create data loaders")
    print("  4. Train caption generation model")
    print()
    print(f"Feature files location: {output_folder}")
    print(f"Feature index: {features_index_file}")
    print()


def check_gpu():
    """
    Dry run to check GPU availability and compatibility.
    Tests a small batch to ensure everything works.
    """
    print("="*70)
    print("GPU COMPATIBILITY CHECK (DRY RUN)")
    print("="*70)
    print()
    
    # Check PyTorch installation
    print("1. Checking PyTorch installation...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   ✓ PyTorch installed")
    print()
    
    # Check CUDA/ROCm
    print("2. Checking GPU support...")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA/ROCm available")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print()
    else:
        print(f"   ⚠ No GPU detected - will use CPU (slower)")
        print()
    
    # Detect device
    device = get_device()
    print(f"3. Selected device: {device}")
    print()
    
    # Test model loading
    print("4. Testing model loading...")
    try:
        from transformers import ViTImageProcessor, ViTModel
        
        print(f"   Loading ViT-Base model...")
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        model = model.to(device)
        model.eval()
        print(f"   ✓ Model loaded successfully on {device}")
        print()
        
        # Test with dummy image
        print("5. Testing feature extraction with dummy image...")
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(pixel_values=dummy_image)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token
        elapsed = time.time() - start_time
        
        print(f"   ✓ Feature extraction successful")
        print(f"   Output shape: {features.shape}")
        print(f"   Processing time: {elapsed*1000:.2f}ms per image")
        print(f"   Estimated speed: {1/elapsed:.1f} images/second")
        print()
        
        # CPU optimization check
        if device.type == 'cpu':
            print(f"   CPU Performance Tips:")
            print(f"   - Using {torch.get_num_threads()} threads")
            print(f"   - Batch size: 8 (optimized for CPU)")
            print(f"   - Expected: 5-20 images/second on modern CPU")
            print()
        
        # Estimate total time
        print("6. Estimating processing time for full dataset...")
        # Assume we have ~93K images
        estimated_images = 93565
        estimated_time = estimated_images / (1/elapsed)
        
        print(f"   Estimated images: ~{estimated_images:,}")
        print(f"   Estimated time: {format_time(estimated_time)}")
        if estimated_time > 3600:
            print(f"   ({estimated_time/3600:.1f} hours)")
        print()
        
        # Memory check
        if torch.cuda.is_available():
            print("7. GPU Memory usage:")
            print(f"   Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"   Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            print()
        
        print("="*70)
        print("✓ DRY RUN SUCCESSFUL - GPU IS READY!")
        print("="*70)
        print()
        print("You can now run feature extraction with:")
        print("  python3 scripts/extract_features.py")
        print()
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error during dry run: {e}")
        print()
        print("="*70)
        print("✗ DRY RUN FAILED")
        print("="*70)
        print()
        return False


def main():
    # Check for dry run flag first
    if len(sys.argv) > 1 and sys.argv[1] in ["--dry-run", "--test", "--check-gpu"]:
        print("Running GPU compatibility check...")
        print()
        success = check_gpu()
        sys.exit(0 if success else 1)
    
    # Internal configuration
    BASE_DIR = Path(__file__).parent.parent
    
    # Input/Output paths
    preprocessed_index_file = BASE_DIR / "data" / "indexes" / "verified_training_index_preprocessed.json"
    output_folder = BASE_DIR / "data" / "features"
    batch_size = 8  # Smaller batch for CPU (adjust: 4-16 depending on RAM)
    
    print("="*70)
    print("FEATURE EXTRACTION CONFIGURATION")
    print("="*70)
    print(f"Preprocessed Index: {preprocessed_index_file}")
    print(f"Output Folder: {output_folder}")
    print(f"Batch Size: {batch_size}")
    print(f"Model: Vision Transformer (ViT-Base)")
    print(f"CPU Threads: {os.cpu_count()}")
    print()
    
    # Check if preprocessed index exists
    if not preprocessed_index_file.exists():
        print(f"❌ Error: Preprocessed index not found: {preprocessed_index_file}")
        print()
        print("Please run preprocessing first:")
        print("  python3 scripts/preprocess_images.py")
        sys.exit(1)
    
    # Optional: Allow command line override
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print("Default behavior: Uses internal configuration")
            print()
            print("Optional command line usage:")
            print("  python3 scripts/extract_features.py [preprocessed_index] [output_folder] [batch_size]")
            print()
            print("Batch size recommendations:")
            print("  GPU Memory 8GB:  batch_size=16-32")
            print("  GPU Memory 16GB: batch_size=32-64")
            print("  GPU Memory 24GB+: batch_size=64-128")
            print("  CPU only: batch_size=8-16")
            sys.exit(0)
        
        if len(sys.argv) >= 3:
            preprocessed_index_file = Path(sys.argv[1])
            output_folder = Path(sys.argv[2])
            if len(sys.argv) >= 4:
                batch_size = int(sys.argv[3])
            print("✓ Using command-line arguments")
            print()
    
    extract_features_from_index(
        str(preprocessed_index_file), 
        str(output_folder), 
        batch_size=batch_size,
        device=None  # Auto-detect
    )


if __name__ == "__main__":
    main()
