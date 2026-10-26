#!/usr/bin/env python3
"""
Preprocess medical images for training.
Based on analysis: 138,170 PNG images, RGB, 24-bit, median 940x652px
"""

import os
import sys
import json
from PIL import Image
import numpy as np
from pathlib import Path
import time
from datetime import datetime, timedelta


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
    """
    Print a progress bar with time estimates.
    
    Args:
        current: Current progress count
        total: Total items to process
        start_time: Start time (from time.time())
        bar_length: Length of the progress bar
    """
    percent = current / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # Calculate time estimates
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
    
    # Print progress bar
    print(f"\r  [{bar}] {percent*100:.1f}% | {current:,}/{total:,} | "
          f"Elapsed: {elapsed_str} | ETA: {eta} | Speed: {rate_str}", 
          end='', flush=True)


def load_training_index(index_file):
    """Load the verified training index."""
    with open(index_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def preprocess_image(image_path, target_size=(512, 512), normalize=True, 
                     apply_clahe=False, maintain_aspect=True):
    """
    Preprocess a single image.
    
    Args:
        image_path: Path to input image
        target_size: Target dimensions (width, height)
        normalize: Normalize pixel values to [0, 1]
        apply_clahe: Apply CLAHE for contrast enhancement
        maintain_aspect: Maintain aspect ratio with padding
    
    Returns:
        Preprocessed image as PIL Image
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if needed (already RGB based on analysis)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize strategy
        if maintain_aspect:
            # Maintain aspect ratio with padding
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_size[0] - img.width) // 2
            paste_y = (target_size[1] - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        else:
            # Direct resize (may distort)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Apply CLAHE for medical image enhancement (optional)
        if apply_clahe:
            try:
                import cv2
                img_array = np.array(img)
                
                # Apply CLAHE to each channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                for i in range(3):
                    img_array[:, :, i] = clahe.apply(img_array[:, :, i])
                
                img = Image.fromarray(img_array)
            except ImportError:
                print("  Warning: OpenCV not installed, skipping CLAHE")
        
        return img
        
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return None


def preprocess_dataset(index_file, output_folder, target_size=(512, 512),
                       normalize=True, apply_clahe=False, maintain_aspect=True,
                       output_format='PNG'):
    """
    Preprocess entire dataset based on training index.
    
    Args:
        index_file: Path to verified_training_index.json
        output_folder: Output folder for preprocessed images
        target_size: Target image dimensions
        normalize: Normalize pixel values
        apply_clahe: Apply CLAHE enhancement
        maintain_aspect: Maintain aspect ratio
        output_format: Output image format (PNG, JPG)
    """
    
    print("="*70)
    print("IMAGE PREPROCESSING PIPELINE")
    print("="*70)
    print()
    
    # Load training index
    print(f"Loading training index: {index_file}")
    index_data = load_training_index(index_file)
    samples = index_data['samples']
    print(f"✓ Loaded {len(samples):,} training samples")
    print()
    
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output folder: {output_folder}")
    print()
    
    # Preprocessing configuration
    print("Preprocessing Configuration:")
    print(f"  Target size: {target_size[0]}x{target_size[1]}")
    print(f"  Maintain aspect ratio: {maintain_aspect}")
    print(f"  Normalize: {normalize}")
    print(f"  Apply CLAHE: {apply_clahe}")
    print(f"  Output format: {output_format}")
    print()
    
    # Process images
    print("Processing images...")
    print()
    start_time = time.time()
    
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    
    preprocessed_samples = []
    
    # Initial progress bar
    print_progress_bar(0, len(samples), start_time)
    
    for i, sample in enumerate(samples):
        # Update progress bar every image (or every 100 for very large datasets)
        update_freq = 1 if len(samples) < 10000 else 100
        if (i + 1) % update_freq == 0 or i == 0:
            print_progress_bar(i + 1, len(samples), start_time)
        
        image_path = sample['image_path']
        image_filename = sample['image_filename']
        pdf_hash = sample['pdf_hash']
        
        # Check if source image exists
        if not os.path.exists(image_path):
            failed_count += 1
            continue
        
        # Generate output filename (preserve original name with hash)
        output_filename = f"{pdf_hash}_{processed_count:06d}.{output_format.lower()}"
        output_path_full = output_path / output_filename
        
        # Skip if already processed
        if output_path_full.exists():
            skipped_count += 1
            continue
        
        # Preprocess image
        processed_img = preprocess_image(
            image_path,
            target_size=target_size,
            normalize=normalize,
            apply_clahe=apply_clahe,
            maintain_aspect=maintain_aspect
        )
        
        if processed_img is not None:
            # Save preprocessed image
            if output_format.upper() == 'JPG':
                processed_img.save(output_path_full, 'JPEG', quality=95)
            else:
                processed_img.save(output_path_full, output_format.upper())
            
            # Update sample with preprocessed path
            preprocessed_sample = sample.copy()
            preprocessed_sample['preprocessed_image_path'] = str(output_path_full)
            preprocessed_sample['preprocessed_image_filename'] = output_filename
            preprocessed_sample['preprocessed_size'] = target_size
            preprocessed_samples.append(preprocessed_sample)
            
            processed_count += 1
        else:
            failed_count += 1
    
    # Final progress bar
    print_progress_bar(len(samples), len(samples), start_time)
    print()  # New line after progress bar
    
    total_time = time.time() - start_time
    
    print()
    print(f"✓ Preprocessing complete in {format_time(total_time)}")
    print(f"✓ Processing rate: {len(samples)/total_time:.1f} images/second")
    print()
    
    # Statistics
    print("="*70)
    print("PREPROCESSING RESULTS")
    print("="*70)
    print(f"Total samples: {len(samples):,}")
    print(f"Successfully processed: {processed_count:,} ({processed_count/len(samples)*100:.1f}%)")
    print(f"Failed: {failed_count:,} ({failed_count/len(samples)*100:.1f}%)")
    print(f"Skipped (already exists): {skipped_count:,}")
    print()
    
    # Save updated index with preprocessed paths
    preprocessed_index_file = index_file.replace('.json', '_preprocessed.json')
    print(f"Saving preprocessed index to: {preprocessed_index_file}")
    
    preprocessed_index = {
        "created": datetime.now().isoformat(),
        "source_index": index_file,
        "preprocessing_config": {
            "target_size": target_size,
            "maintain_aspect": maintain_aspect,
            "normalize": normalize,
            "apply_clahe": apply_clahe,
            "output_format": output_format
        },
        "dataset_info": {
            "total_samples": len(preprocessed_samples),
            "processed": processed_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "output_folder": output_folder
        },
        "samples": preprocessed_samples
    }
    
    with open(preprocessed_index_file, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_index, f, indent=2)
    
    print(f"✓ Preprocessed index saved")
    print()
    
    # Calculate storage
    if processed_count > 0:
        sample_size = os.path.getsize(preprocessed_samples[0]['preprocessed_image_path'])
        estimated_total = sample_size * len(preprocessed_samples) / (1024**3)
        print(f"Estimated total size: {estimated_total:.2f} GB")
        print()
    
    print("="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Verify a few preprocessed images manually")
    print("  2. Create train/validation/test split")
    print("  3. Set up data loaders for training")
    print()


def main():
    # Internal configuration - no command line needed!
    BASE_DIR = Path(__file__).parent.parent  # Go up to project root
    
    # Input/Output paths (edit these if needed)
    training_index_file = BASE_DIR / "data" / "indexes" / "verified_training_index.json"
    output_folder = BASE_DIR / "data" / "preprocessed" / "images"
    target_size = 512  # Options: 224, 256, 384, 512
    output_format = "png"  # Options: png, jpg
    normalize = True
    apply_clahe = True  # Set True for X-ray/CT enhancement
    
    print("="*70)
    print("IMAGE PREPROCESSING CONFIGURATION")
    print("="*70)
    print(f"Training Index: {training_index_file}")
    print(f"Output Folder:  {output_folder}")
    print(f"Target Size:    {target_size}x{target_size}")
    print(f"Output Format:  {output_format}")
    print(f"Normalization:  {normalize}")
    print(f"CLAHE Enhancement: {apply_clahe}")
    print()
    
    # Check if files exist
    if not training_index_file.exists():
        print(f"❌ Error: Training index not found: {training_index_file}")
        print()
        print("Please run first:")
        print("  python3 scripts/fix_and_create_verified_index.py")
        sys.exit(1)
    
    # Optional: Allow command line override
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print("Default behavior: Uses internal configuration (edit script to change)")
            print()
            print("Optional command line usage:")
            print("  python3 scripts/preprocess_images.py [training_index] [output_folder] [size]")
            sys.exit(0)
        
        if len(sys.argv) >= 4:
            training_index_file = Path(sys.argv[1])
            output_folder = Path(sys.argv[2])
            target_size = int(sys.argv[3])
            print("✓ Using command-line arguments")
            print()
    
    preprocess_dataset(
        str(training_index_file), 
        str(output_folder), 
        target_size=(target_size, target_size),
        output_format=output_format,
        normalize=normalize,
        apply_clahe=apply_clahe
    )


if __name__ == "__main__":
    main()
