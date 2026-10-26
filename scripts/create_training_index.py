#!/usr/bin/env python3
"""
Create a training dataset index from matched caption-image pairs.
This index maps each training sample to its image and caption for ML training.
"""

import json
import os
import sys
from datetime import datetime


def load_jsonl(filepath):
    """Load JSONL file into list."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_training_index(matched_file, image_folder, output_file="training_index.json"):
    """
    Create a training-ready index from matched caption-image pairs.
    
    Output format:
    {
      "dataset_info": {...},
      "samples": [
        {
          "id": 0,
          "pdf_hash": "...",
          "image_path": "full/path/to/image.png",
          "image_filename": "image.png",
          "caption": "...",
          "references": [...]
        },
        ...
      ]
    }
    """
    
    print("="*70)
    print("CREATING TRAINING INDEX")
    print("="*70)
    print()
    
    # Load matched entries
    print(f"Loading matched entries from: {matched_file}")
    matched_data = load_jsonl(matched_file)
    print(f"✓ Loaded {len(matched_data):,} matched entries")
    print()
    
    # Create training samples
    print("Creating training samples...")
    training_samples = []
    missing_images = []
    
    for i, entry in enumerate(matched_data):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(matched_data):,} ({(i+1)/len(matched_data)*100:.1f}%)")
        
        pdf_hash = entry.get('pdf_hash', '')
        image_filename = entry.get('image_filename', '')
        caption = entry.get('s2orc_caption', '')
        references = entry.get('s2orc_references', [])
        
        if not image_filename:
            missing_images.append(pdf_hash)
            continue
        
        # Handle multiple images (if image_filename is a list)
        if isinstance(image_filename, list):
            # Use first image, but could be expanded to create multiple samples
            image_filename = image_filename[0]
        
        # Create full image path
        image_path = os.path.join(image_folder, image_filename)
        
        # Check if image actually exists
        if not os.path.exists(image_path):
            missing_images.append(pdf_hash)
            continue
        
        # Create training sample
        sample = {
            "id": len(training_samples),
            "pdf_hash": pdf_hash,
            "image_path": image_path,
            "image_filename": image_filename,
            "caption": caption,
            "references": references,
            "caption_length": len(caption.split()),
            "has_references": len(references) > 0,
            "num_references": len(references)
        }
        
        training_samples.append(sample)
    
    print(f"✓ Created {len(training_samples):,} training samples")
    print()
    
    if missing_images:
        print(f"⚠ Warning: {len(missing_images):,} images not found on disk")
        print(f"  Sample missing (first 5): {missing_images[:5]}")
        print()
    
    # Create index structure
    print("Generating index statistics...")
    
    # Caption length statistics
    caption_lengths = [s['caption_length'] for s in training_samples]
    
    index_data = {
        "created": datetime.now().isoformat(),
        "source_file": matched_file,
        "image_folder": image_folder,
        "dataset_info": {
            "total_samples": len(training_samples),
            "source_entries": len(matched_data),
            "missing_images": len(missing_images),
            "caption_stats": {
                "min_length": min(caption_lengths) if caption_lengths else 0,
                "max_length": max(caption_lengths) if caption_lengths else 0,
                "mean_length": sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0,
                "median_length": sorted(caption_lengths)[len(caption_lengths)//2] if caption_lengths else 0
            },
            "samples_with_references": sum(1 for s in training_samples if s['has_references']),
            "samples_without_references": sum(1 for s in training_samples if not s['has_references'])
        },
        "samples": training_samples
    }
    
    # Save index
    print(f"Saving training index to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    
    # Get file size
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"✓ Training index saved ({file_size_mb:.2f} MB)")
    print()
    
    # Display statistics
    print("="*70)
    print("TRAINING INDEX STATISTICS")
    print("="*70)
    print(f"Total training samples: {len(training_samples):,}")
    print(f"Missing images: {len(missing_images):,}")
    print()
    print("Caption Statistics:")
    print(f"  Min length: {index_data['dataset_info']['caption_stats']['min_length']} words")
    print(f"  Max length: {index_data['dataset_info']['caption_stats']['max_length']} words")
    print(f"  Mean length: {index_data['dataset_info']['caption_stats']['mean_length']:.1f} words")
    print(f"  Median length: {index_data['dataset_info']['caption_stats']['median_length']} words")
    print()
    print("References:")
    print(f"  With references: {index_data['dataset_info']['samples_with_references']:,} "
          f"({index_data['dataset_info']['samples_with_references']/len(training_samples)*100:.1f}%)")
    print(f"  Without references: {index_data['dataset_info']['samples_without_references']:,} "
          f"({index_data['dataset_info']['samples_without_references']/len(training_samples)*100:.1f}%)")
    print()
    
    # Sample training samples
    print("Sample training samples (first 3):")
    for i, sample in enumerate(training_samples[:3]):
        print(f"\n  Sample {i+1}:")
        print(f"    ID: {sample['id']}")
        print(f"    PDF Hash: {sample['pdf_hash']}")
        print(f"    Image: {sample['image_filename']}")
        print(f"    Caption ({sample['caption_length']} words): {sample['caption'][:80]}...")
        print(f"    References: {sample['num_references']}")
    print()
    
    print("="*70)
    print("TRAINING INDEX COMPLETE!")
    print("="*70)
    print()
    print("Usage in training:")
    print("  import json")
    print(f"  with open('{output_file}') as f:")
    print("      index = json.load(f)")
    print("  samples = index['samples']")
    print("  ")
    print("  # Access training sample")
    print("  sample = samples[0]")
    print("  image_path = sample['image_path']")
    print("  caption = sample['caption']")
    print()
    
    return index_data


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 create_training_index.py <matched_file> <image_folder> [output_file]")
        print()
        print("Arguments:")
        print("  matched_file:  JSONL file with matched caption-image pairs (e.g., s2_cleaned_with_images.json)")
        print("  image_folder:  Folder containing the images")
        print("  output_file:   Output training index file (default: training_index.json)")
        print()
        print("Example:")
        print("  python3 create_training_index.py s2_cleaned_with_images.json /path/to/images")
        sys.exit(1)
    
    matched_file = sys.argv[1]
    image_folder = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "training_index.json"
    
    if not os.path.exists(matched_file):
        print(f"Error: Matched file not found: {matched_file}")
        sys.exit(1)
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found: {image_folder}")
        sys.exit(1)
    
    create_training_index(matched_file, image_folder, output_file)


if __name__ == "__main__":
    main()
