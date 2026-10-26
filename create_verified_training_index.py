#!/usr/bin/env python3
"""
Create training index by cross-referencing s2_cleaned.json with image_index.json.
Only includes entries where BOTH caption AND image exist.
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


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_verified_training_index(captions_file, image_index_file, output_file="verified_training_index.json", missing_hashes_file="missing_hashes.txt"):
    """
    Create training index with verified caption-image pairs.
    Only includes entries present in BOTH files.
    Saves missing pdf_hashes to a text file.
    """
    
    print("="*70)
    print("CREATING VERIFIED TRAINING INDEX")
    print("="*70)
    print()
    
    # Load captions
    print(f"Loading captions from: {captions_file}")
    captions_data = load_jsonl(captions_file)
    print(f"✓ Loaded {len(captions_data):,} caption entries")
    print()
    
    # Load image index
    print(f"Loading image index from: {image_index_file}")
    index_data = load_json(image_index_file)
    image_index = index_data['index']
    image_folder = index_data.get('image_folder', '')
    print(f"✓ Loaded {len(image_index):,} image entries")
    print(f"✓ Image folder: {image_folder}")
    print()
    
    # Cross-reference and create training samples
    print("Cross-referencing captions with images...")
    training_samples = []
    missing_in_index = []
    missing_on_disk = []
    multiple_images_list = []
    empty_pdf_hash = 0
    empty_caption = 0
    
    for i, entry in enumerate(captions_data):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(captions_data):,} ({(i+1)/len(captions_data)*100:.1f}%)")
        
        pdf_hash = entry.get('pdf_hash', '')
        caption = entry.get('s2orc_caption', '')
        references = entry.get('s2orc_references', [])
        
        # Skip if no pdf_hash
        if not pdf_hash:
            empty_pdf_hash += 1
            continue
        
        # Skip if no caption
        if not caption or not caption.strip():
            empty_caption += 1
            continue
        
        # Check if image exists in index (O(1) lookup!)
        if pdf_hash not in image_index:
            missing_in_index.append(pdf_hash)
            continue
        
        image_filename = image_index[pdf_hash]
        
        # Handle multiple images per hash
        if isinstance(image_filename, list):
            multiple_images_list.append((pdf_hash, len(image_filename)))
            
            # Create one sample per image for papers with multiple figures
            for img_file in image_filename:
                image_path = os.path.join(image_folder, img_file)
                
                # Verify image exists on disk
                if not os.path.exists(image_path):
                    missing_on_disk.append((pdf_hash, img_file))
                    continue
                
                # Create training sample
                sample = {
                    "id": len(training_samples),
                    "pdf_hash": pdf_hash,
                    "image_path": image_path,
                    "image_filename": img_file,
                    "caption": caption,
                    "references": references,
                    "caption_length": len(caption.split()),
                    "has_references": len(references) > 0,
                    "num_references": len(references),
                    "is_multi_figure": True,
                    "figure_index": image_filename.index(img_file)
                }
                
                training_samples.append(sample)
        else:
            # Single image per hash
            image_path = os.path.join(image_folder, image_filename)
            
            # Verify image exists on disk
            if not os.path.exists(image_path):
                missing_on_disk.append((pdf_hash, image_filename))
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
                "num_references": len(references),
                "is_multi_figure": False,
                "figure_index": 0
            }
            
            training_samples.append(sample)
    
    print(f"✓ Cross-reference complete!")
    print()
    
    # Save missing hashes to text file
    if missing_in_index:
        print(f"Saving missing pdf_hashes to: {missing_hashes_file}")
        with open(missing_hashes_file, 'w', encoding='utf-8') as f:
            f.write(f"# Missing PDF Hashes Report\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total missing: {len(missing_in_index):,}\n")
            f.write(f"# These pdf_hash values exist in {captions_file}\n")
            f.write(f"# but NOT in {image_index_file}\n\n")
            
            for pdf_hash in missing_in_index:
                f.write(f"{pdf_hash}\n")
        
        print(f"✓ Saved {len(missing_in_index):,} missing hashes")
        print()
    
    # Statistics
    print("="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    print(f"Total caption entries:           {len(captions_data):,}")
    print(f"  Empty pdf_hash:                {empty_pdf_hash:,}")
    print(f"  Empty captions:                {empty_caption:,}")
    print(f"✓ Verified samples (all exist):  {len(training_samples):,}")
    print(f"  From unique pdf_hashes:        {len(set(s['pdf_hash'] for s in training_samples)):,}")
    print(f"✗ Missing in image index:        {len(missing_in_index):,} ({len(missing_in_index)/len(captions_data)*100:.2f}%)")
    print(f"✗ Missing on disk:               {len(missing_on_disk):,}")
    print(f"⚠ Papers with multiple figures:  {len(multiple_images_list):,}")
    print()
    
    # Sample issues
    if missing_in_index:
        print("Sample missing in index (first 5 pdf_hash):")
        for pdf_hash in missing_in_index[:5]:
            print(f"  {pdf_hash}")
        print()
    
    if missing_on_disk:
        print("Sample missing on disk (first 5):")
        for pdf_hash, filename in missing_on_disk[:5]:
            print(f"  {pdf_hash} → {filename}")
        print()
    
    if multiple_images_list:
        total_extra_samples = sum(count - 1 for _, count in multiple_images_list)
        print(f"Multiple figures info:")
        print(f"  Papers with multiple figures:  {len(multiple_images_list):,}")
        print(f"  Extra samples generated:       {total_extra_samples:,}")
        print(f"\nSample papers with multiple figures (first 5):")
        for pdf_hash, count in multiple_images_list[:5]:
            print(f"  {pdf_hash}: {count} images")
        print()
    
    # Calculate caption statistics
    if training_samples:
        caption_lengths = [s['caption_length'] for s in training_samples]
        sorted_lengths = sorted(caption_lengths)
        
        caption_stats = {
            "min_length": min(caption_lengths),
            "max_length": max(caption_lengths),
            "mean_length": sum(caption_lengths) / len(caption_lengths),
            "median_length": sorted_lengths[len(caption_lengths)//2],
            "p25_length": sorted_lengths[len(caption_lengths)//4],
            "p75_length": sorted_lengths[3*len(caption_lengths)//4]
        }
    else:
        caption_stats = {}
    
    # Create index structure
    index_output = {
        "created": datetime.now().isoformat(),
        "source_files": {
            "captions": captions_file,
            "image_index": image_index_file
        },
        "image_folder": image_folder,
        "verification_stats": {
            "total_captions": len(captions_data),
            "total_images_in_index": len(image_index),
            "unique_hashes_in_index": len(image_index),
            "verified_training_samples": len(training_samples),
            "unique_pdf_hashes_used": len(set(s['pdf_hash'] for s in training_samples)),
            "empty_pdf_hash": empty_pdf_hash,
            "empty_captions": empty_caption,
            "missing_in_index": len(missing_in_index),
            "missing_on_disk": len(missing_on_disk),
            "papers_with_multiple_figures": len(multiple_images_list),
            "sample_success_rate_percent": round(len(training_samples)/len(captions_data)*100, 2)
        },
        "dataset_info": {
            "total_samples": len(training_samples),
            "unique_papers": len(set(s['pdf_hash'] for s in training_samples)),
            "multi_figure_samples": sum(1 for s in training_samples if s['is_multi_figure']),
            "single_figure_samples": sum(1 for s in training_samples if not s['is_multi_figure']),
            "samples_with_references": sum(1 for s in training_samples if s['has_references']),
            "samples_without_references": sum(1 for s in training_samples if not s['has_references']),
            "caption_stats": caption_stats
        },
        "samples": training_samples
    }
    
    # Save training index
    print(f"Saving verified training index to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_output, f, indent=2)
    
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✓ Index saved ({file_size_mb:.2f} MB)")
    print()
    
    # Display statistics
    print("="*70)
    print("TRAINING DATASET STATISTICS")
    print("="*70)
    print(f"Total training samples:    {len(training_samples):,}")
    print(f"  From unique papers:      {index_output['dataset_info']['unique_papers']:,}")
    print(f"  Multi-figure samples:    {index_output['dataset_info']['multi_figure_samples']:,}")
    print(f"  Single-figure samples:   {index_output['dataset_info']['single_figure_samples']:,}")
    print()
    
    if caption_stats:
        print("Caption Length Statistics:")
        print(f"  Min:    {caption_stats['min_length']} words")
        print(f"  25th:   {caption_stats['p25_length']} words")
        print(f"  Median: {caption_stats['median_length']} words")
        print(f"  Mean:   {caption_stats['mean_length']:.1f} words")
        print(f"  75th:   {caption_stats['p75_length']} words")
        print(f"  Max:    {caption_stats['max_length']} words")
        print()
    
    if training_samples:
        with_refs = index_output['dataset_info']['samples_with_references']
        without_refs = index_output['dataset_info']['samples_without_references']
        print("References:")
        print(f"  With references:    {with_refs:,} ({with_refs/len(training_samples)*100:.1f}%)")
        print(f"  Without references: {without_refs:,} ({without_refs/len(training_samples)*100:.1f}%)")
        print()
    
    # Sample training samples
    print("Sample training entries (first 3):")
    for i, sample in enumerate(training_samples[:3]):
        multi_fig_info = f" (figure {sample['figure_index']+1})" if sample['is_multi_figure'] else ""
        print(f"\n  Sample {i+1}{multi_fig_info}:")
        print(f"    ID:       {sample['id']}")
        print(f"    Hash:     {sample['pdf_hash']}")
        print(f"    Image:    {sample['image_filename']}")
        print(f"    Caption:  {sample['caption'][:70]}...")
        print(f"    Length:   {sample['caption_length']} words")
        print(f"    Refs:     {sample['num_references']}")
        print(f"    Multi-fig: {'Yes' if sample['is_multi_figure'] else 'No'}")
    print()
    
    print("="*70)
    print("TRAINING INDEX COMPLETE!")
    print("="*70)
    print()
    print(f"✓ {len(training_samples):,} verified caption-image pairs ready for training")
    print(f"✓ All images verified to exist on disk")
    print(f"✓ O(1) lookup enabled via pdf_hash")
    print()
    print("Usage in training code:")
    print("  import json")
    print(f"  with open('{output_file}') as f:")
    print("      data = json.load(f)")
    print("  samples = data['samples']")
    print("  ")
    print("  # Access any sample")
    print("  sample = samples[0]")
    print("  image = Image.open(sample['image_path'])")
    print("  caption = sample['caption']")
    print()
    
    return index_output


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 create_verified_training_index.py <captions_file> [image_index_file] [output_file] [missing_hashes_file]")
        print()
        print("Default values:")
        print("  captions_file:        (required) - e.g., s2_cleaned.json")
        print("  image_index_file:     image_index.json")
        print("  output_file:          verified_training_index.json")
        print("  missing_hashes_file:  missing_hashes.txt")
        print()
        print("Example:")
        print("  python3 create_verified_training_index.py s2_cleaned.json")
        print("  python3 create_verified_training_index.py s2_cleaned.json image_index.json")
        sys.exit(1)
    
    captions_file = sys.argv[1]
    image_index_file = sys.argv[2] if len(sys.argv) > 2 else "image_index.json"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "verified_training_index.json"
    missing_hashes_file = sys.argv[4] if len(sys.argv) > 4 else "missing_hashes.txt"
    
    if not os.path.exists(captions_file):
        print(f"Error: Captions file not found: {captions_file}")
        sys.exit(1)
    
    if not os.path.exists(image_index_file):
        print(f"Error: Image index file not found: {image_index_file}")
        print(f"Run: python3 create_image_index.py <image_folder>")
        sys.exit(1)
    
    create_verified_training_index(captions_file, image_index_file, output_file, missing_hashes_file)


if __name__ == "__main__":
    main()
