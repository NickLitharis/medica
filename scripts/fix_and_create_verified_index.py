#!/usr/bin/env python3
"""
Fix pdf_hash extraction and create verified training index.
The pdf_hash in s2_cleaned.json contains full filenames, but we need just the hash part.
"""

import json
import os
import sys
from datetime import datetime

from config.bootstrap import paths


def extract_hash_from_filename(filename):
    """
    Extract the actual hash from a filename.
    Examples:
        '26491ab76c6e8d6acc582e71bb6b3b5f5601ccc2_3-Figure4-1.png' -> '26491ab76c6e8d6acc582e71bb6b3b5f5601ccc2'
        '57c9ad0f4aab133f96d40992c46926fabc901ffa_2-Figure1-1.png' -> '57c9ad0f4aab133f96d40992c46926fabc901ffa'
    """
    # Remove extension first
    name_without_ext = os.path.splitext(filename)[0]
    # Split by underscore and take first part
    hash_part = name_without_ext.split('_')[0]
    return hash_part


def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} due to JSON error: {e}")
    return data


def create_verified_training_index(captions_file, image_index_file, output_file):
    """
    Create verified training index by properly extracting hashes.
    """
    
    print("="*70)
    print("CREATING VERIFIED TRAINING INDEX (WITH HASH EXTRACTION FIX)")
    print("="*70)
    print()
    
    # Load image index
    print(f"Loading image index: {image_index_file}")
    with open(image_index_file, 'r', encoding='utf-8') as f:
        image_data = json.load(f)
        image_index = image_data['index']
        image_folder = image_data['image_folder']

    # The index may record the folder from another machine (e.g. an external
    # drive). Prefer a folder that exists here; paths are rebuilt from it.
    if not os.path.isdir(image_folder) and paths.images_dir.is_dir():
        print(f"  indexed folder not present: {image_folder}")
        print(f"  using resolved folder instead: {paths.images_dir}")
        image_folder = str(paths.images_dir)

    print(f"✓ Loaded {len(image_index):,} unique hashes")
    print(f"✓ Image folder: {image_folder}")
    print()
    
    # Load captions
    print(f"Loading captions: {captions_file}")
    captions_data = load_jsonl(captions_file)
    print(f"✓ Loaded {len(captions_data):,} caption entries")
    print()
    
    # Process and match
    print("Processing captions and matching with images...")
    verified_samples = []
    missing_hashes = []
    hash_extraction_stats = {
        'original_format': 0,  # Already just hash
        'filename_format': 0,  # Full filename, needs extraction
    }
    
    for i, entry in enumerate(captions_data):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(captions_data):,} ({(i+1)/len(captions_data)*100:.1f}%)")
        
        original_hash = entry.get('pdf_hash', '')
        if not original_hash:
            continue
        
        # Extract actual hash from filename if needed
        if '_' in original_hash or '.' in original_hash:
            # It's a full filename, extract hash
            actual_hash = extract_hash_from_filename(original_hash)
            hash_extraction_stats['filename_format'] += 1
        else:
            # It's already just the hash
            actual_hash = original_hash
            hash_extraction_stats['original_format'] += 1
        
        # Check if hash exists in image index
        if actual_hash in image_index:
            image_filename = image_index[actual_hash]
            
            # Handle multiple images per hash (take first one)
            if isinstance(image_filename, list):
                image_filename = image_filename[0]
            
            # Create verified sample
            sample = {
                'id': len(verified_samples),
                'pdf_hash': actual_hash,  # Store the clean hash
                'original_pdf_hash': original_hash,  # Keep original for reference
                'image_filename': image_filename,
                'image_path': os.path.join(image_folder, image_filename),
                'caption': entry.get('s2orc_caption', ''),
                'references': entry.get('s2orc_references', []),
                'caption_length': len(entry.get('s2orc_caption', '').split()),
                'num_references': len(entry.get('s2orc_references', []))
            }
            
            verified_samples.append(sample)
        else:
            missing_hashes.append(actual_hash)
    
    print(f"✓ Processing complete!")
    print()
    
    # Statistics
    print("="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total caption entries: {len(captions_data):,}")
    print(f"Verified samples (matched): {len(verified_samples):,} ({len(verified_samples)/len(captions_data)*100:.1f}%)")
    print(f"Missing images: {len(missing_hashes):,} ({len(missing_hashes)/len(captions_data)*100:.1f}%)")
    print()
    
    print("Hash Format Statistics:")
    print(f"  Already clean hash: {hash_extraction_stats['original_format']:,}")
    print(f"  Extracted from filename: {hash_extraction_stats['filename_format']:,}")
    print()
    
    # Caption statistics
    if verified_samples:
        caption_lengths = [s['caption_length'] for s in verified_samples]
        
        print("Caption Statistics:")
        print(f"  Min length: {min(caption_lengths)} words")
        print(f"  Max length: {max(caption_lengths)} words")
        print(f"  Mean length: {sum(caption_lengths)/len(caption_lengths):.1f} words")
        print(f"  Median length: {sorted(caption_lengths)[len(caption_lengths)//2]} words")
        print()
        
        samples_with_refs = sum(1 for s in verified_samples if s['num_references'] > 0)
        print("References:")
        print(f"  With references: {samples_with_refs:,} ({samples_with_refs/len(verified_samples)*100:.1f}%)")
        print(f"  Without references: {len(verified_samples)-samples_with_refs:,}")
        print()
    
    # Sample verified entries
    print("Sample verified entries (first 5):")
    for i, sample in enumerate(verified_samples[:5]):
        print(f"\n  {i+1}. ID: {sample['id']}")
        print(f"     Hash: {sample['pdf_hash']}")
        print(f"     Original: {sample['original_pdf_hash']}")
        print(f"     Image: {sample['image_filename']}")
        print(f"     Caption ({sample['caption_length']} words): {sample['caption'][:70]}...")
    print()
    
    # Save verified index
    print(f"Saving verified training index to: {output_file}")
    
    index_data = {
        'created': datetime.now().isoformat(),
        'source_files': {
            'captions': captions_file,
            'image_index': image_index_file
        },
        'image_folder': image_folder,
        'statistics': {
            'total_caption_entries': len(captions_data),
            'verified_samples': len(verified_samples),
            'missing_images': len(missing_hashes),
            'match_rate_percent': round(len(verified_samples)/len(captions_data)*100, 2),
            'hash_format_stats': hash_extraction_stats
        },
        'samples': verified_samples
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved verified training index ({file_size_mb:.2f} MB)")
    print()
    
    # Save missing hashes
    if missing_hashes:
        missing_file = output_file.replace('.json', '_missing_hashes.txt')
        with open(missing_file, 'w', encoding='utf-8') as f:
            for h in missing_hashes:
                f.write(f"{h}\n")
        print(f"✓ Saved {len(missing_hashes):,} missing hashes to: {missing_file}")
        print()
    
    print("="*70)
    print("VERIFIED TRAINING INDEX CREATED!")
    print("="*70)
    print()
    print(f"✓ Ready for training: {len(verified_samples):,} samples")
    print(f"✓ Output file: {output_file}")
    print()
    
    return index_data


def main():
    from config.bootstrap import check, paths

    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        captions_file = sys.argv[1]
        image_index_file = sys.argv[2] if len(sys.argv) > 2 else str(paths.image_index)
        output_file = sys.argv[3] if len(sys.argv) > 3 else str(paths.verified_index)
    else:
        if len(sys.argv) > 1:
            print("Usage: python3 scripts/fix_and_create_verified_index.py [captions_file] [image_index_file] [output_file]")
            print("With no arguments, uses MEDICA_CAPTIONS and <work_dir>/indexes/*.json.")
            print()
        if not check(paths):
            print("Attach the images and captions datasets, or set MEDICA_INPUT_DIR / MEDICA_CAPTIONS.")
            sys.exit(1)
        captions_file = str(paths.captions)
        image_index_file = str(paths.image_index)
        output_file = str(paths.verified_index)

    if not os.path.exists(captions_file):
        print(f"Error: Captions file not found: {captions_file}")
        sys.exit(1)

    if not os.path.exists(image_index_file):
        print(f"Error: Image index file not found: {image_index_file}")
        print("Run first: python3 scripts/create_image_index.py")
        sys.exit(1)

    create_verified_training_index(captions_file, image_index_file, output_file)


if __name__ == "__main__":
    main()
