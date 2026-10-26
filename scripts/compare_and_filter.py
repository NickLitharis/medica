#!/usr/bin/env python3
"""
Compare image index with s2_cleaned.json captions.
Identifies and separates entries that don't have matching images.
"""

import json
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


def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def compare_and_filter(captions_file, index_file, output_missing, output_matched):
    """
    Compare captions with image index.
    Separate entries into those with and without matching images.
    """
    
    print("="*70)
    print("COMPARING CAPTIONS WITH IMAGE INDEX")
    print("="*70)
    print()
    
    # Load caption data
    print(f"Loading captions from: {captions_file}")
    captions_data = load_jsonl(captions_file)
    print(f"✓ Loaded {len(captions_data):,} caption entries")
    print()
    
    # Load image index
    print(f"Loading image index from: {index_file}")
    index_data = load_json(index_file)
    image_index = index_data['index']
    print(f"✓ Loaded {len(image_index):,} image entries")
    print()
    
    # Compare and separate
    print("Comparing pdf_hash values...")
    matched_entries = []
    missing_entries = []
    
    for i, entry in enumerate(captions_data):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(captions_data):,} ({(i+1)/len(captions_data)*100:.1f}%)")
        
        pdf_hash = entry.get('pdf_hash', '')
        
        if pdf_hash and pdf_hash in image_index:
            # Image exists - add filename info to entry
            entry['image_filename'] = image_index[pdf_hash]
            matched_entries.append(entry)
        else:
            # Image missing
            missing_entries.append(entry)
    
    print(f"✓ Comparison complete!")
    print()
    
    # Statistics
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total caption entries: {len(captions_data):,}")
    print(f"Matched (have images): {len(matched_entries):,} ({len(matched_entries)/len(captions_data)*100:.1f}%)")
    print(f"Missing (no images):   {len(missing_entries):,} ({len(missing_entries)/len(captions_data)*100:.1f}%)")
    print()
    
    # Sample missing entries
    if missing_entries:
        print("Sample missing entries (first 10 pdf_hash values):")
        for i, entry in enumerate(missing_entries[:10]):
            pdf_hash = entry.get('pdf_hash', 'NO_HASH')
            caption = entry.get('s2orc_caption', '')[:60]
            print(f"  {i+1}. {pdf_hash} - {caption}...")
        print()
    
    # Save matched entries
    if matched_entries:
        print(f"Saving matched entries to: {output_matched}")
        save_jsonl(matched_entries, output_matched)
        print(f"✓ Saved {len(matched_entries):,} matched entries")
        print()
    
    # Save missing entries
    if missing_entries:
        print(f"Saving missing entries to: {output_missing}")
        save_jsonl(missing_entries, output_missing)
        print(f"✓ Saved {len(missing_entries):,} missing entries")
        print()
    
    # Create summary report
    summary = {
        "comparison_date": datetime.now().isoformat(),
        "captions_file": captions_file,
        "index_file": index_file,
        "statistics": {
            "total_captions": len(captions_data),
            "matched_with_images": len(matched_entries),
            "missing_images": len(missing_entries),
            "match_rate_percent": round(len(matched_entries)/len(captions_data)*100, 2)
        },
        "output_files": {
            "matched": output_matched,
            "missing": output_missing
        }
    }
    
    summary_file = "comparison_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_file}")
    print()
    
    print("="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print()
    print("Output files:")
    print(f"  1. Matched (with images): {output_matched}")
    print(f"  2. Missing (no images):   {output_missing}")
    print(f"  3. Summary report:        {summary_file}")
    print()
    
    return matched_entries, missing_entries


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_and_filter.py <captions_file> [index_file] [output_missing] [output_matched]")
        print()
        print("Default values:")
        print("  captions_file: (required)")
        print("  index_file:    image_index.json")
        print("  output_missing: s2_cleaned_missing_images.json")
        print("  output_matched: s2_cleaned_with_images.json")
        print()
        print("Example:")
        print("  python3 compare_and_filter.py s2_cleaned.json")
        sys.exit(1)
    
    captions_file = sys.argv[1]
    index_file = sys.argv[2] if len(sys.argv) > 2 else "image_index.json"
    output_missing = sys.argv[3] if len(sys.argv) > 3 else "s2_cleaned_missing_images.json"
    output_matched = sys.argv[4] if len(sys.argv) > 4 else "s2_cleaned_with_images.json"
    
    compare_and_filter(captions_file, index_file, output_missing, output_matched)


if __name__ == "__main__":
    main()
