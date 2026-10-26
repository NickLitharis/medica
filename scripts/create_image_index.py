#!/usr/bin/env python3
"""
Create an O(1) lookup index for images.
Maps pdf_hash to image filename for instant retrieval.
"""

import os
import sys
import json
import time
from datetime import datetime


def create_image_index(image_folder, output_file="image_index.json"):
    """
    Create a hash map for O(1) image lookup.
    Maps pdf_hash (from filename) to full filename.
    """
    
    print("="*70)
    print("CREATING IMAGE INDEX")
    print("="*70)
    print(f"Scanning folder: {image_folder}")
    print()
    
    # Benchmark first to estimate time
    print("Running quick benchmark...")
    start_benchmark = time.time()
    all_files = os.listdir(image_folder)
    benchmark_time = time.time() - start_benchmark
    total_files = len(all_files)
    
    print(f"✓ Found {total_files:,} files")
    print(f"✓ Folder scan took: {benchmark_time:.2f} seconds")
    print()
    
    # Estimate processing time (very rough: ~10-50 microseconds per file for dict operations)
    estimated_time = benchmark_time + (total_files * 0.00002)  # 20 microseconds per file
    print(f"Estimated total time: {estimated_time:.2f} seconds")
    print("Starting indexing...\n")
    
    start_time = time.time()
    image_index = {}
    processed = 0
    
    # Scan all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.dcm'}
    
    for filename in all_files:
        processed += 1
        
        # Progress indicator every 10,000 files
        if processed % 10000 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (total_files - processed) / rate
            print(f"  Progress: {processed:,}/{total_files:,} ({processed/total_files*100:.1f}%) "
                  f"- {rate:.0f} files/sec - ETA: {remaining:.1f}s")
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in image_extensions:
            # Extract pdf_hash (everything before first underscore or dot)
            # Handles formats like: <hash>.png or <hash>_figure_1.png
            pdf_hash = filename.split('_')[0].split('.')[0]
            
            # Store in index (if multiple images per hash, keep as list)
            if pdf_hash in image_index:
                # Multiple images for same pdf_hash
                if isinstance(image_index[pdf_hash], list):
                    image_index[pdf_hash].append(filename)
                else:
                    image_index[pdf_hash] = [image_index[pdf_hash], filename]
            else:
                image_index[pdf_hash] = filename
    
    total_time = time.time() - start_time
    
    print()
    print(f"✓ Indexing complete in {total_time:.2f} seconds")
    print(f"✓ Processing rate: {total_files/total_time:.0f} files/second")
    print()
    
    # Save index to JSON
    print("Saving index to JSON...")
    save_start = time.time()
    index_data = {
        "created": datetime.now().isoformat(),
        "image_folder": image_folder,
        "total_images": len([f for f in os.listdir(image_folder) 
                            if os.path.splitext(f)[1].lower() in image_extensions]),
        "unique_hashes": len(image_index),
        "index": image_index
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    
    save_time = time.time() - save_start
    
    print(f"✓ JSON saved in {save_time:.2f} seconds")
    print()
    print(f"✓ Total images: {index_data['total_images']:,}")
    print(f"✓ Unique pdf_hash values: {index_data['unique_hashes']:,}")
    print(f"✓ Index saved to: {output_file}")
    print(f"✓ Total execution time: {time.time() - start_time + benchmark_time:.2f} seconds")
    print()
    
    # Statistics
    single_image_hashes = sum(1 for v in image_index.values() if isinstance(v, str))
    multi_image_hashes = sum(1 for v in image_index.values() if isinstance(v, list))
    
    print("Statistics:")
    print(f"  Single image per hash: {single_image_hashes:,}")
    print(f"  Multiple images per hash: {multi_image_hashes:,}")
    
    if multi_image_hashes > 0:
        print(f"\nSample multi-image hashes (first 5):")
        count = 0
        for pdf_hash, files in image_index.items():
            if isinstance(files, list):
                print(f"  {pdf_hash}: {len(files)} images")
                for f in files:
                    print(f"    - {f}")
                count += 1
                if count >= 5:
                    break
    print()
    
    # Show sample lookups
    print("Sample lookups (first 5 hashes):")
    for i, (pdf_hash, filename) in enumerate(list(image_index.items())[:5]):
        if isinstance(filename, list):
            print(f"  {pdf_hash} → {filename[0]} (+ {len(filename)-1} more)")
        else:
            print(f"  {pdf_hash} → {filename}")
    print()
    
    print("="*70)
    print("INDEX CREATED SUCCESSFULLY!")
    print("="*70)
    print("\nUsage example:")
    print("  import json")
    print("  with open('image_index.json') as f:")
    print("      index = json.load(f)['index']")
    print("  filename = index['your_pdf_hash']  # O(1) lookup!")
    print("  image_path = os.path.join(image_folder, filename)")
    print()
    
    return image_index


def lookup_image(pdf_hash, index_file="image_index.json"):
    """
    O(1) lookup of image filename by pdf_hash.
    """
    with open(index_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        image_folder = data['image_folder']
        index = data['index']
    
    if pdf_hash in index:
        filename = index[pdf_hash]
        if isinstance(filename, list):
            # Multiple images, return first one
            return os.path.join(image_folder, filename[0]), filename
        else:
            return os.path.join(image_folder, filename), filename
    else:
        return None, None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 create_image_index.py <image_folder> [output_file]")
        print("Example: python3 create_image_index.py /path/to/images")
        print("\nThis creates a hash map for O(1) image lookup by pdf_hash.")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "image_index.json"
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found: {image_folder}")
        sys.exit(1)
    
    create_image_index(image_folder, output_file)


if __name__ == "__main__":
    main()
