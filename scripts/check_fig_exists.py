#!/usr/bin/env python3
"""
Check if files matching pdf_hash exist in a specified folder.
Ultra-optimized version with multiple speed improvements.
"""

import json
import os
import sys

# Configuration - edit these
json_file = "s2_final.json"
search_folder = "/media/marocos/C61A40931A408303/release/figures"
progress_interval = 10000  # Print progress every N entries

print(f"Reading from: {json_file}")
print(f"Searching in folder: {search_folder}")

# Check if the folder exists
if not os.path.isdir(search_folder):
    print(f"ERROR: Folder '{search_folder}' does not exist")
    sys.exit(1)

# Check if JSON file exists
if not os.path.isfile(json_file):
    print(f"ERROR: JSON file '{json_file}' does not exist")
    sys.exit(1)

# Preload all filenames in the folder
print("Loading directory listing...")
all_files = frozenset(os.listdir(search_folder))  # frozenset is slightly faster than set for read-only
print(f"Loaded {len(all_files)} files from directory")
print()

# Read JSONL file and check each entry
total = 0
found_count = 0
missing_count = 0
missing_list = []

# Pre-compile method lookups for speed
get_method = dict.get

print("Processing entries...")
with open(json_file, "r", encoding="utf-8", buffering=65536) as f:  # Larger buffer for faster I/O
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        
        total += 1
        
        # Progress indicator
        if total % progress_interval == 0:
            print(f"Processed {total:,} entries... (found: {found_count:,}, missing: {missing_count:,})", flush=True)
        
        pdf_hash = get_method(data, "pdf_hash", "")
        
        if not pdf_hash:
            continue
        
        # Ultra-fast lookup: check if any filename contains the pdf_hash
        # Using generator with any() stops at first match (short-circuit evaluation)
        found = any(pdf_hash in filename for filename in all_files)
        
        if found:
            found_count += 1
        else:
            missing_count += 1
            missing_list.append(pdf_hash)
            # Only print missing (reduces I/O bottleneck)
            # Uncomment next line if you want real-time output:
            # print(f"✗ MISSING - {pdf_hash}")

print()
print("="*50)
print("SUMMARY")
print("="*50)
print(f"Total entries: {total:,}")
print(f"Found: {found_count:,}")
print(f"Missing: {missing_count:,}")

# Optionally print all missing hashes at the end
if missing_list:
    print(f"\nMissing entries ({len(missing_list)}):")
    for pdf_hash in missing_list:
        print(f"  ✗ {pdf_hash}")
