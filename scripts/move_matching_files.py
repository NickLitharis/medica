#!/usr/bin/env python3
"""
Move files matching pdf_hash from source to destination folder.
Reads JSONL file and moves only files that exist and match entries.
"""

import json
import os
import sys
import shutil

from config.bootstrap import paths

# Optional arguments: [json_file] [source_folder] [dest_folder]
# Defaults come from config/paths.py (Kaggle-aware). Kaggle input is
# read-only, so the destination defaults to the writable work folder.
json_file = sys.argv[1] if len(sys.argv) > 1 else str(paths.captions)
source_folder = sys.argv[2] if len(sys.argv) > 2 else str(paths.images_dir)
dest_folder = sys.argv[3] if len(sys.argv) > 3 else str(paths.work_dir / "filtered_figures")
progress_interval = 10000  # Print progress every N entries
dry_run = False  # Set to True to see what would be moved without actually moving

print(f"Reading from: {json_file}")
print(f"Source folder: {source_folder}")
print(f"Destination folder: {dest_folder}")
print(f"Dry run mode: {dry_run}")
print()

# Check if source folder exists
if not os.path.isdir(source_folder):
    print(f"ERROR: Source folder '{source_folder}' does not exist")
    sys.exit(1)

# Check if JSON file exists
if not os.path.isfile(json_file):
    print(f"ERROR: JSON file '{json_file}' does not exist")
    sys.exit(1)

# Create destination folder if it doesn't exist
if not dry_run:
    os.makedirs(dest_folder, exist_ok=True)
    print(f"Created/verified destination folder: {dest_folder}")
else:
    print("DRY RUN: Would create destination folder if needed")
print()

# Preload all filenames from source folder
print("Loading source directory listing...")
all_files = frozenset(os.listdir(source_folder))
print(f"Loaded {len(all_files)} files from source directory")
print()

# Process entries
total = 0
found_count = 0
moved_count = 0
missing_count = 0
error_count = 0

print("Processing entries...")
with open(json_file, "r", encoding="utf-8", buffering=65536) as f:
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
            print(f"Processed {total:,} entries... (moved: {moved_count:,}, missing: {missing_count:,})", flush=True)
        
        pdf_hash = data.get("pdf_hash", "")
        
        if not pdf_hash:
            continue
        
        # Find matching file(s)
        matching_files = [filename for filename in all_files if pdf_hash in filename]
        
        if matching_files:
            found_count += 1
            # Move each matching file
            for filename in matching_files:
                source_path = os.path.join(source_folder, filename)
                dest_path = os.path.join(dest_folder, filename)
                
                try:
                    if dry_run:
                        print(f"  [DRY RUN] Would move: {filename}")
                    else:
                        shutil.move(source_path, dest_path)
                    moved_count += 1
                except Exception as e:
                    print(f"  ERROR moving {filename}: {e}")
                    error_count += 1
        else:
            missing_count += 1

print()
print("="*50)
print("SUMMARY")
print("="*50)
print(f"Total entries processed: {total:,}")
print(f"Files found: {found_count:,}")
print(f"Files moved: {moved_count:,}")
print(f"Files missing: {missing_count:,}")
print(f"Errors: {error_count:,}")

if dry_run:
    print("\nThis was a DRY RUN - no files were actually moved.")
    print("Set dry_run = False to perform the actual move.")
