#!/usr/bin/env python3
"""
Medical Image Dataset Analyzer
Analyzes images in the dataset to prepare for preprocessing.
Checks formats, sizes, resolutions, color modes, and quality.

Usage:
    python3 analyze_images.py [json_file] [image_folder]
    
    json_file: Path to cleaned JSON file (default: s2_cleaned.json)
    image_folder: Path to image folder (default: /media/marocos/C61A40931A408303/release/filtered_figures)
    
Output:
    - Console report with statistics
    - Markdown report: <json_file>_image_analysis.md
"""

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from PIL import Image
import hashlib

# Parse command line arguments
if len(sys.argv) > 1:
    json_file = sys.argv[1]
else:
    json_file = "s2_cleaned.json"

if len(sys.argv) > 2:
    image_folder = sys.argv[2]
else:
    image_folder = "/media/marocos/C61A40931A408303/release/filtered_figures"

# Auto-generate output filename
base_name = json_file.rsplit('.', 1)[0]
output_file = f"{base_name}_image_analysis.md"

print("="*70)
print("MEDICAL IMAGE DATASET ANALYZER")
print("="*70)
print(f"JSON file: {json_file}")
print(f"Image folder: {image_folder}")
print(f"Output report: {output_file}")
print()

# Check if files/folders exist
if not os.path.exists(json_file):
    print(f"❌ Error: JSON file '{json_file}' not found!")
    sys.exit(1)

if not os.path.exists(image_folder):
    print(f"❌ Error: Image folder '{image_folder}' not found!")
    sys.exit(1)

# Load JSON data
print("Loading caption data...")
captions_data = []
with open(json_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            captions_data.append(json.loads(line))
        except:
            continue

print(f"✓ Loaded {len(captions_data):,} caption entries\n")

# Build image file mapping (full filename stored)
print("Scanning image folder...")
image_files = []  # List of all image filenames
all_files_count = 0

for filename in os.listdir(image_folder):
    all_files_count += 1
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.dcm')):
        image_files.append(filename)

print(f"✓ Total files in folder: {all_files_count:,}")
print(f"✓ Image files found: {len(image_files):,}")

# Debug: show first 5 image filenames
if image_files:
    print(f"\nSample image filenames (first 5):")
    for i, filename in enumerate(image_files[:5]):
        print(f"  {i+1}. '{filename}'")
print()

# Start markdown output
md_output = []
md_output.append("# Medical Image Dataset Analysis Report")
md_output.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
md_output.append(f"\n**JSON File:** `{json_file}`")
md_output.append(f"\n**Image Folder:** `{image_folder}`")
md_output.append("\n---\n")

# Match captions to images based on pdf_hash (pdf_hash is PART of filename)
print("Matching captions to images by pdf_hash...")

# Debug: show first 5 pdf_hash values
if captions_data:
    print(f"\nSample pdf_hash values (first 5):")
    for i, entry in enumerate(captions_data[:5]):
        pdf_hash = entry.get("pdf_hash", "")
        print(f"  {i+1}. '{pdf_hash}'")
print()

matched_images = []  # List of (pdf_hash, filename) tuples
missing_images = []

for entry in captions_data:
    pdf_hash = entry.get("pdf_hash", "")
    if pdf_hash:
        # Find image filename that STARTS WITH pdf_hash
        found = False
        for filename in image_files:
            if filename.startswith(pdf_hash):
                matched_images.append((pdf_hash, filename))
                found = True
                break
        
        if not found:
            missing_images.append(pdf_hash)

match_rate = (len(matched_images) / len(captions_data) * 100) if captions_data else 0

# Debug: show first matched pairs
if matched_images:
    print(f"Sample matched pairs (first 5):")
    for i, (pdf_hash, filename) in enumerate(matched_images[:5]):
        print(f"  {i+1}. pdf_hash: '{pdf_hash}' → image: '{filename}'")
print()

# Debug: show first missing pdf_hash if any
if missing_images:
    print(f"Sample missing pdf_hash (first 5):")
    for i, pdf_hash in enumerate(missing_images[:5]):
        print(f"  {i+1}. '{pdf_hash}'")
print()

# Dataset overview
print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Caption entries: {len(captions_data):,}")
print(f"Image files in folder: {len(image_files):,}")
print(f"✓ Matched (pdf_hash → image): {len(matched_images):,} ({match_rate:.1f}%)")
print(f"✗ Missing images: {len(missing_images):,} ({100-match_rate:.1f}%)")
print()

md_output.append("## Dataset Overview\n")
md_output.append(f"- **Caption entries:** {len(captions_data):,}")
md_output.append(f"- **Image files in folder:** {len(image_files):,}")
md_output.append(f"- **Matched (pdf_hash → image):** {len(matched_images):,} ({match_rate:.1f}%)")
md_output.append(f"- **Missing images:** {len(missing_images):,} ({100-match_rate:.1f}%)")
md_output.append("\n---\n")

# Analyze ONLY matched images
print("="*70)
print("IMAGE ANALYSIS")
print("="*70)
print(f"Analyzing {len(matched_images):,} matched images...")
print("This may take a while...")
print()

formats = Counter()
modes = Counter()
sizes = []
widths = []
heights = []
aspect_ratios = []
file_sizes = []
corrupted = []

analyzed = 0
progress_interval = max(1, len(matched_images) // 20)  # 5% intervals

for i, (pdf_hash, filename) in enumerate(matched_images):
    if i > 0 and i % progress_interval == 0:
        print(f"  Progress: {i:,} / {len(matched_images):,} ({i/len(matched_images)*100:.0f}%)")
    
    filepath = os.path.join(image_folder, filename)
    
    try:
        # Get file size
        file_size = os.path.getsize(filepath)
        file_sizes.append(file_size)
        
        # Open and analyze image
        with Image.open(filepath) as img:
            # Format
            fmt = img.format if img.format else os.path.splitext(filename)[1][1:].upper()
            formats[fmt] += 1
            
            # Mode (color)
            modes[img.mode] += 1
            
            # Dimensions
            width, height = img.size
            widths.append(width)
            heights.append(height)
            sizes.append((width, height))
            
            # Aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            aspect_ratios.append(aspect_ratio)
            
            analyzed += 1
            
    except Exception as e:
        corrupted.append((filename, str(e)))

print(f"✓ Successfully analyzed: {analyzed:,} images")
if corrupted:
    print(f"✗ Corrupted/unreadable: {len(corrupted):,} images")
print()

# Statistics
md_output.append("## Image Analysis\n")
md_output.append(f"**Analyzed:** {analyzed:,} matched images (from {len(matched_images):,} total matches)\n")

if analyzed == 0:
    print("❌ No images could be analyzed!")
    md_output.append("❌ **No images could be analyzed!**\n")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_output))
    sys.exit(1)

# Format distribution
print("IMAGE FORMATS:")
md_output.append("### Image Formats\n")
md_output.append("```")
for fmt, count in formats.most_common():
    pct = (count / analyzed * 100) if analyzed else 0
    print(f"  {fmt}: {count:,} ({pct:.1f}%)")
    md_output.append(f"{fmt:10s} │ {count:,} ({pct:.1f}%)")
md_output.append("```\n")
print()

# Color modes
print("COLOR MODES:")
md_output.append("### Color Modes\n")
md_output.append("```")
for mode, count in modes.most_common():
    pct = (count / analyzed * 100) if analyzed else 0
    mode_desc = {
        "L": "Grayscale (8-bit)",
        "RGB": "RGB Color (24-bit)",
        "RGBA": "RGB with Alpha (32-bit)",
        "1": "Black & White (1-bit)",
        "P": "Palette mode",
        "CMYK": "CMYK Color",
        "I": "32-bit integer",
        "F": "32-bit float"
    }.get(mode, mode)
    print(f"  {mode} ({mode_desc}): {count:,} ({pct:.1f}%)")
    md_output.append(f"{mode:6s} ({mode_desc:25s}) │ {count:,} ({pct:.1f}%)")
md_output.append("```\n")
print()

# Dimensions
print("IMAGE DIMENSIONS:")
md_output.append("### Image Dimensions\n")

if widths and heights:
    min_width, max_width = min(widths), max(widths)
    min_height, max_height = min(heights), max(heights)
    avg_width = sum(widths) / len(widths)
    avg_height = sum(heights) / len(heights)
    median_width = sorted(widths)[len(widths)//2]
    median_height = sorted(heights)[len(heights)//2]
    
    print(f"  Width:  {min_width} - {max_width} (avg: {avg_width:.0f}, median: {median_width})")
    print(f"  Height: {min_height} - {max_height} (avg: {avg_height:.0f}, median: {median_height})")
    
    md_output.append(f"- **Width range:** {min_width} - {max_width} px")
    md_output.append(f"  - Average: {avg_width:.0f} px")
    md_output.append(f"  - Median: {median_width} px")
    md_output.append(f"- **Height range:** {min_height} - {max_height} px")
    md_output.append(f"  - Average: {avg_height:.0f} px")
    md_output.append(f"  - Median: {median_height} px")
    md_output.append("")
    
    # Common sizes
    size_counter = Counter(sizes)
    print(f"\n  Most common sizes:")
    md_output.append("**Most common resolutions:**")
    md_output.append("```")
    for size, count in size_counter.most_common(10):
        pct = (count / analyzed * 100) if analyzed else 0
        print(f"    {size[0]}x{size[1]}: {count:,} ({pct:.1f}%)")
        md_output.append(f"{size[0]:5d}x{size[1]:<5d} │ {count:,} ({pct:.1f}%)")
    md_output.append("```\n")
print()

# Aspect ratios
if aspect_ratios:
    avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
    print("ASPECT RATIOS:")
    print(f"  Average: {avg_aspect:.2f}")
    
    # Categorize
    landscape = sum(1 for ar in aspect_ratios if ar > 1.1)
    portrait = sum(1 for ar in aspect_ratios if ar < 0.9)
    square = len(aspect_ratios) - landscape - portrait
    
    print(f"  Landscape (>1.1): {landscape:,} ({landscape/len(aspect_ratios)*100:.1f}%)")
    print(f"  Square (0.9-1.1): {square:,} ({square/len(aspect_ratios)*100:.1f}%)")
    print(f"  Portrait (<0.9): {portrait:,} ({portrait/len(aspect_ratios)*100:.1f}%)")
    
    md_output.append("### Aspect Ratios\n")
    md_output.append(f"- **Average aspect ratio:** {avg_aspect:.2f}")
    md_output.append(f"- **Landscape (>1.1):** {landscape:,} ({landscape/len(aspect_ratios)*100:.1f}%)")
    md_output.append(f"- **Square (0.9-1.1):** {square:,} ({square/len(aspect_ratios)*100:.1f}%)")
    md_output.append(f"- **Portrait (<0.9):** {portrait:,} ({portrait/len(aspect_ratios)*100:.1f}%)")
    md_output.append("")
print()

# File sizes
if file_sizes:
    min_size = min(file_sizes)
    max_size = max(file_sizes)
    avg_size = sum(file_sizes) / len(file_sizes)
    median_size = sorted(file_sizes)[len(file_sizes)//2]
    
    print("FILE SIZES:")
    print(f"  Min: {min_size:,} bytes ({min_size/1024:.1f} KB)")
    print(f"  Max: {max_size:,} bytes ({max_size/1024/1024:.1f} MB)")
    print(f"  Average: {avg_size:,.0f} bytes ({avg_size/1024:.1f} KB)")
    print(f"  Median: {median_size:,} bytes ({median_size/1024:.1f} KB)")
    
    md_output.append("### File Sizes\n")
    md_output.append(f"- **Min:** {min_size:,} bytes ({min_size/1024:.1f} KB)")
    md_output.append(f"- **Max:** {max_size:,} bytes ({max_size/1024/1024:.1f} MB)")
    md_output.append(f"- **Average:** {avg_size:,.0f} bytes ({avg_size/1024:.1f} KB)")
    md_output.append(f"- **Median:** {median_size:,} bytes ({median_size/1024:.1f} KB)")
    md_output.append("")
print()

# Corrupted images
if corrupted:
    print("="*70)
    print("CORRUPTED/UNREADABLE IMAGES")
    print("="*70)
    print(f"Found {len(corrupted):,} problematic images:")
    
    md_output.append("---\n\n## Corrupted/Unreadable Images\n")
    md_output.append(f"**Found:** {len(corrupted):,} problematic images\n")
    md_output.append("```")
    
    for i, (filename, error) in enumerate(corrupted[:20], 1):  # Show first 20
        print(f"  {i}. {filename}: {error}")
        md_output.append(f"{filename}: {error}")
    
    if len(corrupted) > 20:
        print(f"  ... and {len(corrupted) - 20} more")
        md_output.append(f"... and {len(corrupted) - 20} more")
    
    md_output.append("```")
    print()

# Recommendations
print("="*70)
print("PREPROCESSING RECOMMENDATIONS")
print("="*70)

md_output.append("\n---\n\n## Preprocessing Recommendations\n")

recommendations = []

# Format recommendation
if len(formats) > 1:
    primary_format = formats.most_common(1)[0][0]
    rec = f"Convert all images to {primary_format} format for consistency"
    recommendations.append(rec)
    print(f"✓ {rec}")

# Size recommendation
if widths and heights:
    if max_width - min_width > 1000 or max_height - min_height > 1000:
        rec = f"Resize images to consistent dimensions (suggest: {median_width}x{median_height} or 512x512)"
        recommendations.append(rec)
        print(f"✓ {rec}")

# Color mode recommendation
if len(modes) > 1:
    if 'RGB' in modes and 'L' in modes:
        rec = "Convert all images to RGB for consistency (or grayscale if appropriate)"
        recommendations.append(rec)
        print(f"✓ {rec}")

# Corrupted images
if corrupted:
    rec = f"Remove or fix {len(corrupted):,} corrupted images"
    recommendations.append(rec)
    print(f"✓ {rec}")

# Missing images
if missing_images:
    rec = f"Resolve {len(missing_images):,} missing images or remove from dataset"
    recommendations.append(rec)
    print(f"✓ {rec}")

# Normalization
rec = "Apply pixel intensity normalization (0-1 or mean/std)"
recommendations.append(rec)
print(f"✓ {rec}")

# Medical specific
rec = "Consider CLAHE (Contrast Limited Adaptive Histogram Equalization) for X-rays"
recommendations.append(rec)
print(f"✓ {rec}")

rec = "Apply data augmentation during training (rotation, flip, brightness, contrast)"
recommendations.append(rec)
print(f"✓ {rec}")

# Add to markdown
for rec in recommendations:
    md_output.append(f"- {rec}")

print()

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Analysis complete!")
print(f"✓ Report saved to: {output_file}")
print()
print("Next steps:")
print("  1. Review the analysis report")
print("  2. Create preprocessing pipeline based on recommendations")
print("  3. Process all images before ML training")
print("="*70)

md_output.append("\n---\n\n## Summary\n")
md_output.append(f"- **Total images analyzed:** {analyzed:,}")
md_output.append(f"- **Matched caption-image pairs:** {len(matched_images):,}")
md_output.append(f"- **Ready for preprocessing:** {'✅ Yes' if match_rate > 95 else '⚠️ Check missing images'}")
md_output.append(f"\n**Next step:** Create preprocessing pipeline based on these findings.")

# Write markdown report
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(md_output))

print(f"\n✓ Markdown report saved: {output_file}")
