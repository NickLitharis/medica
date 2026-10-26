#!/usr/bin/env python3
"""
Comprehensive medical image analysis.
Analyzes all images in a folder for format, dimensions, quality, and characteristics.
No caption matching required - pure image analysis.
"""

import os
import sys
from PIL import Image
from collections import Counter, defaultdict
from datetime import datetime
import math


def format_bytes(size):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def create_histogram(values, bins=10, max_width=60, title=""):
    """Create ASCII histogram."""
    if not values:
        return "No data"
    
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    
    if range_val == 0:
        return f"All values are {min_val}"
    
    bin_width = range_val / bins
    bin_counts = [0] * bins
    
    for val in values:
        bin_idx = min(int((val - min_val) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1
    
    max_count = max(bin_counts)
    lines = []
    if title:
        lines.append(title)
    
    for i, count in enumerate(bin_counts):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_len = int((count / max_count) * max_width) if max_count > 0 else 0
        bar = '█' * bar_len
        lines.append(f"{bin_start:7.1f}-{bin_end:7.1f} | {bar} {count:,}")
    
    return '\n'.join(lines)


def analyze_images(image_folder):
    """Comprehensively analyze all images in folder."""
    
    print("="*70)
    print("SCANNING IMAGE FOLDER")
    print("="*70)
    
    image_files = []
    all_files = []
    
    for filename in os.listdir(image_folder):
        all_files.append(filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.dcm')):
            image_files.append(filename)
    
    print(f"✓ Total files in folder: {len(all_files):,}")
    print(f"✓ Image files found: {len(image_files):,}")
    print()
    
    if not image_files:
        print("No images found!")
        return None
    
    # Sample filenames
    print("Sample filenames (first 10):")
    for i, filename in enumerate(image_files[:10]):
        print(f"  {i+1}. {filename}")
    print()
    
    # Analyze images
    print("="*70)
    print("ANALYZING IMAGE CHARACTERISTICS")
    print("="*70)
    
    formats = Counter()
    modes = Counter()
    widths = []
    heights = []
    aspect_ratios = []
    megapixels = []
    file_sizes = []  # File size in bytes
    file_extensions = Counter()
    bit_depths = Counter()
    corrupted = []
    
    # Additional metrics
    square_images = 0
    portrait_images = 0
    landscape_images = 0
    very_small = []  # < 256x256
    very_large = []  # > 4096x4096
    
    for i, filename in enumerate(image_files):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{len(image_files):,} images...")
        
        img_path = os.path.join(image_folder, filename)
        
        try:
            # Get file extension
            ext = os.path.splitext(filename)[1].lower()
            file_extensions[ext] += 1
            
            # Get file size
            file_size = os.path.getsize(img_path)
            file_sizes.append(file_size)
            
            # Open and analyze image
            with Image.open(img_path) as img:
                formats[img.format if img.format else 'UNKNOWN'] += 1
                modes[img.mode] += 1
                
                # Get bit depth
                if hasattr(img, 'bits'):
                    bit_depths[img.bits] += 1
                elif img.mode == 'L':
                    bit_depths[8] += 1
                elif img.mode in ['RGB', 'RGBA']:
                    bit_depths[24] += 1
                elif img.mode == '1':
                    bit_depths[1] += 1
                else:
                    bit_depths['Unknown'] += 1
                
                # Dimensions
                width, height = img.size
                widths.append(width)
                heights.append(height)
                
                # Aspect ratio
                aspect = width / height if height > 0 else 0
                aspect_ratios.append(aspect)
                
                # Megapixels
                mp = (width * height) / 1_000_000
                megapixels.append(mp)
                
                # Orientation
                if width == height:
                    square_images += 1
                elif width > height:
                    landscape_images += 1
                else:
                    portrait_images += 1
                
                # Size categories
                if width < 256 or height < 256:
                    very_small.append(filename)
                if width > 4096 or height > 4096:
                    very_large.append(filename)
            
        except Exception as e:
            corrupted.append((filename, str(e)))
    
    print(f"✓ Completed analyzing {len(image_files):,} images")
    print()
    
    # Calculate statistics
    md_output = []
    md_output.append(f"# Comprehensive Image Analysis Report")
    md_output.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    md_output.append(f"*Image Folder: {image_folder}*\n")
    
    print("="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    print(f"Total files in folder: {len(all_files):,}")
    print(f"Total image files: {len(image_files):,}")
    print(f"Successfully analyzed: {len(widths):,}")
    print(f"Corrupted/unreadable: {len(corrupted):,}")
    print()

    md_output.append("## Dataset Overview\n")
    md_output.append(f"- **Total files in folder:** {len(all_files):,}")
    md_output.append(f"- **Total image files:** {len(image_files):,}")
    md_output.append(f"- **Successfully analyzed:** {len(widths):,}")
    md_output.append(f"- **Corrupted/unreadable:** {len(corrupted):,}")
    md_output.append("\n---\n")
    
    # File extensions
    print("="*70)
    print("FILE EXTENSIONS")
    print("="*70)
    for ext, count in file_extensions.most_common():
        pct = count / len(image_files) * 100
        print(f"  {ext}: {count:,} ({pct:.1f}%)")
    print()
    
    md_output.append("## File Extensions\n")
    md_output.append("```")
    for ext, count in file_extensions.most_common():
        pct = count / len(image_files) * 100
        md_output.append(f"{ext}: {count:,} ({pct:.1f}%)")
    md_output.append("```\n")
    
    # Image formats
    print("="*70)
    print("IMAGE FORMATS (detected by PIL)")
    print("="*70)
    for fmt, count in formats.most_common():
        pct = count / len(widths) * 100 if widths else 0
        print(f"  {fmt}: {count:,} ({pct:.1f}%)")
    print()
    
    md_output.append("## Image Formats (PIL Detection)\n")
    md_output.append("```")
    for fmt, count in formats.most_common():
        pct = count / len(widths) * 100 if widths else 0
        md_output.append(f"{fmt}: {count:,} ({pct:.1f}%)")
    md_output.append("```\n")
    
    # Color modes
    print("="*70)
    print("COLOR MODES")
    print("="*70)
    mode_descriptions = {
        'L': 'Grayscale (8-bit)',
        'RGB': 'RGB Color (24-bit)',
        'RGBA': 'RGB + Alpha (32-bit)',
        'CMYK': 'CMYK Color',
        '1': 'Binary (1-bit)',
        'P': 'Palette (8-bit)',
        'LA': 'Grayscale + Alpha',
        'I': 'Integer pixels (32-bit)',
        'F': 'Floating point (32-bit)'
    }
    for mode, count in modes.most_common():
        pct = count / len(widths) * 100 if widths else 0
        desc = mode_descriptions.get(mode, 'Unknown')
        print(f"  {mode} ({desc}): {count:,} ({pct:.1f}%)")
    print()
    
    md_output.append("## Color Modes\n")
    md_output.append("```")
    for mode, count in modes.most_common():
        pct = count / len(widths) * 100 if widths else 0
        desc = mode_descriptions.get(mode, 'Unknown')
        md_output.append(f"{mode} ({desc}): {count:,} ({pct:.1f}%)")
    md_output.append("```\n")
    
    # Bit depths
    if bit_depths:
        print("="*70)
        print("BIT DEPTHS")
        print("="*70)
        for bits, count in sorted(bit_depths.items()):
            pct = count / len(widths) * 100 if widths else 0
            print(f"  {bits}-bit: {count:,} ({pct:.1f}%)")
        print()
        
        md_output.append("## Bit Depths\n")
        md_output.append("```")
        for bits, count in sorted(bit_depths.items()):
            pct = count / len(widths) * 100 if widths else 0
            md_output.append(f"{bits}-bit: {count:,} ({pct:.1f}%)")
        md_output.append("```\n")
    
    # Dimensions
    if widths and heights:
        print("="*70)
        print("DIMENSIONS")
        print("="*70)
        
        # Calculate statistics
        sorted_widths = sorted(widths)
        sorted_heights = sorted(heights)
        
        width_mean = sum(widths) / len(widths)
        height_mean = sum(heights) / len(heights)
        width_median = sorted_widths[len(widths) // 2]
        height_median = sorted_heights[len(heights) // 2]
        width_std = math.sqrt(sum((w - width_mean) ** 2 for w in widths) / len(widths))
        height_std = math.sqrt(sum((h - height_mean) ** 2 for h in heights) / len(heights))
        
        print(f"Width Statistics:")
        print(f"  Min: {min(widths):,} px")
        print(f"  Max: {max(widths):,} px")
        print(f"  Mean: {width_mean:.0f} px")
        print(f"  Median: {width_median:,} px")
        print(f"  Std Dev: {width_std:.0f} px")
        print()
        
        print(f"Height Statistics:")
        print(f"  Min: {min(heights):,} px")
        print(f"  Max: {max(heights):,} px")
        print(f"  Mean: {height_mean:.0f} px")
        print(f"  Median: {height_median:,} px")
        print(f"  Std Dev: {height_std:.0f} px")
        print()
        
        # Width histogram
        print("Width Distribution:")
        print(create_histogram(widths, bins=15, max_width=50))
        print()
        
        # Height histogram
        print("Height Distribution:")
        print(create_histogram(heights, bins=15, max_width=50))
        print()
        
        # Most common dimensions
        dimension_pairs = Counter(zip(widths, heights))
        print(f"Most Common Dimensions (Top 15):")
        for (w, h), count in dimension_pairs.most_common(15):
            pct = count / len(widths) * 100
            print(f"  {w:4d} x {h:4d}: {count:6,} ({pct:5.2f}%)")
        print()
        
        # Markdown output
        md_output.append("## Dimensions Analysis\n")
        md_output.append("### Width Statistics\n")
        md_output.append(f"- **Min:** {min(widths):,} px")
        md_output.append(f"- **Max:** {max(widths):,} px")
        md_output.append(f"- **Mean:** {width_mean:.0f} px")
        md_output.append(f"- **Median:** {width_median:,} px")
        md_output.append(f"- **Std Dev:** {width_std:.0f} px\n")
        
        md_output.append("### Height Statistics\n")
        md_output.append(f"- **Min:** {min(heights):,} px")
        md_output.append(f"- **Max:** {max(heights):,} px")
        md_output.append(f"- **Mean:** {height_mean:.0f} px")
        md_output.append(f"- **Median:** {height_median:,} px")
        md_output.append(f"- **Std Dev:** {height_std:.0f} px\n")
        
        md_output.append("### Width Distribution\n```")
        md_output.append(create_histogram(widths, bins=15, max_width=60))
        md_output.append("```\n")
        
        md_output.append("### Height Distribution\n```")
        md_output.append(create_histogram(heights, bins=15, max_width=60))
        md_output.append("```\n")
        
        md_output.append("### Most Common Dimensions (Top 15)\n```")
        for (w, h), count in dimension_pairs.most_common(15):
            pct = count / len(widths) * 100
            md_output.append(f"{w:4d} x {h:4d}: {count:6,} ({pct:5.2f}%)")
        md_output.append("```\n")
    
    # Aspect ratios and orientation
    if aspect_ratios:
        print("="*70)
        print("ASPECT RATIOS & ORIENTATION")
        print("="*70)
        
        sorted_ar = sorted(aspect_ratios)
        ar_mean = sum(aspect_ratios) / len(aspect_ratios)
        ar_median = sorted_ar[len(aspect_ratios) // 2]
        ar_std = math.sqrt(sum((ar - ar_mean) ** 2 for ar in aspect_ratios) / len(aspect_ratios))
        
        print(f"Aspect Ratio Statistics:")
        print(f"  Min: {min(aspect_ratios):.3f}")
        print(f"  Max: {max(aspect_ratios):.3f}")
        print(f"  Mean: {ar_mean:.3f}")
        print(f"  Median: {ar_median:.3f}")
        print(f"  Std Dev: {ar_std:.3f}")
        print()
        
        print(f"Orientation Distribution:")
        print(f"  Square (1:1, AR 0.95-1.05): {square_images:,} ({square_images/len(aspect_ratios)*100:.1f}%)")
        print(f"  Portrait (H>W, AR <0.95): {portrait_images:,} ({portrait_images/len(aspect_ratios)*100:.1f}%)")
        print(f"  Landscape (W>H, AR >1.05): {landscape_images:,} ({landscape_images/len(aspect_ratios)*100:.1f}%)")
        print()
        
        # Aspect ratio histogram
        print("Aspect Ratio Distribution:")
        print(create_histogram(aspect_ratios, bins=20, max_width=50))
        print()
        
        md_output.append("## Aspect Ratios & Orientation\n")
        md_output.append("### Aspect Ratio Statistics\n")
        md_output.append(f"- **Min:** {min(aspect_ratios):.3f}")
        md_output.append(f"- **Max:** {max(aspect_ratios):.3f}")
        md_output.append(f"- **Mean:** {ar_mean:.3f}")
        md_output.append(f"- **Median:** {ar_median:.3f}")
        md_output.append(f"- **Std Dev:** {ar_std:.3f}\n")
        
        md_output.append("### Orientation Distribution\n")
        md_output.append(f"- **Square (1:1, AR 0.95-1.05):** {square_images:,} ({square_images/len(aspect_ratios)*100:.1f}%)")
        md_output.append(f"- **Portrait (H>W, AR <0.95):** {portrait_images:,} ({portrait_images/len(aspect_ratios)*100:.1f}%)")
        md_output.append(f"- **Landscape (W>H, AR >1.05):** {landscape_images:,} ({landscape_images/len(aspect_ratios)*100:.1f}%)\n")
        
        md_output.append("### Aspect Ratio Distribution\n```")
        md_output.append(create_histogram(aspect_ratios, bins=20, max_width=60))
        md_output.append("```\n")
    
    # Megapixels
    if megapixels:
        print("="*70)
        print("MEGAPIXELS (Resolution)")
        print("="*70)
        
        sorted_mp = sorted(megapixels)
        mp_mean = sum(megapixels) / len(megapixels)
        mp_median = sorted_mp[len(megapixels) // 2]
        mp_std = math.sqrt(sum((mp - mp_mean) ** 2 for mp in megapixels) / len(megapixels))
        
        print(f"Megapixel Statistics:")
        print(f"  Min: {min(megapixels):.3f} MP")
        print(f"  Max: {max(megapixels):.3f} MP")
        print(f"  Mean: {mp_mean:.3f} MP")
        print(f"  Median: {mp_median:.3f} MP")
        print(f"  Std Dev: {mp_std:.3f} MP")
        print()
        
        # Megapixel histogram
        print("Megapixel Distribution:")
        print(create_histogram(megapixels, bins=20, max_width=50))
        print()
        
        md_output.append("## Megapixels (Resolution)\n")
        md_output.append("### Statistics\n")
        md_output.append(f"- **Min:** {min(megapixels):.3f} MP")
        md_output.append(f"- **Max:** {max(megapixels):.3f} MP")
        md_output.append(f"- **Mean:** {mp_mean:.3f} MP")
        md_output.append(f"- **Median:** {mp_median:.3f} MP")
        md_output.append(f"- **Std Dev:** {mp_std:.3f} MP\n")
        
        md_output.append("### Megapixel Distribution\n```")
        md_output.append(create_histogram(megapixels, bins=20, max_width=60))
        md_output.append("```\n")
    
    # File sizes
    if file_sizes:
        print("="*70)
        print("FILE SIZES")
        print("="*70)
        
        sorted_fs = sorted(file_sizes)
        total_size = sum(file_sizes)
        avg_size = total_size / len(file_sizes)
        median_size = sorted_fs[len(file_sizes) // 2]
        fs_std = math.sqrt(sum((fs - avg_size) ** 2 for fs in file_sizes) / len(file_sizes))
        
        print(f"Total Dataset Size: {format_bytes(total_size)}")
        print(f"\nFile Size Statistics:")
        print(f"  Min: {format_bytes(min(file_sizes))}")
        print(f"  Max: {format_bytes(max(file_sizes))}")
        print(f"  Mean: {format_bytes(avg_size)}")
        print(f"  Median: {format_bytes(median_size)}")
        print(f"  Std Dev: {format_bytes(fs_std)}")
        print()
        
        # File size histogram
        print("File Size Distribution:")
        print(create_histogram([fs / 1024 for fs in file_sizes], bins=20, max_width=50, title="(in KB)"))
        print()
        
        md_output.append("## File Sizes\n")
        md_output.append(f"**Total Dataset Size:** {format_bytes(total_size)}\n")
        md_output.append("### Statistics\n")
        md_output.append(f"- **Min:** {format_bytes(min(file_sizes))}")
        md_output.append(f"- **Max:** {format_bytes(max(file_sizes))}")
        md_output.append(f"- **Mean:** {format_bytes(avg_size)}")
        md_output.append(f"- **Median:** {format_bytes(median_size)}")
        md_output.append(f"- **Std Dev:** {format_bytes(fs_std)}\n")
        
        md_output.append("### File Size Distribution (KB)\n```")
        md_output.append(create_histogram([fs / 1024 for fs in file_sizes], bins=20, max_width=60))
        md_output.append("```\n")
    
    # Size categories
    print("="*70)
    print("SIZE CATEGORIES")
    print("="*70)
    print(f"Very Small (<256px): {len(very_small):,} ({len(very_small)/len(image_files)*100:.2f}%)")
    print(f"Very Large (>4096px): {len(very_large):,} ({len(very_large)/len(image_files)*100:.2f}%)")
    print()
    
    if very_small:
        print(f"Sample Very Small Images (first 10):")
        for filename in very_small[:10]:
            print(f"  {filename}")
        print()
    
    if very_large:
        print(f"Sample Very Large Images (first 10):")
        for filename in very_large[:10]:
            print(f"  {filename}")
        print()
    
    md_output.append("## Size Categories\n")
    md_output.append(f"- **Very Small (<256px):** {len(very_small):,} ({len(very_small)/len(image_files)*100:.2f}%)")
    md_output.append(f"- **Very Large (>4096px):** {len(very_large):,} ({len(very_large)/len(image_files)*100:.2f}%)\n")
    
    if very_small:
        md_output.append("### Sample Very Small Images (first 10)\n```")
        for filename in very_small[:10]:
            md_output.append(filename)
        md_output.append("```\n")
    
    if very_large:
        md_output.append("### Sample Very Large Images (first 10)\n```")
        for filename in very_large[:10]:
            md_output.append(filename)
        md_output.append("```\n")
    
    # Corrupted images
    if corrupted:
        print("="*70)
        print(f"CORRUPTED/UNREADABLE IMAGES ({len(corrupted)})")
        print("="*70)
        for filename, error in corrupted[:20]:
            print(f"  {filename}: {error}")
        if len(corrupted) > 20:
            print(f"  ... and {len(corrupted) - 20} more")
        print()
        
        md_output.append(f"## Corrupted/Unreadable Images ({len(corrupted)})\n")
        md_output.append("```")
        for filename, error in corrupted[:50]:
            md_output.append(f"{filename}: {error}")
        if len(corrupted) > 50:
            md_output.append(f"... and {len(corrupted) - 50} more")
        md_output.append("```\n")
    
    # Recommendations
    print("="*70)
    print("PREPROCESSING RECOMMENDATIONS")
    print("="*70)
    
    recommendations = []
    
    # Format standardization
    if len(formats) > 1:
        most_common_format = formats.most_common(1)[0][0]
        pct = formats[most_common_format] / sum(formats.values()) * 100
        recommendations.append(
            f"**Format Standardization**: {most_common_format} is most common ({pct:.1f}%). "
            f"Consider converting all images to {most_common_format} for consistency."
        )
    else:
        recommendations.append(f"**Format**: All images are already in {list(formats.keys())[0]} format. ✓")
    
    # Color mode consistency
    if len(modes) > 1:
        most_common_mode = modes.most_common(1)[0][0]
        pct = modes[most_common_mode] / sum(modes.values()) * 100
        recommendations.append(
            f"**Color Mode**: {most_common_mode} is most common ({pct:.1f}%). "
            f"Medical images benefit from consistent color representation. "
            f"Consider converting all to {most_common_mode}."
        )
    else:
        recommendations.append(f"**Color Mode**: All images use {list(modes.keys())[0]} mode. ✓")
    
    # Dimension recommendations
    if widths and heights:
        median_width = sorted(widths)[len(widths)//2]
        median_height = sorted(heights)[len(heights)//2]
        recommendations.append(
            f"**Resize**: Median dimensions are {median_width}x{median_height}. "
            f"For neural networks, consider resizing to standard sizes like: "
            f"224x224 (ResNet/VGG), 256x256, 384x384, or 512x512."
        )
        
        if width_std > width_mean * 0.5 or height_std > height_mean * 0.5:
            recommendations.append(
                f"**High Variance**: Large variation in dimensions detected. "
                f"Consistent sizing is critical for batch processing."
            )
    
    # Resolution recommendations
    if megapixels:
        if max(megapixels) > 10:
            recommendations.append(
                f"**High Resolution**: Some images exceed 10 MP. Consider downsampling to reduce "
                f"computational cost while preserving medical features."
            )
        if min(megapixels) < 0.1:
            recommendations.append(
                f"**Low Resolution**: Some images are very small (<0.1 MP). "
                f"May need upsampling or removal if too low quality."
            )
    
    # Intensity normalization
    recommendations.append(
        f"**Intensity Normalization**: Apply pixel value normalization (0-1 range or z-score) "
        f"to standardize input ranges across all images."
    )
    
    # Medical-specific recommendations
    recommendations.append(
        f"**Medical Enhancement**: Consider CLAHE (Contrast Limited Adaptive Histogram Equalization) "
        f"for X-rays and CT scans to enhance anatomical details."
    )
    
    # Data augmentation
    recommendations.append(
        f"**Data Augmentation**: Apply random rotations (±10°), horizontal flips, "
        f"brightness/contrast adjustments, and elastic deformations during training."
    )
    
    # Quality checks
    if corrupted:
        recommendations.append(
            f"**Corrupted Files**: {len(corrupted):,} corrupted/unreadable images detected. "
            f"Remove these before training."
        )
    
    if very_small:
        recommendations.append(
            f"**Very Small Images**: {len(very_small):,} images < 256px detected. "
            f"Consider removing or treating separately."
        )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
        print()
    
    md_output.append("## Preprocessing Recommendations\n")
    for i, rec in enumerate(recommendations, 1):
        md_output.append(f"{i}. {rec}\n")
    md_output.append("\n---\n")
    
    md_output.append("## Summary\n")
    md_output.append(f"This analysis covered **{len(image_files):,} images** from the medical dataset. ")
    md_output.append(f"The images show ")
    if len(formats) == 1:
        md_output.append(f"consistent format ({list(formats.keys())[0]}) ")
    else:
        md_output.append(f"mixed formats ")
    if len(modes) == 1:
        md_output.append(f"and consistent color mode ({list(modes.keys())[0]}). ")
    else:
        md_output.append(f"but varied color modes. ")
    
    if widths and heights:
        md_output.append(f"Dimensions range from {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)} pixels, ")
        md_output.append(f"with median {width_median}x{height_median}. ")
    md_output.append(f"Total dataset size is {format_bytes(total_size)}. ")
    
    if corrupted:
        md_output.append(f"**Note:** {len(corrupted):,} corrupted images need attention. ")
    
    md_output.append("Apply the recommendations above to prepare the dataset for machine learning training.")
    
    return "\n".join(md_output)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_images_comprehensive.py <image_folder>")
        print("Example: python3 analyze_images_comprehensive.py /path/to/images")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found: {image_folder}")
        sys.exit(1)
    
    print(f"Starting comprehensive image analysis...")
    print(f"Folder: {image_folder}")
    print()
    
    # Analyze images
    md_report = analyze_images(image_folder)
    
    if md_report is None:
        print("No images found to analyze!")
        sys.exit(1)
    
    # Generate output filename from folder name
    folder_name = os.path.basename(os.path.normpath(image_folder))
    output_file = f"{folder_name}_image_analysis.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print("="*70)
    print(f"✓ Comprehensive analysis complete!")
    print(f"✓ Report saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
