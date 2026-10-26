#!/usr/bin/env python3
"""
Clean and filter captions from s2_final.json.
Apply multiple cleaning rules to improve data quality.
Configure which rules to apply below.
"""

import json
import re

# Configuration - edit these
input_file = "s2_final.json"
output_file = "s2_cleaned.json"

# Cleaning rules - set to True to enable
# Based on analysis of s2_final.json (138,202 captions)
RULES = {
    # CRITICAL FIXES (resolve major data quality issues)
    "remove_figure_prefix": True,        # Remove "Figure X." or "Fig. X:" from start (affects 56.9% - 78,620 captions)
    "remove_incomplete": True,           # Remove incomplete sentences ending with "..." or no period (22.7% - 31,383 captions)
    "remove_very_short": True,           # Remove captions with < 5 words (0.7% - 1,010 captions)
    "remove_urls": True,                 # Remove captions containing URLs (1.0% - 1,330 captions)
    "remove_citations": True,            # Remove citation markers like [1], (ref 2), etc. (0.8% - 1,072 captions)
    "fix_repeated_words": True,          # Fix repeated words like "the the" → "the" (0.5% - 716 captions)
    
    # QUALITY IMPROVEMENTS (enhance but don't remove)
    "normalize_medical_terms": True,     # Standardize medical terminology (x-ray/xray → X-ray, mri → MRI, ct → CT)
    "clean_whitespace": True,            # Fix extra spaces, tabs, newlines
    "clean_punctuation": True,           # Fix punctuation issues (multiple periods, extra commas, etc.)
    "fix_capitalization": True,          # Fix ALL CAPS sentences (keep medical abbreviations like MRI, CT)
    "remove_special_chars": True,        # Clean excessive special characters (keep medical symbols)
    "normalize_brackets": True,          # Normalize bracket styles and remove empty brackets
    
    # OPTIONAL FILTERS (disabled by default - may remove valuable data)
    "remove_very_long": False,           # Remove captions with > 200 words (disabled - long captions often detailed)
    "remove_no_medical_terms": False,    # Remove captions without medical terms (disabled - may remove valid captions)
    "remove_generic": False,             # Remove generic/boilerplate captions (disabled - may be too aggressive)
    "remove_duplicates": True,           # Keep only unique captions
    "remove_non_english": False,         # Remove captions with non-English characters (disabled - may remove valid medical terms)
    "filter_by_modality": False,         # Keep only specific modalities (configure below)
}

# Modality filter (only used if filter_by_modality is True)
ALLOWED_MODALITIES = ["x-ray", "mri", "ct", "ultrasound", "microscopy"]

# Medical terms (used if remove_no_medical_terms is True)
# Based on actual data - top terms that appear in 88.4% of captions
MEDICAL_KEYWORDS = [
    # Top modalities (from your data)
    "ct", "mri", "x-ray", "xray", "radiograph", "computed tomography", 
    "magnetic resonance", "ct scan", "ultrasound", "sonography",
    "microscopy", "histology", "pathology", "pet", "scan",
    # Top anatomy terms (from your data)
    "cell", "head", "brain", "tissue", "chest", "abdomen", "heart", "lung",
    "liver", "kidney", "spine", "bone", "muscle", "blood", "vessel", "organ",
    "pelvis", "neck", "artery", "vein",
    # Top findings (from your data)
    "lesion", "mass", "tumor", "normal", "abnormal", "cancer", "fracture",
    "inflammation", "infection", "edema", "hemorrhage", "nodule", "disease",
    # Common medical terms
    "patient", "diagnosis", "clinical", "treatment", "medical", "surgical",
    "contrast", "enhancement", "imaging"
]

# Generic phrases to filter out
GENERIC_PHRASES = [
    "figure shows",
    "figure depicts",
    "image shows",
    "image depicts",
    "this figure",
    "this image",
    "the figure",
    "the image",
    "as shown",
    "see figure",
    "see image"
]

# Medical term normalization mapping
# Maps various spellings/formats to a single standardized form
MEDICAL_TERM_NORMALIZATION = {
    # Imaging modalities
    r'\bxray\b': 'X-ray',
    r'\bx ray\b': 'X-ray',
    r'\bx-rays?\b': 'X-ray',
    r'\bct\b': 'CT',
    r'\bct scan\b': 'CT scan',
    r'\bcat scan\b': 'CT scan',
    r'\bmri\b': 'MRI',
    r'\bmr\b': 'MR',
    r'\bpet\b': 'PET',
    r'\bpet scan\b': 'PET scan',
    r'\bpet-ct\b': 'PET-CT',
    r'\bpet/ct\b': 'PET-CT',
    
    # Full names to abbreviations
    r'\bcomputed tomography\b': 'CT',
    r'\bmagnetic resonance imaging\b': 'MRI',
    r'\bmagnetic resonance\b': 'MR',
    r'\bpositron emission tomography\b': 'PET',
    
    # Ultrasound variations
    r'\bultra sound\b': 'ultrasound',
    r'\bultra-sound\b': 'ultrasound',
    r'\bu/s\b': 'ultrasound',
    r'\bus\b': 'ultrasound',
    
    # Common medical terms
    r'\bedema\b': 'edema',
    r'\boedema\b': 'edema',
    r'\btumor\b': 'tumor',
    r'\btumour\b': 'tumor',
    r'\bhemorrhage\b': 'hemorrhage',
    r'\bhaemorrhage\b': 'hemorrhage',
    
    # Directional terms
    r'\bleft hand side\b': 'left-hand side',
    r'\bright hand side\b': 'right-hand side',
    r'\bleft-sided\b': 'left sided',
    r'\bright-sided\b': 'right sided',
    
    # Measurements
    r'\bmillimeter\b': 'mm',
    r'\bmillimeters\b': 'mm',
    r'\bcentimeter\b': 'cm',
    r'\bcentimeters\b': 'cm',
}

print("="*60)
print("CAPTION CLEANING - CRITICAL ISSUES RESOLUTION")
print("="*60)
print(f"Input: {input_file}")
print(f"Output: {output_file}")
print()

print("Active cleaning rules:")
print("\nCRITICAL FIXES:")
critical_rules = ["remove_figure_prefix", "remove_incomplete", "remove_very_short", 
                  "remove_urls", "remove_citations", "fix_repeated_words"]
for rule in critical_rules:
    if RULES.get(rule, False):
        print(f"  ✓ {rule}")

print("\nQUALITY IMPROVEMENTS:")
quality_rules = ["normalize_medical_terms", "clean_whitespace", "clean_punctuation", 
                 "fix_capitalization", "remove_special_chars", "normalize_brackets"]
for rule in quality_rules:
    if RULES.get(rule, False):
        print(f"  ✓ {rule}")

print("\nOPTIONAL FILTERS:")
optional_rules = ["remove_very_long", "remove_no_medical_terms", "remove_generic",
                  "remove_duplicates", "remove_non_english", "filter_by_modality"]
for rule in optional_rules:
    if RULES.get(rule, False):
        print(f"  ✓ {rule}")
print()

# Load data
print("Loading data...")
entries = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except:
            continue

print(f"Loaded {len(entries):,} entries\n")

# Track statistics
stats = {
    "total": len(entries),
    "removed_short": 0,
    "removed_long": 0,
    "removed_no_medical": 0,
    "removed_generic": 0,
    "removed_duplicates": 0,
    "removed_modality": 0,
    "removed_urls": 0,
    "removed_incomplete": 0,
    "removed_non_english": 0,
    "cleaned_prefix": 0,
    "cleaned_whitespace": 0,
    "cleaned_punctuation": 0,
    "cleaned_repeated_words": 0,
    "cleaned_medical_terms": 0,
    "cleaned_special_chars": 0,
    "cleaned_brackets": 0,
    "cleaned_citations": 0,
    "cleaned_capitalization": 0,
}

cleaned_entries = []
seen_captions = set()

for entry in entries:
    caption = entry.get("s2orc_caption", "")
    
    if not caption:
        continue
    
    original_caption = caption
    keep = True
    
    # Rule: Clean whitespace
    if RULES["clean_whitespace"]:
        # Replace multiple spaces/tabs/newlines with single space
        caption = re.sub(r'\s+', ' ', caption)
        caption = caption.strip()
        if caption != original_caption:
            stats["cleaned_whitespace"] += 1
    
    # Rule: Remove URLs
    if RULES["remove_urls"]:
        caption_lower = caption.lower()
        if "http" in caption_lower or "www." in caption_lower or ".com" in caption_lower:
            stats["removed_urls"] += 1
            keep = False
            continue
    
    # Rule: Remove non-English characters (keep medical symbols)
    if RULES["remove_non_english"]:
        # Allow: letters, numbers, basic punctuation, medical symbols (±, ≤, ≥, ×, °, μ, %, etc.)
        # Check if caption has excessive non-ASCII or non-medical special chars
        ascii_ratio = sum(c.isascii() for c in caption) / len(caption) if caption else 1
        if ascii_ratio < 0.9:  # More than 10% non-ASCII
            stats["removed_non_english"] += 1
            keep = False
            continue
    
    # Rule: Clean punctuation
    if RULES["clean_punctuation"]:
        original = caption
        # Fix multiple periods (e.g., "..." -> ".")
        caption = re.sub(r'\.{2,}', '.', caption)
        # Fix multiple commas (e.g., ",," -> ",")
        caption = re.sub(r',{2,}', ',', caption)
        # Fix multiple question marks or exclamation marks
        caption = re.sub(r'\?{2,}', '?', caption)
        caption = re.sub(r'!{2,}', '!', caption)
        # Fix space before punctuation (e.g., "word ." -> "word.")
        caption = re.sub(r'\s+([.,;:!?])', r'\1', caption)
        # Fix no space after punctuation (e.g., "word.Another" -> "word. Another")
        caption = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', caption)
        # Fix multiple spaces after punctuation (e.g., "word.  Another" -> "word. Another")
        caption = re.sub(r'([.,;:!?])\s{2,}', r'\1 ', caption)
        # Remove trailing punctuation except period
        caption = re.sub(r'[,;:]+$', '', caption)
        # Ensure single space after punctuation
        caption = caption.strip()
        if caption != original:
            stats["cleaned_punctuation"] += 1
    
    # Rule: Fix repeated words
    if RULES["fix_repeated_words"]:
        original = caption
        # Fix repeated words: "the the" -> "the", "and and" -> "and"
        # Use word boundaries to avoid partial matches
        # Match: word + space + same word (case-insensitive)
        caption = re.sub(r'\b(\w+)\s+\1\b', r'\1', caption, flags=re.IGNORECASE)
        # Clean up any resulting double spaces
        caption = re.sub(r'\s+', ' ', caption)
        caption = caption.strip()
        if caption != original:
            stats["cleaned_repeated_words"] += 1
    
    # Rule: Normalize medical terms
    if RULES["normalize_medical_terms"]:
        original = caption
        # Apply all normalization patterns
        for pattern, replacement in MEDICAL_TERM_NORMALIZATION.items():
            caption = re.sub(pattern, replacement, caption, flags=re.IGNORECASE)
        # Clean up any resulting double spaces
        caption = re.sub(r'\s+', ' ', caption)
        caption = caption.strip()
        if caption != original:
            stats["cleaned_medical_terms"] += 1
    
    # Rule: Remove special characters
    if RULES["remove_special_chars"]:
        original = caption
        # Remove excessive special chars but keep medical symbols
        # Keep: letters, numbers, spaces, basic punctuation, ±≤≥×°μ%
        caption = re.sub(r'[^\w\s.,;:!?()\[\]{}\-±≤≥×°μ%/\'\"]+', ' ', caption)
        # Clean up multiple spaces created
        caption = re.sub(r'\s+', ' ', caption)
        caption = caption.strip()
        if caption != original:
            stats["cleaned_special_chars"] += 1
    
    # Rule: Normalize brackets
    if RULES["normalize_brackets"]:
        original = caption
        # Remove empty brackets
        caption = re.sub(r'\(\s*\)', '', caption)
        caption = re.sub(r'\[\s*\]', '', caption)
        caption = re.sub(r'\{\s*\}', '', caption)
        # Ensure space before opening bracket
        caption = re.sub(r'(\w)(\()', r'\1 \2', caption)
        # Ensure space after closing bracket
        caption = re.sub(r'(\))(\w)', r'\1 \2', caption)
        caption = caption.strip()
        if caption != original:
            stats["cleaned_brackets"] += 1
    
    # Rule: Remove citations
    if RULES["remove_citations"]:
        original = caption
        # Remove citation patterns: [1], [2,3], (ref 1), (Smith et al.), etc.
        caption = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', caption)
        caption = re.sub(r'\((?:ref|refs|reference|references)\.?\s*\d+(?:,\s*\d+)*\)', '', caption, flags=re.IGNORECASE)
        caption = re.sub(r'\([A-Z][a-z]+\s+et\s+al\.?,?\s*\d{4}?\)', '', caption)
        # Clean up spaces
        caption = re.sub(r'\s+', ' ', caption)
        caption = caption.strip()
        if caption != original:
            stats["cleaned_citations"] += 1
    
    # Rule: Fix capitalization
    if RULES["fix_capitalization"]:
        original = caption
        # Check if entire caption is ALL CAPS
        if caption.isupper() and len(caption) > 10:
            # Convert to sentence case
            caption = caption.capitalize()
            stats["cleaned_capitalization"] += 1
        # Fix sentences that don't start with capital letter
        elif caption and caption[0].islower():
            caption = caption[0].upper() + caption[1:]
            if caption != original:
                stats["cleaned_capitalization"] += 1
    
    # Rule: Remove figure prefix
    if RULES["remove_figure_prefix"]:
        original = caption
        # Remove patterns like "Figure 1.", "Fig. 2:", "Figure 1A-C.", etc.
        caption = re.sub(r'^(Figure|Fig\.?)\s*\d+[A-Za-z]*[\.\-–—:]?\s*', '', caption, flags=re.IGNORECASE)
        caption = caption.strip()
        if caption != original:
            stats["cleaned_prefix"] += 1
    
    # Rule: Remove incomplete sentences
    if RULES["remove_incomplete"]:
        # Remove captions ending with "..." or captions without proper ending
        if caption.endswith('...') or caption.endswith('..'):
            stats["removed_incomplete"] += 1
            keep = False
            continue
        # Check if caption is a complete sentence (has some punctuation at end)
        if len(caption) > 20 and not re.search(r'[.!?]$', caption):
            # Skip check for very technical captions (may be labels)
            word_count = len(caption.split())
            if word_count > 15:  # Only check longer captions
                stats["removed_incomplete"] += 1
                keep = False
                continue
    
    # Rule: Remove very short captions
    if RULES["remove_very_short"]:
        word_count = len(caption.split())
        if word_count < 5:
            stats["removed_short"] += 1
            keep = False
            continue
    
    # Rule: Remove very long captions
    if RULES["remove_very_long"]:
        word_count = len(caption.split())
        if word_count > 200:  # Increased from 150 - keep detailed captions
            stats["removed_long"] += 1
            keep = False
            continue
    
    # Rule: Remove captions without medical terms
    if RULES["remove_no_medical_terms"]:
        caption_lower = caption.lower()
        has_medical = any(term in caption_lower for term in MEDICAL_KEYWORDS)
        if not has_medical:
            stats["removed_no_medical"] += 1
            keep = False
            continue
    
    # Rule: Remove generic captions
    if RULES["remove_generic"]:
        caption_lower = caption.lower()
        # Check if caption is very short and generic
        word_count = len(caption.split())
        if word_count < 15:  # Only check short captions
            is_generic = any(phrase in caption_lower for phrase in GENERIC_PHRASES)
            if is_generic:
                stats["removed_generic"] += 1
                keep = False
                continue
    
    # Rule: Filter by modality
    if RULES["filter_by_modality"]:
        caption_lower = caption.lower()
        has_modality = any(mod in caption_lower for mod in ALLOWED_MODALITIES)
        if not has_modality:
            stats["removed_modality"] += 1
            keep = False
            continue
    
    # Rule: Remove duplicates
    if RULES["remove_duplicates"]:
        if caption in seen_captions:
            stats["removed_duplicates"] += 1
            keep = False
            continue
        seen_captions.add(caption)
    
    # Keep the entry with cleaned caption
    if keep:
        entry["s2orc_caption"] = caption
        cleaned_entries.append(entry)

# Write cleaned data
print("Writing cleaned data...")
with open(output_file, "w", encoding="utf-8") as f:
    for entry in cleaned_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Print statistics
print()
print("="*60)
print("CLEANING SUMMARY - CRITICAL ISSUES RESOLVED")
print("="*60)
print(f"Original entries: {stats['total']:,}")
print()

# Calculate critical issues resolved
critical_fixed = (stats["cleaned_prefix"] + stats["removed_incomplete"] + 
                  stats["removed_short"] + stats["removed_urls"] + 
                  stats["cleaned_citations"] + stats["cleaned_repeated_words"])
print(f"CRITICAL ISSUES RESOLVED: {critical_fixed:,}")
print()

if stats["cleaned_prefix"] > 0:
    print(f"Cleaned figure prefixes: {stats['cleaned_prefix']:,}")
if stats["cleaned_whitespace"] > 0:
    print(f"Cleaned whitespace: {stats['cleaned_whitespace']:,}")
if stats["cleaned_punctuation"] > 0:
    print(f"Cleaned punctuation: {stats['cleaned_punctuation']:,}")
if stats["cleaned_repeated_words"] > 0:
    print(f"Fixed repeated words: {stats['cleaned_repeated_words']:,}")
if stats["cleaned_medical_terms"] > 0:
    print(f"Normalized medical terms: {stats['cleaned_medical_terms']:,}")
if stats["cleaned_special_chars"] > 0:
    print(f"Cleaned special characters: {stats['cleaned_special_chars']:,}")
if stats["cleaned_brackets"] > 0:
    print(f"Normalized brackets: {stats['cleaned_brackets']:,}")
if stats["cleaned_citations"] > 0:
    print(f"Removed citations: {stats['cleaned_citations']:,}")
if stats["cleaned_capitalization"] > 0:
    print(f"Fixed capitalization: {stats['cleaned_capitalization']:,}")
print()

print("Removed entries:")
if stats["removed_short"] > 0:
    print(f"  Too short (< 5 words): {stats['removed_short']:,}")
if stats["removed_long"] > 0:
    print(f"  Too long (> 200 words): {stats['removed_long']:,}")
if stats["removed_no_medical"] > 0:
    print(f"  No medical terms: {stats['removed_no_medical']:,}")
if stats["removed_generic"] > 0:
    print(f"  Generic phrases: {stats['removed_generic']:,}")
if stats["removed_duplicates"] > 0:
    print(f"  Duplicates: {stats['removed_duplicates']:,}")
if stats["removed_modality"] > 0:
    print(f"  Wrong modality: {stats['removed_modality']:,}")
if stats["removed_urls"] > 0:
    print(f"  Contains URLs: {stats['removed_urls']:,}")
if stats["removed_incomplete"] > 0:
    print(f"  Incomplete sentences: {stats['removed_incomplete']:,}")
if stats["removed_non_english"] > 0:
    print(f"  Non-English content: {stats['removed_non_english']:,}")

total_removed = (stats["removed_short"] + stats["removed_long"] + 
                 stats["removed_no_medical"] + stats["removed_generic"] + 
                 stats["removed_duplicates"] + stats["removed_modality"] +
                 stats["removed_urls"] + stats["removed_incomplete"] + 
                 stats["removed_non_english"])

print()
print(f"Total removed: {total_removed:,}")
print(f"Final entries: {len(cleaned_entries):,}")
print(f"Kept: {len(cleaned_entries) / stats['total'] * 100:.1f}%")
print()
print(f"Output saved to: {output_file}")
