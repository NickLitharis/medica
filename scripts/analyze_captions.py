#!/usr/bin/env python3
"""
Simple caption analyzer for S2ORC medical image captions.
Analyzes s2_final.json to provide statistics and insights.
No external dependencies required.

Usage:
    python3 analyze_captions.py [input_file]
    
    input_file: Path to the JSON/JSONL file to analyze (default: s2_cleaned.json)
    
    Output file is automatically generated as: <input_file>_analysis_report.md
    
Examples:
    python3 analyze_captions.py
    # Analyzes s2_cleaned.json → generates s2_cleaned_analysis_report.md
    
    python3 analyze_captions.py s2_final.json
    # Analyzes s2_final.json → generates s2_final_analysis_report.md
"""

import json
import re
from collections import Counter
import math
from datetime import datetime
import sys

# Parse command line arguments
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = "s2_cleaned.json"

# Auto-generate output filename based on input
base_name = input_file.rsplit('.', 1)[0]
output_file = f"{base_name}_analysis_report.md"

# Medical terms to search for
MEDICAL_TERMS = {
    "modalities": ["x-ray", "xray", "radiograph", "mri", "magnetic resonance", "ct scan", "ct", 
                   "computed tomography", "ultrasound", "sonography", "pet scan", "microscopy",
                   "histology", "pathology", "endoscopy"],
    "anatomy": ["brain", "heart", "lung", "liver", "kidney", "spine", "chest", "abdomen",
                "pelvis", "head", "neck", "bone", "muscle", "blood", "vessel", "artery",
                "vein", "tissue", "organ", "cell"],
    "findings": ["lesion", "tumor", "cancer", "fracture", "inflammation", "infection",
                 "edema", "hemorrhage", "mass", "nodule", "abnormal", "normal", "pathology"]
}

FIGURE_TYPES = {
    "x-ray": ["x-ray", "xray", "radiograph", "radiographic"],
    "mri": ["mri", "magnetic resonance"],
    "ct": ["ct", "computed tomography", "cat scan"],
    "ultrasound": ["ultrasound", "sonography", "sonogram"],
    "microscopy": ["microscopy", "microscopic", "histology", "histologic", "pathology"],
    "chart": ["chart", "graph", "plot", "diagram", "table"]
}

# Additional analysis categories
PUNCTUATION_PATTERNS = {
    "ends_with_period": r'\.$',
    "ends_with_question": r'\?$',
    "ends_with_exclamation": r'!$',
    "contains_parentheses": r'\([^)]+\)',
    "contains_brackets": r'\[[^\]]+\]',
    "contains_quotes": r'["\']',
    "contains_colon": r':',
    "contains_semicolon": r';',
}

QUALITY_INDICATORS = {
    "good": ["shows", "demonstrates", "reveals", "indicates", "depicts", "presents", 
             "illustrates", "displays", "represents"],
    "descriptive": ["patient", "case", "study", "examination", "scan", "image"],
    "technical": ["axial", "sagittal", "coronal", "anterior", "posterior", "lateral",
                  "transverse", "longitudinal"],
    "quantitative": ["mm", "cm", "size", "volume", "diameter", "length", "width", 
                    "measurement", "quantification"],
}

PROBLEMATIC_PATTERNS = {
    "citation_markers": r'\[\d+\]|\(ref\s*\d+\)',
    "urls": r'http[s]?://|www\.',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "excessive_caps": r'\b[A-Z]{5,}\b',
    "repeated_words": r'\b(\w+)\s+\1\b',
    "placeholder": r'\[INSERT\]|\[TODO\]|\[XXX\]|PLACEHOLDER',
}

def create_bar_chart(data, max_width=50, show_count=True):
    """Create a horizontal bar chart in ASCII"""
    if not data:
        return ""
    
    max_value = max(v for _, v in data)
    lines = []
    
    for label, value in data:
        bar_length = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_length
        if show_count:
            lines.append(f"{label:25s} │{bar} {value:,}")
        else:
            pct = (value / sum(v for _, v in data) * 100) if sum(v for _, v in data) > 0 else 0
            lines.append(f"{label:25s} │{bar} {pct:.1f}%")
    
    return "\n".join(lines)

def create_histogram(values, bins=10, max_width=50):
    """Create a histogram in ASCII"""
    if not values:
        return ""
    
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / bins
    
    # Create bins
    bin_counts = [0] * bins
    for v in values:
        bin_idx = min(int((v - min_val) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1
    
    max_count = max(bin_counts)
    lines = []
    
    for i, count in enumerate(bin_counts):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar_length = int((count / max_count) * max_width) if max_count > 0 else 0
        bar = "█" * bar_length
        lines.append(f"{bin_start:6.0f}-{bin_end:<6.0f} │{bar} {count:,}")
    
    return "\n".join(lines)

print("="*60)
print("CAPTION ANALYZER")
print("="*60)
print(f"Input file: {input_file}")
print(f"Output file: {output_file}")

# Check if input file exists
import os
if not os.path.exists(input_file):
    print(f"\n❌ Error: Input file '{input_file}' not found!")
    print("\nUsage:")
    print(f"    python3 {sys.argv[0]} [input_file]")
    print("\nExamples:")
    print(f"    python3 {sys.argv[0]}")
    print(f"    python3 {sys.argv[0]} s2_final.json")
    print(f"\nOutput file will be auto-generated as: <input_file>_analysis_report.md")
    sys.exit(1)

print()

# Start building markdown output
md_output = []
md_output.append(f"# Caption Analysis Report")
md_output.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
md_output.append(f"\n**Source File:** `{input_file}`")
md_output.append("\n---\n")

# Load data
print("Loading data...")
captions = []
references = []
pdf_hashes = []
fig_uris = []
all_entries = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            caption = data.get("s2orc_caption", "")
            ref = data.get("s2orc_references", [])
            pdf_hash = data.get("pdf_hash", "")
            fig_uri = data.get("fig_uri", "")
            
            all_entries.append(data)
            if caption:
                captions.append(caption)
            if ref:
                references.append(ref)
            if pdf_hash:
                pdf_hashes.append(pdf_hash)
            if fig_uri:
                fig_uris.append(fig_uri)
        except:
            continue

total_entries = len(all_entries)
total_captions = len(captions)
total_with_refs = len(references)
total_fig_uris = len(fig_uris)

print(f"Loaded {total_entries:,} entries")
print(f"  With captions: {total_captions:,} ({total_captions/total_entries*100:.1f}%)")
print(f"  With references: {total_with_refs:,} ({total_with_refs/total_entries*100:.1f}%)")
print(f"  With fig_uri: {total_fig_uris:,} ({total_fig_uris/total_entries*100:.1f}%)")
print()

md_output.append("## Dataset Overview\n")
md_output.append(f"- **Total entries:** {total_entries:,}")
md_output.append(f"- **With captions:** {total_captions:,} ({total_captions/total_entries*100:.1f}%)")
md_output.append(f"- **With references:** {total_with_refs:,} ({total_with_refs/total_entries*100:.1f}%)")
md_output.append(f"- **With fig_uri:** {total_fig_uris:,} ({total_fig_uris/total_entries*100:.1f}%)")
md_output.append("\n---\n")

# Basic statistics
print("="*60)
print("BASIC STATISTICS")
print("="*60)

md_output.append("## Basic Statistics\n")

if captions:
    caption_lengths = [len(c) for c in captions]
    word_counts = [len(c.split()) for c in captions]
    sentence_counts = [len(re.split(r'[.!?]+', c)) - 1 for c in captions]
    
    # Calculate standard deviation
    avg_length = sum(caption_lengths) / len(caption_lengths)
    std_length = math.sqrt(sum((x - avg_length) ** 2 for x in caption_lengths) / len(caption_lengths))
    
    avg_words = sum(word_counts) / len(word_counts)
    std_words = math.sqrt(sum((x - avg_words) ** 2 for x in word_counts) / len(word_counts))
    
    print(f"Caption length (characters):")
    print(f"  Min: {min(caption_lengths):,}")
    print(f"  Max: {max(caption_lengths):,}")
    print(f"  Average: {avg_length:.1f} (±{std_length:.1f})")
    print(f"  Median: {sorted(caption_lengths)[len(caption_lengths)//2]:,}")
    print(f"  25th percentile: {sorted(caption_lengths)[len(caption_lengths)//4]:,}")
    print(f"  75th percentile: {sorted(caption_lengths)[3*len(caption_lengths)//4]:,}")
    print()
    
    md_output.append("### Caption Length (characters)\n")
    md_output.append(f"- **Min:** {min(caption_lengths):,}")
    md_output.append(f"- **Max:** {max(caption_lengths):,}")
    md_output.append(f"- **Average:** {avg_length:.1f} (±{std_length:.1f})")
    md_output.append(f"- **Median:** {sorted(caption_lengths)[len(caption_lengths)//2]:,}")
    md_output.append(f"- **25th percentile:** {sorted(caption_lengths)[len(caption_lengths)//4]:,}")
    md_output.append(f"- **75th percentile:** {sorted(caption_lengths)[3*len(caption_lengths)//4]:,}")
    md_output.append("\n#### Distribution\n")
    md_output.append("```")
    md_output.append(create_histogram(caption_lengths, bins=10))
    md_output.append("```\n")
    
    print(f"Word count:")
    print(f"  Min: {min(word_counts)}")
    print(f"  Max: {max(word_counts)}")
    print(f"  Average: {avg_words:.1f} (±{std_words:.1f})")
    print(f"  Median: {sorted(word_counts)[len(word_counts)//2]}")
    print(f"  25th percentile: {sorted(word_counts)[len(word_counts)//4]}")
    print(f"  75th percentile: {sorted(word_counts)[3*len(word_counts)//4]}")
    print()
    
    md_output.append("### Word Count\n")
    md_output.append(f"- **Min:** {min(word_counts)}")
    md_output.append(f"- **Max:** {max(word_counts)}")
    md_output.append(f"- **Average:** {avg_words:.1f} (±{std_words:.1f})")
    md_output.append(f"- **Median:** {sorted(word_counts)[len(word_counts)//2]}")
    md_output.append(f"- **25th percentile:** {sorted(word_counts)[len(word_counts)//4]}")
    md_output.append(f"- **75th percentile:** {sorted(word_counts)[3*len(word_counts)//4]}")
    md_output.append("\n#### Distribution\n")
    md_output.append("```")
    md_output.append(create_histogram(word_counts, bins=10))
    md_output.append("```\n")
    
    print(f"Sentence count per caption:")
    print(f"  Average: {sum(sentence_counts) / len(sentence_counts):.1f}")
    print(f"  Single sentence: {sum(1 for s in sentence_counts if s == 1):,} ({sum(1 for s in sentence_counts if s == 1)/len(captions)*100:.1f}%)")
    print(f"  Multiple sentences: {sum(1 for s in sentence_counts if s > 1):,} ({sum(1 for s in sentence_counts if s > 1)/len(captions)*100:.1f}%)")
    print()
    
    md_output.append("### Sentence Count\n")
    md_output.append(f"- **Average:** {sum(sentence_counts) / len(sentence_counts):.1f}")
    md_output.append(f"- **Single sentence:** {sum(1 for s in sentence_counts if s == 1):,} ({sum(1 for s in sentence_counts if s == 1)/len(captions)*100:.1f}%)")
    md_output.append(f"- **Multiple sentences:** {sum(1 for s in sentence_counts if s > 1):,} ({sum(1 for s in sentence_counts if s > 1)/len(captions)*100:.1f}%)\n")
    
    # Word length distribution
    all_words = []
    for caption in captions:
        all_words.extend(caption.split())
    
    if all_words:
        word_lengths = [len(w) for w in all_words]
        avg_word_len = sum(word_lengths) / len(word_lengths)
        print(f"Word characteristics:")
        print(f"  Total unique words: {len(set(all_words)):,}")
        print(f"  Average word length: {avg_word_len:.1f} characters")
        print(f"  Vocabulary richness: {len(set(all_words)) / len(all_words):.3f}")
        
        md_output.append("### Word Characteristics\n")
        md_output.append(f"- **Total unique words:** {len(set(all_words)):,}")
        md_output.append(f"- **Average word length:** {avg_word_len:.1f} characters")
        md_output.append(f"- **Vocabulary richness:** {len(set(all_words)) / len(all_words):.3f}\n")
    print()

# Medical term analysis
print("="*60)
print("MEDICAL TERM ANALYSIS")
print("="*60)

md_output.append("---\n\n## Medical Term Analysis\n")

for category, terms in MEDICAL_TERMS.items():
    print(f"\n{category.upper()}:")
    term_counts = Counter()
    captions_with_term = 0
    
    for caption in captions:
        caption_lower = caption.lower()
        has_term = False
        for term in terms:
            if term in caption_lower:
                term_counts[term] += 1
                has_term = True
        if has_term:
            captions_with_term += 1
    
    coverage = (captions_with_term / len(captions) * 100) if captions else 0
    print(f"  Coverage: {captions_with_term:,} captions ({coverage:.1f}%)")
    
    md_output.append(f"### {category.upper()}\n")
    md_output.append(f"**Coverage:** {captions_with_term:,} captions ({coverage:.1f}%)\n")
    
    if term_counts:
        print(f"  Top terms:")
        md_output.append("**Top terms:**\n")
        md_output.append("```")
        top_terms = [(term, count) for term, count in term_counts.most_common(10)]
        md_output.append(create_bar_chart(top_terms, max_width=40))
        md_output.append("```\n")
        for term, count in term_counts.most_common(5):
            print(f"    - {term}: {count:,}")

# Figure type classification
print()
print("="*60)
print("FIGURE TYPE CLASSIFICATION")
print("="*60)

md_output.append("---\n\n## Figure Type Classification\n")

type_counts = Counter()
unclassified = 0

for caption in captions:
    caption_lower = caption.lower()
    found_type = False
    
    for fig_type, keywords in FIGURE_TYPES.items():
        if any(kw in caption_lower for kw in keywords):
            type_counts[fig_type] += 1
            found_type = True
            break
    
    if not found_type:
        unclassified += 1

print(f"Total captions: {len(captions):,}")
md_output.append(f"**Total captions:** {len(captions):,}\n")
md_output.append("```")

chart_data = [(fig_type, count) for fig_type, count in type_counts.most_common()]
chart_data.append(("unclassified", unclassified))
md_output.append(create_bar_chart(chart_data, max_width=50, show_count=False))
md_output.append("```\n")

for fig_type, count in type_counts.most_common():
    pct = (count / len(captions) * 100) if captions else 0
    print(f"  {fig_type}: {count:,} ({pct:.1f}%)")
print(f"  unclassified: {unclassified:,} ({unclassified/len(captions)*100:.1f}%)")

# Punctuation analysis
print()
print("="*60)
print("PUNCTUATION ANALYSIS")
print("="*60)

md_output.append("---\n\n## Punctuation Analysis\n")
md_output.append("```")

punct_data = []
for pattern_name, pattern in PUNCTUATION_PATTERNS.items():
    count = sum(1 for c in captions if re.search(pattern, c))
    pct = (count / len(captions) * 100) if captions else 0
    print(f"{pattern_name.replace('_', ' ').title()}: {count:,} ({pct:.1f}%)")
    punct_data.append((pattern_name.replace('_', ' ').title(), count))

md_output.append(create_bar_chart(punct_data, max_width=40))
md_output.append("```\n")

# Quality indicators
print()
print("="*60)
print("CAPTION QUALITY INDICATORS")
print("="*60)

md_output.append("---\n\n## Caption Quality Indicators\n")
md_output.append("```")

quality_data = []
for indicator_type, keywords in QUALITY_INDICATORS.items():
    count = 0
    for caption in captions:
        caption_lower = caption.lower()
        if any(kw in caption_lower for kw in keywords):
            count += 1
    pct = (count / len(captions) * 100) if captions else 0
    print(f"{indicator_type.replace('_', ' ').title()} language: {count:,} ({pct:.1f}%)")
    quality_data.append((indicator_type.replace('_', ' ').title(), count))

md_output.append(create_bar_chart(quality_data, max_width=40))
md_output.append("```\n")

# Problematic patterns
print()
print("="*60)
print("PROBLEMATIC PATTERNS DETECTED")
print("="*60)

md_output.append("---\n\n## Problematic Patterns Detected\n")

problem_data = []
for problem_name, pattern in PROBLEMATIC_PATTERNS.items():
    count = sum(1 for c in captions if re.search(pattern, c, re.IGNORECASE))
    if count > 0:
        pct = (count / len(captions) * 100) if captions else 0
        print(f"{problem_name.replace('_', ' ').title()}: {count:,} ({pct:.1f}%)")
        problem_data.append((problem_name.replace('_', ' ').title(), count))

if problem_data:
    md_output.append("```")
    md_output.append(create_bar_chart(problem_data, max_width=40))
    md_output.append("```\n")
else:
    md_output.append("✅ No problematic patterns detected!\n")

# Most common words
print()
print("="*60)
print("MOST COMMON WORDS")
print("="*60)

md_output.append("---\n\n## Most Common Words\n")

all_words = []
for caption in captions:
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b[a-z]+\b', caption.lower())
    all_words.extend(words)

# Filter out very short words (likely articles, prepositions)
meaningful_words = [w for w in all_words if len(w) > 3]
word_freq = Counter(meaningful_words)

print("Top 20 words:")
md_output.append("**Top 20 words (>3 characters):**\n")
md_output.append("```")
top_words = [(word, count) for word, count in word_freq.most_common(20)]
md_output.append(create_bar_chart(top_words, max_width=40))
md_output.append("```\n")

for word, count in word_freq.most_common(20):
    pct = (count / len(all_words) * 100) if all_words else 0
    print(f"  {word}: {count:,} ({pct:.2f}%)")

# N-gram analysis
print()
print("="*60)
print("COMMON PHRASES (BIGRAMS)")
print("="*60)

md_output.append("---\n\n## Common Phrases (Bigrams)\n")

bigrams = []
for caption in captions:
    words = caption.lower().split()
    for i in range(len(words) - 1):
        bigrams.append(f"{words[i]} {words[i+1]}")

bigram_freq = Counter(bigrams)
print("Top 15 two-word phrases:")
md_output.append("**Top 15 two-word phrases:**\n")
md_output.append("```")
top_bigrams = [(phrase, count) for phrase, count in bigram_freq.most_common(15)]
md_output.append(create_bar_chart(top_bigrams, max_width=40))
md_output.append("```\n")

for phrase, count in bigram_freq.most_common(15):
    print(f"  '{phrase}': {count:,}")

# Reference statistics
if references:
    print()
    print("="*60)
    print("REFERENCE STATISTICS")
    print("="*60)
    
    md_output.append("---\n\n## Reference Statistics\n")
    
    ref_counts = [len(r) if isinstance(r, list) else 0 for r in references]
    print(f"Total entries with references: {len(references):,}")
    print(f"Average references per entry: {sum(ref_counts) / len(ref_counts):.1f}")
    print(f"Min references: {min(ref_counts)}")
    print(f"Max references: {max(ref_counts)}")
    print(f"Median references: {sorted(ref_counts)[len(ref_counts)//2]}")
    
    md_output.append(f"- **Total entries with references:** {len(references):,}")
    md_output.append(f"- **Average references per entry:** {sum(ref_counts) / len(ref_counts):.1f}")
    md_output.append(f"- **Min references:** {min(ref_counts)}")
    md_output.append(f"- **Max references:** {max(ref_counts)}")
    md_output.append(f"- **Median references:** {sorted(ref_counts)[len(ref_counts)//2]}")
    md_output.append("\n**Distribution:**\n")
    md_output.append("```")
    md_output.append(create_histogram(ref_counts, bins=10))
    md_output.append("```\n")

# Length distribution
print()
print("="*60)
print("LENGTH DISTRIBUTION")
print("="*60)

md_output.append("---\n\n## Length Distribution\n")
md_output.append("```")

length_bins = {
    "Very short (1-10)": (1, 10),
    "Short (11-30)": (11, 30),
    "Medium (31-60)": (31, 60),
    "Long (61-100)": (61, 100),
    "Very long (101-200)": (101, 200),
    "Extremely long (>200)": (201, float('inf'))
}

length_data = []
for label, (min_w, max_w) in length_bins.items():
    count = sum(1 for wc in word_counts if min_w <= wc <= max_w)
    pct = (count / len(word_counts) * 100) if word_counts else 0
    print(f"{label}: {count:,} ({pct:.1f}%)")
    length_data.append((label, count))

md_output.append(create_bar_chart(length_data, max_width=50, show_count=False))
md_output.append("```\n")

# Quality checks
print()
print("="*60)
print("QUALITY CHECKS")
print("="*60)

md_output.append("---\n\n## Quality Checks\n")

very_short = sum(1 for c in captions if len(c.split()) < 5)
very_long = sum(1 for c in captions if len(c.split()) > 100)
has_figure_word = sum(1 for c in captions if "figure" in c.lower())
starts_with_figure = sum(1 for c in captions if c.lower().startswith("figure"))
ends_incomplete = sum(1 for c in captions if c.endswith('...') or (len(c.split()) > 15 and not re.search(r'[.!?]$', c)))
has_numbers = sum(1 for c in captions if re.search(r'\d', c))
all_lowercase = sum(1 for c in captions if c.islower())
all_uppercase = sum(1 for c in captions if c.isupper() and len(c) > 10)
proper_capitalization = sum(1 for c in captions if c and c[0].isupper())

quality_checks = [
    ("Very short (< 5 words)", very_short),
    ("Very long (> 100 words)", very_long),
    ("Contains 'figure'", has_figure_word),
    ("Starts with 'Figure'", starts_with_figure),
    ("Incomplete sentences", ends_incomplete),
    ("Contains numbers", has_numbers),
    ("All lowercase", all_lowercase),
    ("All uppercase", all_uppercase),
    ("Proper capitalization", proper_capitalization),
]

print(f"Very short captions (< 5 words): {very_short:,} ({very_short/len(captions)*100:.1f}%)")
print(f"Very long captions (> 100 words): {very_long:,} ({very_long/len(captions)*100:.1f}%)")
print(f"Contains 'figure': {has_figure_word:,} ({has_figure_word/len(captions)*100:.1f}%)")
print(f"Starts with 'Figure': {starts_with_figure:,} ({starts_with_figure/len(captions)*100:.1f}%)")
print(f"Incomplete sentences: {ends_incomplete:,} ({ends_incomplete/len(captions)*100:.1f}%)")
print(f"Contains numbers: {has_numbers:,} ({has_numbers/len(captions)*100:.1f}%)")
print(f"All lowercase: {all_lowercase:,} ({all_lowercase/len(captions)*100:.1f}%)")
print(f"All uppercase: {all_uppercase:,} ({all_uppercase/len(captions)*100:.1f}%)")
print(f"Proper capitalization: {proper_capitalization:,} ({proper_capitalization/len(captions)*100:.1f}%)")

md_output.append("```")
md_output.append(create_bar_chart(quality_checks, max_width=40))
md_output.append("```\n")

# Data completeness
print()
print("="*60)
print("DATA COMPLETENESS")
print("="*60)

md_output.append("---\n\n## Data Completeness\n")

complete_entries = sum(1 for e in all_entries if e.get("s2orc_caption") and e.get("s2orc_references") and e.get("pdf_hash"))
print(f"Complete entries (caption + refs + hash): {complete_entries:,} ({complete_entries/len(all_entries)*100:.1f}%)")
print(f"Missing captions: {len(all_entries) - total_captions:,}")
print(f"Missing references: {len(all_entries) - total_with_refs:,}")
print(f"Missing pdf_hash: {len(all_entries) - len(pdf_hashes):,}")

md_output.append(f"- **Complete entries (caption + refs + hash):** {complete_entries:,} ({complete_entries/len(all_entries)*100:.1f}%)")
md_output.append(f"- **Missing captions:** {len(all_entries) - total_captions:,}")
md_output.append(f"- **Missing references:** {len(all_entries) - total_with_refs:,}")
md_output.append(f"- **Missing pdf_hash:** {len(all_entries) - len(pdf_hashes):,}\n")

# Sample captions
print()
print("="*60)
print("SAMPLE CAPTIONS")
print("="*60)

md_output.append("---\n\n## Sample Captions\n")

# Show a few examples
sample_indices = [0, len(captions)//4, len(captions)//2, 3*len(captions)//4, -1]
for i, idx in enumerate(sample_indices, 1):
    if 0 <= idx < len(captions):
        caption = captions[idx]
        print(f"\nSample {i} (entry {idx}):")
        print(f"  Length: {len(caption.split())} words")
        
        md_output.append(f"### Sample {i} (entry {idx})\n")
        md_output.append(f"**Length:** {len(caption.split())} words\n")
        md_output.append(f"**Text:**\n")
        
        # Truncate if too long
        if len(caption) > 200:
            print(f"  Text: {caption[:200]}...")
            md_output.append(f"> {caption[:300]}...\n")
        else:
            print(f"  Text: {caption}")
            md_output.append(f"> {caption}\n")

# Write markdown file
print()
print("="*60)
print("WRITING MARKDOWN REPORT")
print("="*60)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(md_output))

print(f"✓ Report saved to: {output_file}")

print()
print("="*60)
print("ANALYSIS COMPLETE")
print("="*60)
