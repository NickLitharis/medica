#!/usr/bin/env python3
"""
Extract selected fields from a JSONL file.
Keeps only pdf_hash, s2orc_caption, and s2orc_references from each line.
"""

import json

# Configuration - edit these
input_file = "s2_full_figures.jsonl"
output_file = "s2_parsed.json"

# Read JSONL file line by line and write filtered output
with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Create the output with only the desired fields
        final_data = {}
        
        # Add fields if they exist
        if "pdf_hash" in data:
            final_data["pdf_hash"] = data["pdf_hash"]
        if "s2orc_caption" in data:
            final_data["s2orc_caption"] = data["s2orc_caption"]
        if "s2orc_references" in data:
            final_data["s2orc_references"] = data["s2orc_references"]
        
        # Write as JSONL (one JSON object per line)
        f_out.write(json.dumps(final_data, ensure_ascii=False) + "\n")
