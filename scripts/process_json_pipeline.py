#!/usr/bin/env python3
"""
Combined JSON filtering pipeline.
Performs multiple filtering steps on JSONL data:
1. Extract selected keys and combine pdf_hash + fig_uri
2. Remove entries with null s2orc_caption
3. Remove entries with null s2orc_references

Configure the steps and file names below.
"""

import json

# Configuration - edit these
input_file = "s2_full_figures.jsonl"
output_step1 = "s2_parsed.json"
output_step2 = "s2_parsed_filtered.json"
output_final = "s2_parsed_with_refs.json"

# Step 1: Extract selected keys and combine pdf_hash + fig_uri
print("Step 1: Extracting and combining pdf_hash + fig_uri...")
step1_total = 0

with open(input_file, "r") as f_in, open(output_step1, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        step1_total += 1
        
        # Create output with combined pdf_hash and fig_uri
        final_data = {}
        
        # Combine pdf_hash and fig_uri with underscore
        if "pdf_hash" in data and "fig_uri" in data:
            final_data["pdf_hash"] = f"{data['pdf_hash']}_{data['fig_uri']}"
        elif "pdf_hash" in data:
            final_data["pdf_hash"] = data["pdf_hash"]
        
        if "s2orc_caption" in data:
            final_data["s2orc_caption"] = data["s2orc_caption"]
        if "s2orc_references" in data:
            final_data["s2orc_references"] = data["s2orc_references"]
        
        f_out.write(json.dumps(final_data, ensure_ascii=False) + "\n")

print(f"  Processed: {step1_total} entries")
print(f"  Output: {output_step1}")
print()

# Step 2: Remove entries with null s2orc_caption
print("Step 2: Removing entries with null s2orc_caption...")
step2_total = 0
step2_kept = 0
step2_removed = 0

with open(output_step1, "r") as f_in, open(output_step2, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        
        step2_total += 1
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        if data.get("s2orc_caption") is not None:
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            step2_kept += 1
        else:
            step2_removed += 1

print(f"  Total: {step2_total}")
print(f"  Kept: {step2_kept}")
print(f"  Removed (null captions): {step2_removed}")
print(f"  Output: {output_step2}")
print()

# Step 3: Remove entries with null s2orc_references
print("Step 3: Removing entries with null s2orc_references...")
step3_total = 0
step3_kept = 0
step3_removed = 0

with open(output_step2, "r") as f_in, open(output_final, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        
        step3_total += 1
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        if data.get("s2orc_references") is not None:
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            step3_kept += 1
        else:
            step3_removed += 1

print(f"  Total: {step3_total}")
print(f"  Kept (with references): {step3_kept}")
print(f"  Removed (null references): {step3_removed}")
print(f"  Output: {output_final}")
print()

# Final summary
print("="*50)
print("PIPELINE COMPLETE")
print("="*50)
print(f"Input file: {input_file}")
print(f"Final output: {output_final}")
print(f"Started with: {step1_total} entries")
print(f"After removing null captions: {step2_kept} entries")
print(f"Final count (with captions and references): {step3_kept} entries")
print(f"Total removed: {step1_total - step3_kept} entries")
