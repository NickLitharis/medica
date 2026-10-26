#!/usr/bin/env python3
"""
Remove entries with null s2orc_caption from a JSONL file.
"""

import json

# Configuration - edit these
input_file = "s2_parsed.json"
output_file = "s2_parsed_filtered.json"

# Read JSONL file line by line and write only entries with non-null s2orc_caption
total = 0
kept = 0
removed = 0

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        
        total += 1
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Check if s2orc_caption exists and is not null
        if data.get("s2orc_caption") is not None:
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept += 1
        else:
            removed += 1

print(f"Total entries: {total}")
print(f"Kept: {kept}")
print(f"Removed (null captions): {removed}")
