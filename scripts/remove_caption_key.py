#!/usr/bin/env python3
"""
Remove the s2orc_caption key from entries where it is null.
Keeps the entry but removes the null key.
"""

import json

# Configuration - edit these
input_file = "s2_parsed_filtered.jsonl"
output_file = "s2_parsed_no_null_keys.jsonl"

# Read JSONL file line by line and remove null s2orc_caption keys
total = 0
removed_keys = 0

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
        
        # Remove s2orc_caption key if it's null
        if "s2orc_caption" in data and data["s2orc_caption"] is None:
            del data["s2orc_caption"]
            removed_keys += 1
        
        # Write the modified entry
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Total entries: {total}")
print(f"Removed null s2orc_caption keys: {removed_keys}")
