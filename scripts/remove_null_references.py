#!/usr/bin/env python3
"""
Remove entries where s2orc_references is null.
Deletes the entire entry if s2orc_references is null.
"""

import json

# Configuration - edit these
input_file = "s2_parsed_filtered.json"
output_file = "s2_parsed_with_refs.json"

# Read JSONL file line by line and keep only entries with non-null s2orc_references
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
        
        # Check if s2orc_references exists and is not null
        if data.get("s2orc_references") is not None:
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept += 1
        else:
            removed += 1

print(f"Total entries: {total}")
print(f"Kept (with references): {kept}")
print(f"Removed (null references): {removed}")
