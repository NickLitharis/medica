#!/bin/bash
# Check if a file matching fig_id exists in a specified folder

# Configuration - edit these
json_file="test_first5.json"
search_folder="/media/marocos/C61A40931A408303/release/figures"  # Folder to search in

# Extract fig_id from JSON file
fig_id=$(python3 -c "import json; print(json.load(open('$json_file'))['pdf_hash'])")

echo "Looking for fig_id: $fig_id"
echo "Searching in folder: $search_folder"
echo ""

# Check if the folder exists
if [ ! -d "$search_folder" ]; then
    echo "ERROR: Folder '$search_folder' does not exist"
    exit 1
fi

# Loop through files in the folder and check for matches
found=false
for file in "$search_folder"/*; do
    filename=$(basename "$file")
    if [[ "$filename" == *"$fig_id"* ]] || [[ "$fig_id" == *"$filename"* ]]; then
        echo "✓ FOUND: $file"
        found=true
    fi
done

if [ "$found" = false ]; then
    echo "✗ NOT FOUND: No matching file for fig_id '$fig_id' in '$search_folder'"
    exit 1
fi
