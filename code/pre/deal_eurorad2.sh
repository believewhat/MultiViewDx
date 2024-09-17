#!/bin/bash

JSON_FILE=$1

# Read the JSON file and process each entry
INDEX=0
jq -c '.[]' "$JSON_FILE" | while IFS= read -r LINE; do
    # Call the Python script with the current index and the JSON content
    python3 deal_euro2.py "$INDEX" "$LINE"
    ((INDEX++))
    break  # Exit after the first JSON object is processed
done
echo "Processing complete."
