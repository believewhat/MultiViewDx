#!/bin/bash

#input_file="/mnt/public_data_01/33_Pubmed/paths_with_tar_gz.txt"
input_file="check.txt"
output_path="/home/data/38_Pubmed_cleaned"
batch_size=100000
max_parallel=1000

# Create directories if they don't exist
mkdir -p "$output_path/images"
mkdir -p "$output_path/captions"

# Read the input file line by line
mapfile -t tar_gz_files < "$input_file"
total_files=${#tar_gz_files[@]}
echo $total_files
batches=$((total_files / batch_size + (total_files % batch_size != 0)))

# Process each batch
for ((batch=0; batch<batches; batch++)); do
    start=$((batch * batch_size))
    end=$((start + batch_size))
    if [ $end -gt $total_files ]; then
        end=$total_files
    fi

    echo "Processing batch $((batch + 1))/$batches: files $start to $((end - 1))"

    # Process each file in the batch
    for ((i=start; i<end; i++)); do
        file_path="/mnt/public_data_01/33_Pubmed/${tar_gz_files[$i]}"
        output_captions="$output_path/captions/$(basename "${tar_gz_files[$i]}" .tar.gz).json"
        
        # Check if the caption file already exists
        #if [ ! -f "$output_captions" ]; then
        #    python deal2.py "$file_path" "$output_path" "$output_captions" &
        #else
        #    echo "Skipping ${tar_gz_files[$i]} as captions already exist."
        #fi
        python deal2.py "$file_path" "$output_path" "$output_captions" &
        # Control the parallel processes
        if (( (i + 1) % max_parallel == 0 )); then
            wait
        fi
    done

    # Wait for the last batch of processes to finish
    wait
done
