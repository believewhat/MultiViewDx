import os
import json
import tarfile
import shutil
import concurrent.futures
import argparse

def find_json_files(directory):
    """Find all JSON files in the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]

def read_json_file(filepath):
    """Read the content of a JSON file."""
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {filepath}")
            return None

def find_image_path(root, image_filename):
    possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.bmp']
    for ext in possible_extensions:
        full_path = os.path.join(root, image_filename + ext)
        if os.path.exists(full_path):
            return full_path
    return None

def process_single_json_file(json_file, tar_gz_paths, extract_dir, images_dir):
    # Read JSON file
    data = read_json_file(json_file)
    if not data or not isinstance(data, list) or len(data) == 0:
        return
    
    # Get the JSON file name without extension
    json_filename = os.path.splitext(os.path.basename(json_file))[0]
    
    # Skip if the destination directory already exists
    dst_image_dir = os.path.join(images_dir, json_filename)
    if os.path.exists(dst_image_dir):
        print(f"Skipping {json_filename} as destination directory already exists")
        return
    
    # Skip JSON file if corresponding tar.gz file is not found
    tar_gz_path = tar_gz_paths.get(json_filename)
    if not tar_gz_path:
        print(f"No corresponding tar.gz file found for {json_filename}")
        return
    
    # Extract the tar.gz file
    extraction_path = os.path.join(extract_dir, json_filename)    
    try:
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
    except:
        error = 1
    
    # Process JSON data
    image_filenames = [item['image_filename'] for item in data if 'image_filename' in item]

    # Move specified images to the destination directory
    os.makedirs(dst_image_dir, exist_ok=True)
    
    for image_filename in image_filenames:
        src_image_path = find_image_path(extraction_path, image_filename)
        if src_image_path:
            dst_image_path = os.path.join(dst_image_dir, os.path.basename(src_image_path))
            shutil.move(src_image_path, dst_image_path)
        else:
            print(f"Image {image_filename} not found in {extraction_path}")

    # Clean up extracted files except the images
    shutil.rmtree(extraction_path)

def process_json_files(json_dir, paths_file, extract_dir, images_dir, num_workers):
    # Read paths_with_tar_gz.txt and create a dictionary
    tar_gz_paths = {}
    with open(paths_file, 'r') as f:
        for line in f:
            tar_gz_path = line.strip()
            file_name = os.path.basename(tar_gz_path).replace('.tar.gz', '')
            tar_gz_paths[file_name] = os.path.join("/mnt/public_data_01/33_Pubmed", tar_gz_path)
    
    # Find all JSON files in the given directory
    json_files = find_json_files(json_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_json_file, json_file, tar_gz_paths, extract_dir, images_dir) for json_file in json_files]
        for future in concurrent.futures.as_completed(futures):
            future.result()

def main():
    parser = argparse.ArgumentParser(description="Process JSON files and extract images.")
    parser.add_argument("json_directory", type=str, help="Directory containing JSON files")
    parser.add_argument("paths_with_tar_gz", type=str, help="File containing paths to tar.gz files")
    parser.add_argument("extract_directory", type=str, help="Directory to extract tar.gz files")
    parser.add_argument("images_directory", type=str, help="Directory to save extracted images")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    process_json_files(args.json_directory, args.paths_with_tar_gz, args.extract_directory, args.images_directory, args.num_workers)

if __name__ == "__main__":
    main()
