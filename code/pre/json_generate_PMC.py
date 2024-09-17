import os
import json
import glob

def process_json_files(json_base_dir, image_base_dir):
    """
    This function processes all JSON files in the given json_base_dir, extracts content,
    modifies the first question, and adds the corresponding image paths by searching in the image_base_dir using glob.
    """
    combined_data = []

    # Traverse the JSON base directory and process each .json file
    for root, dirs, files in os.walk(json_base_dir):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)

                # Open and read content from the JSON file
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_content = json.load(f)

                # Extract the base name of the file for the image search
                base_filename = os.path.splitext(file)[0]  # Remove the .json extension

                # Search for any image file with various extensions (e.g., .jpeg, .jpg, .png)
                image_pattern = os.path.join(root.replace(json_base_dir, image_base_dir), base_filename + ".*")
                image_files = glob.glob(image_pattern)

                # If an image is found, use the first match; otherwise, set image_path to None
                image_path = image_files[0] if image_files else None

                # Modify conversations
                for i, conversation in enumerate(json_content):
                    for j, entry in enumerate(conversation['conversations']):
                        # Change "user" to "human"
                        if entry['role'] == 'user':
                            entry['role'] = 'human'
                        
                        # Add "\n<image>" to the first question
                        if j == 0:
                            entry['value'] = entry['value'].strip() + "\n<image>"
                    
                    # Add the image path to each entry
                    conversation['image'] = image_path

                # Append the modified content to combined_data
                combined_data.extend(json_content)

    return combined_data

def save_to_json(data, output_file):
    """
    Save the combined data to a single JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"Combined data saved to {output_file}")

# Specify the base directories where .json files and images are located
json_base_dir = '/home/data/38_Pubmed_cleaned/pmc_instruct'
image_base_dir = '/home/data/38_Pubmed_cleaned/images2'
output_file = 'PMC_Instruct.json'

# Process the directory and save the combined output
combined_data = process_json_files(json_base_dir, image_base_dir)
save_to_json(combined_data, output_file)
