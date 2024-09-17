import os
import json
import glob

def process_txt_files(base_dir):
    """
    This function processes all .txt files in the given base directory, extracts content,
    and generates conversations and image paths as per the given format.
    """
    all_conversations = []

    # Traverse the base directory and process each .txt file
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)

                # Open and read content from the .txt file
                with open(txt_path, 'r', encoding='utf-8') as f:
                    txt_content = f.read().strip()

                # Extract the base name of the file for the image name
                base_filename = os.path.splitext(file)[0].replace("09", "03")  # Modify file name as per your example

                # Search for any image file with various extensions (e.g., .jpeg, .jpg, .png)
                image_pattern = os.path.join(root.replace('/instruct/', '/'), base_filename + ".*")
                image_files = glob.glob(image_pattern)
                
                # If an image is found, use the first match
                image_path = image_files[0] if image_files else None

                if image_path:
                    # Create conversation structure
                    conversation_entry = {
                        "conversations": [
                            {
                                "from": "human",
                                "value": "Generate the description of the image.\n<image>"
                            },
                            {
                                "from": "gpt",
                                "value": txt_content
                            }
                        ],
                        "image": image_path
                    }

                    # Add the conversation entry to the list
                    all_conversations.append(conversation_entry)

    return all_conversations

def save_to_json(data, output_file):
    """
    Save the processed conversations to a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"Conversations saved to {output_file}")

# Specify the base directory where .txt files are located
base_dir = '/mnt/cache_share/Radiopaedia/instruct'
output_file = 'Radiopaedia.json'

# Process the directory and save the output
conversations = process_txt_files(base_dir)
save_to_json(conversations, output_file)
