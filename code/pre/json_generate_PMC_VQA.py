import csv
import json
import os
base_url = "/mnt/cache_share/MedTrinity-25M/PMC-VQA/figures"
def process_csv(input_csv, output_json):
    """
    This function reads the CSV file, processes the question, options, and answer,
    and generates a conversation format for the output.
    """
    conversations_list = []

    # Read the CSV file
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        # Process each row in the CSV
        for row in csvreader:
            # Create the human question and options format
            question_with_options = f"{row['Question']}\n" \
                                    f"A. {row['Choice A'].strip()}\n" \
                                    f"B. {row['Choice B'].strip()}\n" \
                                    f"C. {row['Choice C'].strip()}\n" \
                                    f"D. {row['Choice D'].strip()}\n" \
                                    "Answer with the option's letter from the given choices directly.\n<image>"
            
            # Create a conversation entry
            conversation_entry = {
                "conversations": [
                    {
                        "from": "human",
                        "value": question_with_options
                    },
                    {
                        "from": "gpt",
                        "value": row['Answer'].strip()
                    }
                ],
                "image": os.path.join(base_url, row['Figure_path'].strip())
            }

            # Add the conversation entry to the list
            conversations_list.append(conversation_entry)

    # Save the processed conversations to the output JSON file
    with open(output_json, 'w', encoding='utf-8') as out_file:
        json.dump(conversations_list, out_file, indent=4, ensure_ascii=False)

    print(f"Conversations saved to {output_json}")

# Specify the input CSV and output JSON file paths
input_csv = '/mnt/cache_share/MedTrinity-25M/PMC-VQA/train_2.csv'  # Replace with your actual CSV file path
output_json = 'PMC_VQA.json'

# Process the CSV and generate the conversations output
process_csv(input_csv, output_json)
