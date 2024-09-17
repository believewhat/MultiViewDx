import os
import re
import json

def extract_questions_answers_with_image_path(content, txt_path, image_base_dir):
    """
    This function extracts questions and answers from the given content and matches them by number.
    It also generates the image path based on the provided txt_path and image_base_dir.
    """
    # Check if 'Question' or 'Questions' exists in the content, otherwise skip
    if 'Question' not in content:
        print(f"Not Found {txt_path}")
        return None, None

    paired_conversations = []
    
    # Improved regex to handle variations including #### Report, **Report**, _Report_, etc.
    report_match = re.search(r"(?i)(?:[*_#]{0,4}Description[*_#]{0,4}:\s*)(.*?)(?=\n+[*_#]{0,4}Questions?[*_#]{0,4}:|\Z)", content, re.DOTALL | re.IGNORECASE)
    
    if report_match:
        report = report_match.group(1).strip()

        # Add the report part as a conversation
        paired_conversations.append({
            "from": "human",
            "value": "Please generate the image findings of the following image:\n<image>"
        })
        paired_conversations.append({
            "from": "gpt",
            "value": report
        })
    else:
        print(f"未找到 Description 部分在文件: {txt_path}")
    
    # Extracting the question and answer blocks, handling #### Question, **Question**, _Question_, etc.
    questions_block_match = re.search(r"[*_#]{0,4}Questions?[*_#]{0,4}:\n+(.*?)\n+[*_#]{0,4}Answers?[*_#]{0,4}:", content, re.DOTALL | re.IGNORECASE)
    answers_block_match = re.search(r"[*_#]{0,4}Answers?[*_#]{0,4}:\n+(.*)", content, re.DOTALL | re.IGNORECASE)

    questions = []
    answers = []

    # Extracting questions
    if questions_block_match:
        questions_block = questions_block_match.group(1).strip()
        questions = re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.|$)", questions_block, re.DOTALL)

    # Extracting answers
    if answers_block_match:
        answers_block = answers_block_match.group(1).strip()
        answers = re.findall(r"(\d+)\.\s*(.*?)(?=\n\d+\.|$)", answers_block, re.DOTALL)

    # Matching questions with answers
    for question, answer in zip(questions, answers):
        q_num, q_text = question
        a_num, a_text = answer
        if q_num == a_num:  # Ensure that question and answer numbers match
            paired_conversations.append({
                "from": "human",
                "value": q_text.strip()
            })
            paired_conversations.append({
                "from": "gpt",
                "value": a_text.strip()
            })

    # Generate the correct image path based on the txt_path
    txt_dirname = os.path.dirname(txt_path)  # Get the directory of the txt file
    txt_filename = os.path.basename(txt_path)  # Get the filename of the txt file
    image_filename = txt_filename.replace('.txt', '.png')  # Replace the .txt extension with .png
    relative_path = txt_dirname.split('json', 1)[1]  # Extract the relative path
    image_path = os.path.join(image_base_dir, relative_path, image_filename)  # Full image path

    return paired_conversations, image_path

def process_directory(base_dir, image_base_dir):
    """
    This function processes all .txt files in the given base directory and extracts conversations and image paths.
    It saves the extracted data to a JSON file.
    """
    all_conversations = []

    # Traverse the base directory and process each .txt file
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                txt_path = os.path.join(root, file)
                # Open and read content from the .txt file
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract conversations and generate the image path
                conversations, image_path = extract_questions_answers_with_image_path(content, txt_path, image_base_dir)

                if conversations and image_path:  # Ensure both are valid
                    all_conversations.append({
                        "conversations": conversations,
                        "image": image_path
                    })

    # Save the extracted conversations to a JSON file
    with open('Quilt.json', 'w', encoding='utf-8') as json_file:
        json.dump(all_conversations, json_file, indent=4, ensure_ascii=False)

    print(f"Conversations extracted and saved to 'Quilt.json'.")

# Specify base directories and run the process
base_dir = '/mnt/cache_share/MedTrinity-25M/Quilt/json'
image_base_dir = '/mnt/cache_share/MedTrinity-25M/Quilt/json/images'
process_directory(base_dir, image_base_dir)
