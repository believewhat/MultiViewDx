import os
import json
import glob
import base64
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set API key and client for OpenAI
client = OpenAI()

# Define system prompt template
system_prompt = """
Please generate a set of questions and answers based on the given caption and image. The questions should be centered around the image, and the answers must be found within the caption or inferred from the image. Ideally, the answers should be directly found in the caption. Depending on the length of the caption, create 1-8 questions with corresponding answers, ranging in difficulty from easy to more challenging.

The returned content must strictly follow this format:
Question:
1.
2.
3.
4.
5.
...
Answer:
1.
2.
3.
4.
5.
...
Now here is the image and caption:
{content}
"""

# Define directories
image_dir = "/home/data/38_Pubmed_cleaned/images2"
save_dir = "/home/data/38_Pubmed_cleaned/pmc_instruct"

# Function to encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to parse the QA content
def parse_qa_content(content):
    questions = []
    answers = []
    lines = content.splitlines()
    question_start = False
    answer_start = False

    # Regular expression to match the start of questions or answers (numbered list)
    question_pattern = re.compile(r'^\d+\.\s*')

    # Loop through lines to find questions and answers
    for line in lines:
        stripped_line = line.strip()
        if 'Question:' in stripped_line:
            question_start = True
            continue
        elif 'Answer:' in stripped_line:
            answer_start = True
            continue

        if question_start and not answer_start:
            # Remove enumeration from the beginning of questions if present
            question = question_pattern.sub('', stripped_line)
            if question:  # Ensure non-empty question
                questions.append(question.strip())
        elif answer_start:
            # Remove enumeration from the beginning of answers if present
            answer = question_pattern.sub('', stripped_line)
            if answer:  # Ensure non-empty answer
                answers.append(answer.strip())

    return questions, answers

# Function to process each file
def process_file(line, base_caption_directory):
    conversations_data = []
    line = line.strip()
    parts = line.split('/')
    pmc_folder = parts[-2]
    txt_filename = parts[-1]
    image_id = os.path.splitext(txt_filename)[0]  # Remove extension

    output_json_path = os.path.join(save_dir, pmc_folder)
    os.makedirs(output_json_path, exist_ok=True)
    output_json_path = os.path.join(output_json_path, f"{image_id}.json")

    if os.path.exists(output_json_path):
        return

    # Construct JSON file path
    json_file_path = os.path.join(base_caption_directory, pmc_folder + '.json')

    # Read JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Find the matching caption
        for item in data:
            if item['image_filename'] == image_id:
                caption = item.get('caption', 'No caption found')
                break
        else:
            return  # Skip if no matching caption found

        # Find the corresponding image file
        image_path_pattern = os.path.join(image_dir, pmc_folder, image_id + '.*')
        image_files = glob.glob(image_path_pattern)
        if not image_files:
            return  # Skip if no image file found
        image_path = image_files[0]

        # Encode image to base64
        base64_image = encode_image(image_path)

        # Generate QA pairs using OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt.replace("{content}", caption),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )
        result = response.choices[0].message.content
        # Split the result into questions and answers
        questions, answers = parse_qa_content(result)

        # Format the conversation data
        conversation = []
        for question, answer in zip(questions, answers):
            conversation.append({"role": "user", "value": question})
            conversation.append({"role": "gpt", "value": answer})

        # Add to conversations data
        conversations_data.append({
            "image_id": image_id,
            "conversations": conversation
        })
        
        # Save the conversations data to a JSON file
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(conversations_data, outfile, ensure_ascii=False, indent=4)

# Function to extract captions and process in parallel
def extract_captions(filtered_files_path, base_caption_directory):
    with open(filtered_files_path, 'r') as file:
        lines = file.readlines()
    #process_file(lines[0], base_caption_directory)
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(process_file, line, base_caption_directory) for line in lines]
        for future in as_completed(futures):
            future.result()  # Ensure any exceptions are raised

# Example usage
filtered_files_path = 'filtered_pmc_category.txt'
base_caption_directory = '/home/data/38_Pubmed_cleaned/captions/'
extract_captions(filtered_files_path, base_caption_directory)
