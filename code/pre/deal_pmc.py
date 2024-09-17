import json
import json
import ipdb
import glob
import os
import ipdb
from openai import OpenAI
import os
import json
import csv
import pandas as pd
import argparse
os.environ["OPENAI_API_KEY"]=''

client = OpenAI()


import json
import os
import random
"""
# 定义输入和输出文件路径
input_json_path = '/home/data/38_Pubmed_cleaned/combined_captions_revise_filtered.json'  # 请替换为你的JSON文件路径
output_txt_path = 'output_name.txt'

# 读取JSON文件
with open(input_json_path, 'r') as file:
    data = json.load(file)

# 提取所有的JSON文件名
json_filenames = set()
for item in data:
    image_path = item.get('image', '')
    if image_path:
        json_filename = os.path.dirname(image_path) + '.json' + ' ' + image_path
        json_filenames.add(json_filename)

# 转换为列表并进行打乱
json_filenames = list(json_filenames)
random.shuffle(json_filenames)
import ipdb
ipdb.set_trace()
# 保存到txt文件
with open(output_txt_path, 'w') as file:
    for filename in json_filenames:
        file_name = filename.split()[0].split('.')[0]
        
        files_image = filename.split('/')[-1]
        save_path = f'/home/data/38_Pubmed_cleaned/gpt4/{file_name}/{files_image}.txt'
        if os.path.exists(save_path):
            continue
        file.write(filename + '\n')

print(f"已将所有文件名保存到 {output_txt_path}")

"""
system_prompt="""
I have a collection of PMC paper images along with their corresponding captions. I need your help to classify them. Specifically, there are three main categories of information to be filled: Topic/Disease, Modality, and Body part.

Some images are medical images while others are statistical charts from papers. Here's the process you need to follow:

First, determine if the image is about a Disease. If it is, provide three keywords related to the disease. If it is not, simply label it as "other".

Next, classify the Modality. Check which category from the provided list the image belongs to. If it does not fit any category, label it as "other".

Finally, look at the Body part. Determine if the image pertains to a body part description. If it does not, label it as "other".

Below is the list of categories:

1. Topic/Disease

2.Modality:

- X-ray
- DSA
- CT
- MR
- PET/SPECT
- Altralsound 
- Pathology 
- Camara 
- gross pathology
- Demotology 
- OCT
- EM
- Electrophoresis
- Colonscopy
- Simulated illustration
- Radioisotope
- Optical Image
- Mitotic
- Other

3.Body part:

- Head & neck
- Thoroax
- Abdomen
- Extremity 
- Upper
- Lower
- Other

Now, I need you to perform the classification. Given the caption

Caption: 
{caption}

Please provide the categories in the following format:

1. Topic/Disease: key1, key2, key3
2. Modality: (single category)
3. Body part: (single category)

The caption is used to help you to understand the image. The classification result should only be based on the image. For example, if the image is a camera image but the caption is to describe a CT image. You should return Camara. 

Remember to strictly follow the output format and do not provide any extra information.
"""

import sys
import json
import os
import glob
from IPython.display import Image, display, Audio, Markdown
import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_line(json_filename, image_filename):
    
    # 构造json文件路径
    json_base_dir = '/home/data/38_Pubmed_cleaned/captions'
    json_filepath = os.path.join(json_base_dir, json_filename)
    file_name = json_filename.split('.')[0]
    folder_path = f'/home/data/38_Pubmed_cleaned/gpt4/{file_name}'
    os.makedirs(folder_path, exist_ok=True)
    save_file = os.path.join(folder_path, image_filename.split('/')[1]) + '.txt'
    if os.path.exists(save_file):
        return


    # 读取json文件并提取caption
    caption = ''
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
            # 遍历json数据并提取caption
            for item in data:
                if item['image_filename'] == image_filename.split('/')[1]:
                    caption = item['caption']
                    break  # 找到对应的caption后跳出循环
    
    # 查找图片文件
    image_dir = '/home/data/38_Pubmed_cleaned/images2'  # 假设图片存储在该目录下
    image_path_pattern = os.path.join(image_dir, image_filename + '.*')
    image_files = glob.glob(image_path_pattern)
    print(image_path_pattern, image_files)
    
    image_path = image_files[0]  # 取找到的第一个文件
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt.replace("{caption}", caption),
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
        max_tokens=300,
    )
    result = response.choices[0].message.content
    
    with open(save_file, 'w') as file:
        file.write(result)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 process_json_image.py <line_content> <line_number>")
        sys.exit(1)
    
    json_filename = sys.argv[1]
    image_filename = sys.argv[2]
    process_line(json_filename, image_filename)





"""
for x in data:
    if 'gpt3_rate' in x:
        continue
    pred = x['gpt3_answer']
    target = x["output"]
    question = x["input"]
    text = prompt.replace('{pred}', pred)
    text = text.replace('{target}', target)
    text = text.replace('{question}', question)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", f"content": text},
        ]
    )
    result = response.choices[0].message.content
    json_string = result.strip('```json\n')
    json_object = json.loads(json_string)
    x['gpt3_rate_q1'] = json_object["diagnosis 1"]["question 1"]
    x['gpt3_rate_q2'] = json_object["diagnosis 1"]["question 2"]
    x['gpt3_rate_q3'] = json_object["diagnosis 1"]["question 3"]
    x['gpt3_rate'] = json_object["overall score"]
"""