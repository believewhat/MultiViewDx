import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor

import sys
import json
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
import base64
import time
import concurrent.futures

os.environ["OPENAI_API_KEY"]=''

client = OpenAI()

save_file = "/mnt/cache_share/MedTrinity-25M/DeepLesion/instruct"

system_prompt = """
Analyze the provided CT scan images and generate a set of questions and answers based on the image. Create 1-3 valuable questions with corresponding answers.

The returned content must strictly follow this format:
Question:
1.
2.
...
Answer:
1.
2.
...
"""


import base64
directory_path = '/mnt/cache_share/MedTrinity-25M/DeepLesion/Key_slices'
output_directory = '/mnt/cache_share/MedTrinity-25M/DeepLesion/instruct'
png_files = glob.glob(os.path.join(directory_path, '*.png'))
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 遍历文件并读取
def process_file(file_path):
    output_file_path = os.path.join(output_directory, os.path.basename(file_path) + '.txt')
    if os.path.exists(output_file_path):
        return
    base64_image = encode_image(file_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
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
    
    # 保存结果为txt文件
    
    with open(output_file_path, 'w') as f:
        f.write(result)

def main():
    png_files = glob.glob(os.path.join(directory_path, '*.png'))

    # 使用 ThreadPoolExecutor 来并行处理文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(process_file, png_files)

if __name__ == "__main__":
    main()