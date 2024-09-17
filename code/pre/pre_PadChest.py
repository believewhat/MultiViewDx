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
from collections import defaultdict



data = pd.read_csv("/mnt/cache_share/MedTrinity-25M/PadChest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")


system_prompt = """
Analyze the provided X-Ray image and generate a detailed and professional medical report that describes only the abnormalities, significant features, or relevant observations directly seen in the image. Use precise medical terminology and maintain a formal tone. Do not include any introductory phrases, such as "The provided image reveals," or any concluding remarks. Start the report directly with the findings, and focus only on describing what is observed in the image.

Here is the Clinical Note (if available):
{Clinical Note}

Here is the image:
"""

os.environ["OPENAI_API_KEY"]=''

client = OpenAI()


import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

output_path = "/mnt/cache_share/MedTrinity-25M/PadChest/json"
def process_and_save(row):
    image_id = row['ImageID']
    report = row['Report']
    output_file_path = os.path.join(output_path, '.'.join(image_id.split('.')[:-1])+'.txt')
    if os.path.exists(output_file_path):
        return
    base64_image = encode_image(os.path.join('/mnt/cache_share/MedTrinity-25M/PadChest', image_id))

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt.replace('{Clinical Note}', report),
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
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return
    print(response)
    result = response.choices[0].message.content
    
    with open(output_file_path, 'w') as f:
        f.write(result)
    

# 使用ThreadPoolExecutor进行并行处理
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(process_and_save, [row for _, row in data.iterrows()])