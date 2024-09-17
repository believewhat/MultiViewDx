"""
import os
import json
from concurrent.futures import ThreadPoolExecutor

# 目标文件夹路径
input_folder = '/home/data/38_Pubmed_cleaned/captions'
output_file = '/home/data/38_Pubmed_cleaned/combined_captions.json'

# 用于存储所有提取的数据
all_data = []

def process_file(filename):
    file_path = os.path.join(input_folder, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # 获取不带扩展名的文件名
        base_filename = os.path.splitext(filename)[0]
        
        # 修改 image_filename 字段
        for item in data:
            if 'image_filename' not in item:
                continue
            item['image_filename'] = f"{base_filename}/{item['image_filename']}"
        
        return data

# 遍历文件夹中的所有JSON文件并使用线程池并行处理
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, filename) for filename in os.listdir(input_folder) if filename.endswith('.json')]
    
    for future in futures:
        all_data.extend(future.result())

# 将所有数据写入到一个新的JSON文件中
with open(output_file, 'w', encoding='utf-8') as output:
    json.dump(all_data, output, ensure_ascii=False, indent=4)

print(f"数据已合并到 {output_file}")
"""
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

# 定义读取 JSON 文件的函数
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 定义转换数据的函数
def transform_item(idx, item):
    if "caption" not in item or 'image_filename' not in item:
        return None
    new_item = {
        "id": f"{idx:09d}",
        "image": item["image_filename"],
        "conversations": [
            {
                "from": "human",
                "value": "Generate the caption of this figure. \n<image>"
            },
            {
                "from": "gpt",
                "value": item["caption"]
            }
        ]
    }
    return new_item

# 读入 JSON 文件
input_file_path = '/home/data/38_Pubmed_cleaned/combined_captions.json'
input_data = read_json(input_file_path)

# 并行转换数据
output_data = []
with ThreadPoolExecutor() as executor:
    results = executor.map(transform_item, range(len(input_data)), input_data)
    output_data = list(results)

# 保存新的 JSON 文件
output_file_path = '/home/data/38_Pubmed_cleaned/combined_captions_revise.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
print(len(output_data))
print("Done!")
"""
"""
import json
from concurrent.futures import ThreadPoolExecutor

# 定义读取 JSON 文件的函数
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 定义保存 JSON 文件的函数
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 过滤函数，检查 gpt value 的单词数是否少于 20
def filter_sample(item):
    if item is None or "conversations" not in item:
        return None
    gpt_value = item["conversations"][1]["value"]
    word_count = len(gpt_value.split())
    return item if word_count >= 20 else None

# 并行过滤函数
def filter_samples_parallel(data):
    with ThreadPoolExecutor() as executor:
        results = executor.map(filter_sample, data)
        return [result for result in results if result is not None]

# 文件路径
input_file_path = '/home/data/38_Pubmed_cleaned/combined_captions_revise.json'
output_file_path = '/home/data/38_Pubmed_cleaned/combined_captions_revise_filtered.json'

# 读入 JSON 文件
input_data = read_json(input_file_path)

# 过滤数据（并行）
filtered_data = filter_samples_parallel(input_data)

# 保存过滤后的数据
save_json(filtered_data, output_file_path)

print("Done!")
"""
import os
import json
from concurrent.futures import ThreadPoolExecutor

# Define the paths
data_path = "/home/data/38_Pubmed_cleaned/images2"
output_path = "/home/data/38_Pubmed_cleaned/combined_captions_revise.json"
json_file_path = "/home/data/38_Pubmed_cleaned/combined_captions_revise_filtered.json"  # Replace with the actual path to your JSON file

# Function to find the correct image file with extension
def find_image_file(image_name):
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.tiff', '.bmp']:
        if os.path.isfile(os.path.join(data_path, image_name + ext)):
            return image_name + ext
    return None

# Load the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Function to update image path in a single item
def update_image_path(item):
    image_name = item['image']
    updated_image_name = find_image_file(image_name)
    if updated_image_name:
        item['image'] = os.path.join(data_path, updated_image_name)
    return item

# Use ThreadPoolExecutor to update image paths in parallel
with ThreadPoolExecutor() as executor:
    data = list(executor.map(update_image_path, data))

# Save the updated JSON
with open(output_path, 'w') as file:
    json.dump(data, file, indent=4)
