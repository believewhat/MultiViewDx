import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义路径
jsonl_files = [
    "/mnt/cache_share/MedTrinity-25M/PMC_OA/test.jsonl",
    "/mnt/cache_share/MedTrinity-25M/PMC_OA/train.jsonl",
    "/mnt/cache_share/MedTrinity-25M/PMC_OA/valid.jsonl"
]

output_dir = "/home/data/38_Pubmed_cleaned/gpt4_wrong"
pmc_instruct_dir = "/home/data/38_Pubmed_cleaned/pmc_instruct"
images_dir = "/home/data/38_Pubmed_cleaned/images2"
os.makedirs(output_dir, exist_ok=True)

# 存储pmcid和url_name组合去重后的数据
unique_data = {}

# 合并jsonl文件并去重
for file_path in jsonl_files:
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            pmcid = data.get("pmcid")
            url_name = data.get("url_name")
            if pmcid and url_name:
                key = (pmcid, url_name)
                if key not in unique_data:
                    unique_data[key] = data

def process_data(pmcid, url_name):
    pmc_instruct_path = os.path.join(pmc_instruct_dir, pmcid, url_name.replace('.jpg', '.txt'))
    image_path = os.path.join(images_dir, pmcid, url_name)

    # 检查文件是否存在
    if os.path.exists(pmc_instruct_path) or not os.path.exists(image_path):
        return  # 如果文件存在于pmc_instruct或者图片不存在于images2目录中，跳过

    # 构造输出内容
    output_content = (
        "1. Topic/Disease: thyroid\n"
        "2. Modality: Modality\n"
        "3. Body part: Other\n"
    )

    # 创建pmcid命名的子文件夹
    pmcid_dir = os.path.join(output_dir, pmcid)
    os.makedirs(pmcid_dir, exist_ok=True)

    # 输出文件路径
    output_file_path = os.path.join(pmcid_dir, url_name.replace('.jpg', '.txt'))
    
    # 写入文件
    with open(output_file_path, 'w') as output_file:
        output_file.write(output_content)
    return output_file_path

# 并行处理数据
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_data, pmcid, url_name) for pmcid, url_name in unique_data.keys()]
    for future in as_completed(futures):
        result = future.result()
        if result:
            print(f"Processed and saved: {result}")

print("Processing complete. Files have been written to:", output_dir)
