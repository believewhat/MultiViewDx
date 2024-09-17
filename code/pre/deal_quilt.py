import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
import base64
import glob
from openai import OpenAI

# 定义文件路径
csv_path = "/mnt/cache_share/MedTrinity-25M/Quilt/quilt_1M_lookup-001.csv"
images_dir = "/mnt/cache_share/MedTrinity-25M/Quilt/images/quilt_1m"
output_path = "/mnt/cache_share/MedTrinity-25M/Quilt/json"

# 读取CSV文件
df = pd.read_csv(csv_path)

# 提取 caption 和 image_path 列
captions = df['caption'].tolist()
image_paths = df['image_path'].tolist()

# 定义系统提示
system_prompt = """
I have a caption that provides partial information about a medical image. Your task is to first generate a complete and accurate description of the image based on the given caption. Keep in mind that the caption is incomplete and might not provide all necessary details, so infer missing information where possible. Then, based on the image, generate 1-2 valuable questions and their corresponding answers that are relevant to the image's content.

Caption: {caption}

Return the results in the following format:

Description:

Question:
1. 
2. 

Answer:
1.
2.

Don't generate any other information
"""

# 设置 API 密钥
os.environ["OPENAI_API_KEY"]=''

client = OpenAI()

# 将图像编码为 base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 处理每一行数据
def process_row(row, index):
    image_id = row['image_path']
    report = ' '.join(row['caption'].split()[:12000])
    # 输出文件路径
    output_file_path = os.path.join(output_path, f"{index}.txt")

    # 如果文件已经存在，则跳过处理
    if os.path.exists(output_file_path):
        return

    # 获取图像的 base64 编码
    base64_image = encode_image(os.path.join(images_dir, image_id))

    try:
        # 请求 GPT-4 API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt.format(caption=report),
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
            max_tokens=3000,
        )
    except Exception as e:
        print(f"An error occurred while processing row {index}: {str(e)}")
        return

    # 获取响应内容
    result = response.choices[0].message.content

    # 将结果保存为文本文件
    with open(output_file_path, 'w') as f:
        f.write(result)

    print(f"Processed row {index} and saved result to {output_file_path}")

# 使用并行处理处理所有行
with ThreadPoolExecutor() as executor:
    executor.map(lambda idx: process_row(df.iloc[idx], idx), range(len(df)))
