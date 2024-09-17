import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor
import base64
from PIL import Image
from openai import OpenAI

# 设置 OpenAI API Key
os.environ["OPENAI_API_KEY"] = ''

client = OpenAI()

# 定义根目录路径
root_dir = '/mnt/cache_share/MedTrinity-25M/MAMA-MIA/images2'
output_base_dir = '/mnt/cache_share/MedTrinity-25M/MAMA-MIA/json'

# 创建输出目录（如果不存在）
os.makedirs(output_base_dir, exist_ok=True)

# 获取所有的图像文件路径，包括子目录
image_files = glob.glob(os.path.join(root_dir, '**', '*.*'), recursive=True)
image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

system_prompt = """
Analyze the provided MRI image and generate a detailed and professional medical report that describes only the abnormalities, significant features, or relevant observations directly seen in the image. Use precise medical terminology and maintain a formal tone. Do not include any introductory phrases, such as "The provided image reveals," or any concluding remarks.

Your second task is to generate 1-2 valuable questions and their corresponding answers that are relevant to the image's content.

Return the results in the following format:

Report:

Question:
1. 
2. 

Answer:
1.
2.

Don't generate any other information
"""

# 函数用于对图像进行Base64编码并生成描述和问题答案
def process_image(image_path):
    try:
        # 打开图像并进行Base64编码
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 构建 OpenAI API 请求
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
            max_tokens=3000,
        )

        # 提取生成的描述和问题答案
        output_content = response.choices[0].message.content

        # 获取相对于root_dir的相对路径，并更改文件扩展名为 .txt
        relative_path = os.path.relpath(image_path, root_dir)
        relative_path_without_ext = os.path.splitext(relative_path)[0]
        
        # 构建输出路径
        output_path = os.path.join(output_base_dir, relative_path_without_ext + '.txt')

        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 将生成的内容保存为文本文件
        with open(output_path, 'w') as json_file:
            json_file.write(output_content)
        
        print(f"Processed and saved: {output_path}")
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# 使用ThreadPoolExecutor进行并行处理
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_image, image_files)

    print("所有图像文件已处理完成。")
