import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from openai import OpenAI
import base64

# 设置 OpenAI API Key
os.environ["OPENAI_API_KEY"] = ''

client = OpenAI()

system_prompt = """
Based on the following discussion and the provided image, generate 10 questions and their corresponding answers. Each question should focus on the details or features visible in the image, and where possible, the answers must be explicitly found within the discussion. If the discussion is not available, create questions and answers solely based on the image content. Ensure the questions are formatted clearly, and the answers are directly related to either the discussion or the image.

Discussion:
{Discussion}

Image ID: {ImageID}
"""

root_path = "/mnt/cache_share/EURORAD_instruct/"
image_base_path = "/path/to/images"  # 图片的基础路径

# 函数用于对图片进行Base64编码
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 函数用于处理每个条目
def process_entry(entry):
    title = entry['title']
    discussion = entry.get('IMAGING_FINDINGS', '')
    images_name = []
    content = []

    for image in entry.get('img', []):
        img_id = image['img_id']
        img_alt2 = image['img_alt2']

        # 构造图片文件路径
        image_path_pattern = os.path.join(image_base_path, title, img_id + '.*')
        image_files = glob.glob(image_path_pattern)

        if not image_files:
            print(f"No image found for pattern: {image_path_pattern}")
            continue

        image_path = image_files[0]  # 取找到的第一个文件
        image_data = encode_image(image_path)

        images_name.append(img_id)
        content.append({
            "type": "text",
            "text": f"Image ID {img_id}:",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}",
            },
        })

        input_text = system_prompt.replace("{Discussion}", discussion).replace("{ImageID}", img_id)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            max_tokens=2048,
        )

        result = response.choices[0].message.content

        json_output = {
            "image_id": img_id,
            "questions_answers": result
        }

        # 保存每个图片的问答为 JSON 文件
        output_path = os.path.join(root_path, f"{img_id}.json")
        with open(output_path, 'w') as json_file:
            json.dump(json_output, json_file, indent=4)

        print(f"Processed {img_id} and saved to {output_path}")

if __name__ == '__main__':
    # 读取 JSON 数据
    with open('eurorad.json', 'r') as f:
        data = json.load(f)

    keys_to_keep = ['title', 'CLINICAL_HISTORY', 'IMAGING_FINDINGS', 'img', 'FINAL_DIAGNOSIS']

    # 提取需要的字段
    entries_to_process = [{key: entry[key] for key in keys_to_keep if key in entry} for entry in data]

    # 使用 ThreadPoolExecutor 进行并行处理
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_entry, entry): entry for entry in entries_to_process}

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"An error occurred: {exc}")

    print("所有文件已处理完成")
