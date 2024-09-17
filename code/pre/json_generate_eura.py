import os
import json
import re
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义路径
json_dir_implement2 = '/mnt/cache_share/EURORAD_implement2'
txt_dir_implement = '/mnt/cache_share/EURORAD_implement'
instruct_dir = '/mnt/cache_share/EURORAD_instruct'
image_base_path = "/mnt/cache_share/EURORAD/"

# 任务1: 读取txt文件并更新img_alt2
def update_img_alt2_from_txt(data, txt_file_content):
    for img in data.get('img', []):
        img_id = img['img_id']
        if img_id in txt_file_content:
            img['img_alt2'] = txt_file_content[img_id]
    return data

# 查找图像路径
def find_image_path(title, img_id, base_path=image_base_path):
    folder_path = os.path.join(base_path, title)
    search_pattern = os.path.join(folder_path, img_id + '.*')  # 使用通配符搜索所有格式
    matching_files = glob.glob(search_pattern)
    if matching_files:
        return matching_files[0]  # 返回找到的第一个匹配文件
    else:
        return None

# 任务2: 生成每个 image_id 的 IMAGING_FINDINGS，并添加 image_path
def generate_imaging_findings(clinical_history, images, title):
    imaging_findings = []
    for img in images:
        # 查找每个 image_id 的图像路径
        image_path = find_image_path(title, img['img_id'])
        # 为每个 image_id 生成一个单独的 finding
        human_text = f"Clinical History:\n{clinical_history}\nPlease generate the description of the following image:\n<image>"
        gpt_text = img['img_alt2']
        # 添加human的请求
        imaging_finding = []
        imaging_finding.append({"role": "human", "value": human_text})
        # 添加gpt的响应
        imaging_finding.append({"role": "gpt", "value": gpt_text})
        # 添加image_path
        imaging_finding.append({"image": image_path})
        imaging_findings.append({'conversations': imaging_finding})
    return imaging_findings

# 任务3: 生成 DISCUSSION
def generate_discussion(clinical_history, images, original_discussion):
    discussion = []
    user_input = f"Clinical History:\n{clinical_history}\n\nImage Descriptions:\n"
    for img in images:
        user_input += f"{img['img_id']}: {img['img_alt2']}\n"
    # Human部分包含请求生成讨论的指令
    user_input += "\nPlease generate a discussion based on the provided clinical history and image descriptions."
    discussion.append({"role": "human", "value": user_input.strip()})
    # GPT部分是原始数据里的discussion
    discussion.append({"role": "gpt", "value": original_discussion.strip()})
    return {'conversations': discussion}

# 任务4: 从questions_answers中提取对话
def split_questions_answers(qa_text, title, img_id):
    """
    处理问题和回答，确保只解析包含正确格式的 **Question** 和 **Answer**，并且处理图像路径。
    """
    conversations = []
    parts = qa_text.split("\n\n")  # 使用双换行符分隔问题和回答
    first_question = True  # 标记第一个问题

    # 查找图像路径
    image_path = find_image_path(title, img_id)

    for part in parts:
        part = part.strip()  # 移除前后空格

        # 使用正则表达式提取 Question 和 Answer
        question_match = re.search(r"\*\*Question.*?:\*\*\s*(.*)", part, re.IGNORECASE)
        answer_match = re.search(r"\*\*Answer.*?:\*\*\s*(.*)", part, re.IGNORECASE)

        if question_match and answer_match:
            question = question_match.group(1).strip()
            answer = answer_match.group(1).strip()

            # 如果是第一个问题，添加 <image> 标记
            if first_question:
                question += "\n<image>"
                first_question = False  # 之后的问题不再添加 <image>

            # 添加问题到 conversations 列表
            conversations.append({
                "from": "human",
                "value": question
            })

            # 添加回答到 conversations 列表
            conversations.append({
                "from": "gpt",
                "value": answer
            })

    # 返回生成的 conversations 以及图像路径
    return {"conversations": conversations, "image": image_path}

# 处理单个文件的函数
def process_single_file(json_file):
    json_file_name = json_file.split('.')[0]

    # Step 1: 读取 EURORAD_implement2 中的 JSON 文件
    with open(os.path.join(json_dir_implement2, json_file), 'r') as f:
        data = json.load(f)

    # Step 2: 读取对应的 txt 文件
    txt_file_path = os.path.join(txt_dir_implement, f"{json_file_name}.txt")
    if not os.path.exists(txt_file_path):
        print(f"Missing corresponding txt file for {json_file}")
        return None

    txt_file_content = {}
    with open(txt_file_path, 'r') as txt_f:
        for line in txt_f:
            if ':' not in line:
                continue
            try:
                img_id, description = line.split(": ", 1)
                description = description.strip()
                # 跳过没有内容的 img_id
                if not description:
                    print(f"Skipping empty description for img_id: {img_id.strip()}")
                    continue
                txt_file_content[img_id.strip()] = description
            except Exception as e:
                print(f"Error processing line in {txt_file_path}: {e}")

    # 更新 img_alt2
    data = update_img_alt2_from_txt(data, txt_file_content)

    # 移除 img 列表中没有内容的 img 对象
    images = [img for img in data.get('img', []) if img.get('img_alt2')]

    if not images:
        print(f"No valid images found in {json_file}")
        return None

    # 生成 IMAGING_FINDINGS
    clinical_history = data.get('CLINICAL_HISTORY', '')
    title = data.get('title', '')
    imaging_findings = generate_imaging_findings(clinical_history, images, title)

    # 生成 DISCUSSION
    original_discussion = data.get('DISCUSSION', 'No discussion available.')
    discussion = generate_discussion(clinical_history, images, original_discussion)

    # 查找 title 文件夹中的 JSON 文件并提取对话
    for img in images:
        img_id = img['img_id']
        img_json_file = os.path.join(instruct_dir, title, f"{img_id}.json")
        if not os.path.exists(img_json_file):
            print(f"Missing corresponding JSON file for image {img_id} in {title}")
            continue

        with open(img_json_file, 'r') as img_f:
            img_data = json.load(img_f)

        # 处理 questions_answers 并转换为对话
        qa_text = img_data.get('questions_answers', '')
        conversations = split_questions_answers(qa_text, title, img_id)

        return {
            "conversations": conversations['conversations'],
            "imaging_findings": imaging_findings,
            "discussion": discussion
        }

# 使用并行处理
def process_files_in_parallel():
    summary_data = []
    json_files = [f for f in os.listdir(json_dir_implement2) if f.endswith('.json')]
    for json_file in json_files:
        process_single_file(json_file)
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_file, json_file): json_file for json_file in json_files}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                summary_data.append(result)

    # 保存结果为eurad.json文件
    output_file = 'eurad.json'
    with open(output_file, 'w') as f_out:
        json.dump(summary_data, f_out, indent=4)

    print(f"Data has been saved to {output_file}")

# 执行并行处理
if __name__ == "__main__":
    process_files_in_parallel()
