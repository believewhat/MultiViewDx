import os

# 输入文件和根目录路径
input_file = 'output_name.txt'  # 替换为你的实际文件路径
root_directory = '/home/data/38_Pubmed_cleaned/gpt4/'

# 构建文件夹名到文件名集合的字典
folder_files_dict = {}
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    if os.path.isdir(folder_path):
        # 获取该文件夹下所有的txt文件的名称（去掉扩展名）
        txt_files = set(f[:-4] for f in os.listdir(folder_path) if f.endswith('.txt'))
        folder_files_dict[folder_name] = txt_files

# 读取输入文件
with open(input_file, 'r') as file:
    lines = file.readlines()

# 用于保存找不到文件的记录
not_found = []

# 逐行处理
for line in lines:
    line = line.strip()
    json_file, txt_file = line.split()
    
    folder_name = os.path.splitext(json_file)[0]
    txt_filename = txt_file.split('/')[-1]  # 获取文件名部分

    # 判断文件是否存在于字典中
    if folder_name not in folder_files_dict or txt_filename not in folder_files_dict[folder_name]:
        not_found.append(line)

# 将未找到的记录保存回文件，覆盖原文件内容
with open(input_file, 'w') as file:
    for record in not_found:
        file.write(record + '\n')

print(f"Processing completed. Updated file saved as '{input_file}'.")
