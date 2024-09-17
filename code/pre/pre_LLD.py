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
os.environ["OPENAI_API_KEY"]=''

client = OpenAI()

import json

# 读取JSON文件
import json
from collections import defaultdict

# 读取JSON文件
with open('/mnt/cache_share/MedTrinity-25M/LLD-MMRI/labels/Annotation.json', 'r') as file:
    data = json.load(file)

# 初始化结果字典

result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# 遍历 Annotation_info 中的每个条目
for id_key, annotations in data["Annotation_info"].items():
    for annotation in annotations:
        studyUID = annotation.get("studyUID")
        seriesUID = annotation.get("seriesUID")
        
        # 遍历 lesion 中的每个条目
        for lesion_id, lesion_data in annotation["annotation"]["lesion"].items():
            for box in lesion_data["bbox"]["2D_box"]:
                result[id_key][studyUID][seriesUID].append(box["slice_idx"])


source_base_dir = '/mnt/cache_share/MedTrinity-25M/LLD-MMRI/images'
output_base_dir = '/mnt/cache_share/MedTrinity-25M/LLD-MMRI/images2'
import nibabel as nib

# 确保目标目录存在
os.makedirs(output_base_dir, exist_ok=True)

# 遍历每个 studyUID 和 seriesUID
for id_key, studies in result.items():
    for studyUID, series in studies.items():
        # 创建studyUID文件夹
        study_dir = os.path.join(output_base_dir, id_key, studyUID)
        os.makedirs(study_dir, exist_ok=True)
        
        for seriesUID, slice_idxs in series.items():
            # 构建源文件路径
            nii_file_path = os.path.join(source_base_dir, id_key, studyUID, f"{seriesUID}.nii")

            # 检查文件是否存在
            if os.path.exists(nii_file_path):
                # 读取 NIfTI 文件
                img = nib.load(nii_file_path)
                img_data = img.get_fdata()

                # 为每个 slice_idx 提取并保存图像
                for slice_idx in slice_idxs:
                    if 0 <= slice_idx < img_data.shape[2]:
                        slice_data = img_data[:, :, slice_idx]
                        
                        # 保存图片的文件名
                        output_file_path = os.path.join(id_key, study_dir, f"{seriesUID}_{slice_idx}.png")
                        
                        # 使用 matplotlib 保存图像为 PNG
                        import matplotlib.pyplot as plt
                        plt.imsave(output_file_path, slice_data, cmap='gray')
                        
                        print(f"Saved slice {slice_idx} of {seriesUID} in {output_file_path}")
                    else:
                        print(f"Slice index {slice_idx} out of bounds for {seriesUID}")
            else:
                print(f"NIfTI file {nii_file_path} not found.")

print("所有文件已处理完成")







