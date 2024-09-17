import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 定义源目录和目标目录
source_dir = '/mnt/cache_share/MedTrinity-25M/MAMA-MIA/images'
output_dir = '/mnt/cache_share/MedTrinity-25M/MAMA-MIA/images2'

# 遍历源目录下的所有文件夹
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.nii'):
            nii_file_path = os.path.join(root, file)
            
            # 读取 NIfTI 文件
            img = nib.load(nii_file_path)
            img_data = img.get_fdata()
            
            # 获取切片数目
            num_slices = img_data.shape[2]
            
            # 计算步长并抽取 10 个均匀分布的切片索引
            step = max(num_slices // 10, 1)
            selected_slices = list(range(0, num_slices, step))[:10]
            
            # 创建输出文件夹
            relative_dir = os.path.relpath(root, source_dir)
            output_folder = os.path.join(output_dir, relative_dir)
            os.makedirs(output_folder, exist_ok=True)
            
            # 保存抽取的切片
            for i, slice_idx in enumerate(selected_slices):
                slice_data = img_data[:, :, slice_idx]
                
                # 构建输出文件名
                output_file_name = f"{os.path.splitext(file)[0]}_{slice_idx}.png"
                output_file_path = os.path.join(output_folder, output_file_name)
                
                # 保存切片为 PNG 图片
                plt.imsave(output_file_path, slice_data, cmap='gray')
                
                print(f"Saved {output_file_path}")

print("所有文件已处理完成")
