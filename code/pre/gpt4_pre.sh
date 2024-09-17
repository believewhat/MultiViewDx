#!/bin/bash

# 读取 output_name.txt 文件行数
file="output_name.txt"
max_lines=14015665
i=0
parallel_jobs=40
pids=()

head -n $max_lines "$file" > temp_output_name.txt

# 循环读取每一行并行处理
while IFS= read -r line
do
    # 增加行计数器
    i=$((i + 1))

    # 提取 JSON 文件名和图像文件名
    json_filename=$(echo "$line" | awk '{print $1}')
    image_filename=$(echo "$line" | awk '{print $2}')
    
    # 构造保存文件的路径
    file_name=$(echo "$json_filename" | awk -F'.' '{print $1}')
    files_image=$(echo "$image_filename" | awk -F'/' '{print $2}')
    save_path="/home/data/38_Pubmed_cleaned/gpt4/${file_name}/${files_image}.txt"

    # 如果保存文件已存在，则跳过
    if [ -f "$save_path" ]; then
        continue
    fi
    
    # 将当前行写入临时文件
    echo "$line" >> temp_output_name.txt
    
    # 调用 Python 脚本并传递行内容，放到后台执行
    python3 deal_pmc.py $line &
    
    # 获取进程ID并存储
    pids+=($!)

    # 每10个并行任务等待一次
    if [ $((i % parallel_jobs)) -eq 0 ]; then
        # 等待所有后台任务完成
        for pid in "${pids[@]}"; do
            wait $pid
        done
        # 清空进程ID数组
        pids=()
    fi
done < temp_output_name.txt

# 等待剩余的所有后台任务完成
for pid in "${pids[@]}"; do
    wait $pid
done

# 删除临时文件
rm temp_output_name.txt
