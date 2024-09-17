#!/bin/bash

# 读取 JSON 文件
json_file="eurorad.json"

# 创建临时文件来保存讨论内容
temp_file="discussions_temp.txt"

index=0
parallel_jobs=10
pids=()


while IFS= read -r line; do
    # 提取索引和讨论内容
    
    # 调用 Python 脚本并传递索引和讨论内容，放到后台执行
    python3 deal_euro.py $line &
    
    # 获取进程 ID 并存储
    pids+=($!)

    # 每10个并行任务等待一次
    if [ ${#pids[@]} -ge $parallel_jobs ]; then
        # 等待所有后台任务完成
        for pid in "${pids[@]}"; do
            wait $pid
        done
        # 清空进程 ID 数组
        pids=()
    fi
done < "$temp_file"

# 等待剩余的所有后台任务完成
for pid in "${pids[@]}"; do
    wait $pid
done

