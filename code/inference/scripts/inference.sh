#!/bin/bash

# 设置模型路径
MODEL_PATH="/path/to/your/model"  # 请替换为实际的模型路径
IMAGE_PATH="/path/to/your/image.jpg"  # 请替换为实际的图片路径
QUESTION="What is shown in this image?"
CONV_MODE="llama_3"
TEMPERATURE=0

# 运行 Python 脚本
torchrun inference.py --model_path $MODEL_PATH --image_path $IMAGE_PATH --question "$QUESTION" --conv_mode $CONV_MODE --temperature $TEMPERATURE
