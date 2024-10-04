import os
import sys
# 获取当前工作目录
current_directory = os.getcwd()
print("当前工作目录:", current_directory)

# 切换到新的目录，例如 "/path/to/new/directory"
new_directory = "/home/junda.wang/project/cambrian"
os.chdir(new_directory)
print("切换后的工作目录:", os.getcwd())
sys.path.append(current_directory)
from cambrian.train.train_fsdp import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
