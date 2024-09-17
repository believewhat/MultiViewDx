import argparse
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from cambrian.model.language_model.cambrian_llama import CambrianLlamaForCausalLM

def main(model_path, weights_path):
    # 加载原始权重
    model = CambrianLlamaForCausalLM.from_pretrained(model_path)

    # 加载自定义模型权重
    merged_dict = {}

    # 加载 weights_rank 和 opt_rank 的权重到 merged_dict 中
    for prefix in ['weights_rank', 'opt_rank']:
        for i in range(4):
            part_path = f"{prefix}-0000000{i}-of-00000004-pytorch_model.bin"
            state_dict = torch.load(os.path.join(weights_path, part_path), map_location='cuda:0')
            
            # 合并加载的 state_dict 到 merged_dict 中
            for key, value in state_dict.items():
                if key in merged_dict:
                    merged_dict[key] = value
                else:
                    merged_dict[key] = value

    merged_dict = merged_dict['model']

    processed_dict = {}
    for key, value in merged_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        processed_dict[new_key] = value

    # 获取 model 的 state_dict
    model_state_dict = model.state_dict()

    # 将 processed_dict 和 model_state_dict 中的所有张量移动到 CPU 并转换为 bfloat16 数据类型
    for key in processed_dict:
        processed_dict[key] = processed_dict[key].cpu().to(torch.bfloat16)

    for key in model_state_dict:
        model_state_dict[key] = model_state_dict[key].cpu().to(torch.bfloat16)

    # 找出不同的键
    keys_in_processed_not_in_model = processed_dict.keys() - model_state_dict.keys()
    keys_in_model_not_in_processed = model_state_dict.keys() - processed_dict.keys()

    # 找出相同键但值不同的项
    keys_with_different_values = []
    for key in processed_dict.keys() & model_state_dict.keys():
        if not torch.equal(processed_dict[key], model_state_dict[key]):
            keys_with_different_values.append(key)

    # 输出不同的键
    if keys_in_processed_not_in_model:
        print("Keys in processed_dict but not in model_state_dict:")
        for key in keys_in_processed_not_in_model:
            print(key)

    if keys_in_model_not_in_processed:
        print("Keys in model_state_dict but not in processed_dict:")
        for key in keys_in_model_not_in_processed:
            print(key)

    # 输出相同键但值不同的键
    if keys_with_different_values:
        print("Keys with different values in processed_dict and model_state_dict:")
        for key in keys_with_different_values:
            print(key)
    else:
        print("All keys with the same name have the same values in both dicts.")

    for key in keys_with_different_values:
        model_state_dict[key] = processed_dict[key]
        print(f"Updated key {key} in model_state_dict.")

    # 将更新后的 model_state_dict 加载回 model 中
    model.load_state_dict(model_state_dict, strict=True)

    # 保存更新后的模型
    model.save_pretrained(weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update model weights and save.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the directory containing weights to load and where to save the updated model.")

    args = parser.parse_args()

    main(args.model_path, args.weights_path)
