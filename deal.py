import os

def generate_tree(root_dir, prefix=""):
    """递归生成目录树"""
    # 获取当前目录中的所有文件和子目录
    items = sorted(os.listdir(root_dir))
    tree_str = ""

    for index, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "

        tree_str += f"{prefix}{connector}{item}\n"

        # 如果是目录，递归生成子目录树
        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree_str += generate_tree(path, new_prefix)

    return tree_str

def main():
    # 指定要生成树状图的根目录
    root_directory = '.'  # 当前目录
    # 生成并打印目录树
    tree_output = generate_tree(root_directory)
    print(tree_output)

if __name__ == "__main__":
    main()

