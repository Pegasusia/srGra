import re


def check_path(path):
    # 使用正则表达式检查路径中是否包含中文字符
    if re.search(r'[\u4e00-\u9fff]', path):
        print("路径包含中文字符，请使用仅包含英文的路径。")
        return False
    else:
        print("路径有效，仅包含英文字符。")
        return True


if __name__ == "__main__":
    # 示例路径
    path = r'D:/Resource/Picture/y.png'

    # 调用函数检查路径
    check_path(path)
