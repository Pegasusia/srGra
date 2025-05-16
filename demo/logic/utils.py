import re
import os
import subprocess
from PyQt5.QtWidgets import QMessageBox

def get_model_info(input_type: str, scale: int):
    """根据输入类型和放大倍数返回模型类型和脚本路径"""
    if input_type == 'video':
        model_type = 'DUF'
        script = f"inference/inference_duf.py"
    else:
        if scale != 4:
            model_type = 'EDSR'
            script = f"inference/inference_edsr.py"
        else:
            model_type = 'ESRGAN'
            script = f"inference/inference_esrgan.py"
    return model_type, script


def check_file_type(file_path):
    """检查文件类型"""
    # 获取文件扩展名
    _, ext = os.path.splitext(file_path)

    # 定义图片和视频的扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv']

    if ext.lower() in image_extensions:
        return 'image'
    elif ext.lower() in video_extensions:
        return 'video'
    else:
        return 'folder'

def open_output_folder(folder_path):
    """自动打开输出文件夹"""
    if os.name == "nt":  # Windows
        os.startfile(folder_path)
    elif os.name == "posix":  # macOS 或 Linux
        subprocess.Popen(["open", folder_path])
    else:
        raise OSError("Unsupported operating system")


def open_output_file(file_path):
    """打开文件所在的文件夹，并选中该文件"""
    if os.name == "nt":  # Windows
        subprocess.run(["explorer", "/select,", os.path.normpath(file_path)])
    elif sys.platform == "darwin":  # macOS
        subprocess.run(["open", "-R", file_path])
    elif sys.platform.startswith("linux"):
        # Linux 不支持选中文件，只能打开目录
        folder = os.path.dirname(file_path)
        subprocess.run(["xdg-open", folder])
    else:
        raise OSError("Unsupported operating system")

def contains_chinese(path):
    return bool(re.search(r'[\u4e00-\u9fff]', path))


def show_warning_chinese():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("路径检查警告")
    msg.setText("路径包含中文字符，请使用仅包含英文的路径")
    msg.exec_()


def show_warning_file():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("文件类型警告")
    msg.setText("输入文件和选择的文件类型不匹配")
    msg.exec_()

def show_warning_path():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("路径检查警告")
    msg.setText("请检查路径是否存在")
    msg.exec_()


def validate_paths(paths):
    return all(os.path.exists(p) and not contains_chinese(p) for p in paths)
