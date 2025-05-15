import re
import os
from PyQt5.QtWidgets import QMessageBox


def contains_chinese(path):
    return bool(re.search(r'[\u4e00-\u9fff]', path))


def show_warning_chinese():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("路径检查警告")
    msg.setText("路径包含中文字符，请使用仅包含英文的路径")
    msg.exec_()


def show_warning_path():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("路径检查警告")
    msg.setText("请检查路径是否存在")
    msg.exec_()


def validate_paths(paths):
    return all(os.path.exists(p) and not contains_chinese(p) for p in paths)
