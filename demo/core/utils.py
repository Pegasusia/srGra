import re
import os
from PyQt5.QtWidgets import QMessageBox


def validate_paths(*paths):
    """路径校验"""
    for path in paths:
        if re.search(r'[\u4e00-\u9fff]', path):
            return False
        if not os.path.exists(path):
            return False
    return True


def show_warning(title, message):
    """显示警告弹窗"""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.exec_()