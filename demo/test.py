# 项目结构如下：
# gui/main_window.py
# gui/log_signal.py
# gui/image_display.py
# logic/config_manager.py
# logic/inference_runner.py
# utils/path_utils.py
# guisr.py

# ========== 文件: gui/log_signal.py ==========
from PyQt5.QtCore import QObject, pyqtSignal


class LogSignal(QObject):
    log_signal = pyqtSignal(str)


# ========== 文件: logic/config_manager.py ==========
import os
import yaml


class ConfigManager:
    CONFIG_FILE = "./userdata/config.yaml"

    @staticmethod
    def ensure_userdata_folder():
        userdata_folder = os.path.dirname(ConfigManager.CONFIG_FILE)
        if not os.path.exists(userdata_folder):
            os.makedirs(userdata_folder)

    @staticmethod
    def load():
        if os.path.exists(ConfigManager.CONFIG_FILE):
            with open(ConfigManager.CONFIG_FILE, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    @staticmethod
    def save(config):
        with open(ConfigManager.CONFIG_FILE, "w") as f:
            yaml.safe_dump(config, f)


# ========== 文件: utils/path_utils.py ==========
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


# ========== 文件: logic/inference_runner.py ==========
import threading
import multiprocessing
from inference_process import run_inference_process


class InferenceRunner:

    def __init__(self, gui):
        self.gui = gui

    def start(self, input_path, model_path, output_folder):
        if self.gui.video_radio.isChecked():
            self._start_process(input_path, model_path, output_folder)
        else:
            self._start_thread(input_path, model_path, output_folder)

    def _start_thread(self, input_path, model_path, output_folder):
        thread = threading.Thread(target=self.gui.run_inference, args=(input_path, model_path, output_folder))
        thread.start()

    def _start_process(self, input_path, model_path, output_folder):
        log_queue = multiprocessing.Queue()
        model = self.gui.model_selection_combo.currentText()
        process = multiprocessing.Process(
            target=run_inference_process, args=(input_path, model_path, output_folder, model, log_queue))
        process.start()

        threading.Thread(target=self.gui.update_log_queue, args=(log_queue, process), daemon=True).start()


# ========== 文件: gui/image_display.py ==========
from PyQt5.QtGui import QPixmap
from PIL import Image
import os


def display_images(gui, input_image_path, output_image_path):
    pix_in = QPixmap(input_image_path).scaled(gui.input_image_label.size())
    gui.input_image_label.setPixmap(pix_in)

    with Image.open(input_image_path) as img:
        w, h = img.size
    size = os.path.getsize(input_image_path) / 1024
    gui.input_image_info.setText(f"输入图片信息: {w}x{h}, {size:.2f} KB")

    pix_out = QPixmap(output_image_path).scaled(gui.output_image_label.size())
    gui.output_image_label.setPixmap(pix_out)

    with Image.open(output_image_path) as img:
        w, h = img.size
    size = os.path.getsize(output_image_path) / 1024
    gui.output_image_info.setText(f"分辨率提升后的图片信息: {w}x{h}, {size:.2f} KB")


# ========== 文件: guisr.py ==========
import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication
from gui.main_window import InferenceGUI

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    app = QApplication(sys.argv)
    window = InferenceGUI()
    window.show()
    sys.exit(app.exec_())
