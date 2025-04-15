import sys
import os
import threading
import subprocess
import yaml  # 用于处理 YAML 文件
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QTextEdit, QWidget, QFileDialog, QLabel, QLineEdit, QHBoxLayout, QRadioButton, QButtonGroup, QAction
)
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QPixmap

from PIL import Image  # 添加导入，用于读取图像的真实分辨率


# 信号类，用于线程间通信
class LogSignal(QObject):
    log_signal = pyqtSignal(str)


# 主窗口类
class InferenceGUI(QMainWindow):
    CONFIG_FILE = "./userdata/config.yaml"  # 配置文件路径

    def __init__(self):
        super().__init__()
        self.init_ui()

        # 创建日志信号
        self.log_signal = LogSignal()
        self.log_signal.log_signal.connect(self.update_log)

        # 确保 userdata 文件夹存在
        self.ensure_userdata_folder()

        # 加载默认配置
        self.load_config()

        # 推理进程
        self.process = None

        # 设置快捷键
        self.create_shortcuts()


    def keyPressEvent(self, event):
        # 判断是否按下 ESC 键
        if event.key() == 16777216:  # ESC 的键值
            QApplication.quit()  # 退出程序
            # 创建功能的快捷键

    def create_shortcuts(self):
        # 设置快捷键
        enter_start = QAction('enter', self)
        enter_start.setShortcut(Qt.Key_Return)
        enter_start.triggered.connect(self.start_inference)
        self.addAction(enter_start)  # 添加到窗口


    def log_message(self, level, message):
        """统一日志输出格式"""
        formatted_message = f"[{level}] {message}"
        self.log_signal.log_signal.emit(formatted_message)

    def ensure_userdata_folder(self):
        """确保 userdata 文件夹存在"""
        userdata_folder = os.path.dirname(self.CONFIG_FILE)
        if not os.path.exists(userdata_folder):
            os.makedirs(userdata_folder)
            self.log_message("INFO", f"Created folder: {userdata_folder}")

    def init_ui(self):
        self.setWindowTitle("ESRGAN Demo")
        self.setGeometry(100, 100, 1000, 800)  # 调整窗口大小以容纳图像显示

        # 主布局
        layout = QVBoxLayout()

        # 输入类型选择
        input_type_layout = QHBoxLayout()
        self.input_type_label = QLabel("Input Type:")
        self.file_radio = QRadioButton("Single File")
        self.folder_radio = QRadioButton("Folder")
        self.file_radio.setChecked(True)  # 默认选择single文件
        self.input_type_group = QButtonGroup(self)
        self.input_type_group.addButton(self.file_radio)
        self.input_type_group.addButton(self.folder_radio)
        input_type_layout.addWidget(self.input_type_label)
        input_type_layout.addWidget(self.file_radio)
        input_type_layout.addWidget(self.folder_radio)
        layout.addLayout(input_type_layout)

        # 输入路径选择
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Input Path:")
        self.input_path = QLineEdit(self)
        self.input_button = QPushButton("Browse", self)
        self.input_button.clicked.connect(self.select_input_path)
        # self.input_button.clicked.connect(lambda: self.select_input_path(self.input_path))
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(self.input_button)
        layout.addLayout(input_layout)

        # 模型路径选择
        model_layout = QHBoxLayout()
        self.model_label = QLabel("Model Path:")
        self.model_path = QLineEdit(self)
        self.model_button = QPushButton("Browse", self)
        self.model_button.clicked.connect(self.select_model_file)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.model_button)
        layout.addLayout(model_layout)

        # 输出文件夹选择
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Folder:")
        self.output_path = QLineEdit(self)
        self.output_button = QPushButton("Browse", self)
        self.output_button.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_button)
        layout.addLayout(output_layout)

        # 日志显示窗口
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # 图像显示窗口
        image_display_layout = QHBoxLayout()
        self.input_image_label = QLabel("Input Image")
        self.input_image_label.setFixedSize(400, 400)  # 设置固定大小
        self.input_image_label.setStyleSheet("border: 1px solid black;")  # 添加边框
        self.output_image_label = QLabel("Output Image")
        self.output_image_label.setFixedSize(400, 400)  # 设置固定大小
        self.output_image_label.setStyleSheet("border: 1px solid black;")  # 添加边框
        image_display_layout.addWidget(self.input_image_label)
        image_display_layout.addWidget(self.output_image_label)
        layout.addLayout(image_display_layout)

        # 图像信息显示
        image_info_layout = QHBoxLayout()
        self.input_image_info = QLabel("Input Image Info: ")
        self.output_image_info = QLabel("Output Image Info: ")
        image_info_layout.addWidget(self.input_image_info)
        image_info_layout.addWidget(self.output_image_info)
        layout.addLayout(image_info_layout)

        # 开始推理按钮
        self.start_button = QPushButton("Start Inference", self)
        self.start_button.clicked.connect(self.start_inference)
        layout.addWidget(self.start_button)

        # 停止推理按钮
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)  # 初始状态为禁用
        layout.addWidget(self.stop_button)

        # 设置中心窗口
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_config(self):
        """加载 YAML 配置文件"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    config = yaml.safe_load(f)
                    self.input_path.setText(config.get("input_path", ""))
                    self.model_path.setText(config.get("model_path", ""))
                    self.output_path.setText(config.get("output_folder", ""))
                    self.log_message("INFO", f"Loaded configuration from {self.CONFIG_FILE}")
            except Exception as e:
                self.log_message("ERROR", f"Error loading configuration: {str(e)}")
        else:
            self.log_message("ERROR", f"No configuration file found at {self.CONFIG_FILE}")

    def save_config(self):
        """保存配置到 YAML 文件"""
        config = {
            "input_path": self.input_path.text(),
            "model_path": self.model_path.text(),
            "output_folder": self.output_path.text()
        }
        try:
            with open(self.CONFIG_FILE, "w") as f:
                yaml.safe_dump(config, f)
                self.log_message("INFO", f"Saved configuration to {self.CONFIG_FILE}")
        except Exception as e:
            self.log_message("ERROR", f"Error saving configuration: {str(e)}")

    def select_input_path(self):
        if self.file_radio.isChecked():
            # 选择单个文件
            # file, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "Image Files (*.png *.jpg *.jpeg)")

            file, _ = QFileDialog.getOpenFileName(self, "Select Input File", self.input_path.text(), "Image Files (*.png *.jpg *.jpeg)")
            if file:
                self.input_path.setText(file)
        elif self.folder_radio.isChecked():
            # 选择文件夹
            folder = QFileDialog.getExistingDirectory(self, "Select Input Folder", self.input_path.text())
            if folder:
                self.input_path.setText(folder)
        self.save_config()  # 保存配置

    def select_model_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Model File", self.model_path.text(),"PyTorch Model Files (*.pth)")
        if file:
            self.model_path.setText(file)
            self.save_config()  # 保存配置

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_path.text())
        if folder:
            self.output_path.setText(folder)
            self.save_config()  # 保存配置

    def start_inference(self):
        """开始推理"""

        # 清空日志窗口
        self.log_text.clear()

        self.log_message("INFO", "Starting inference...")

        # 获取输入、模型和输出路径
        input_path = self.input_path.text()
        model_path = self.model_path.text()
        output_folder = self.output_path.text()

        # 检查路径是否有效
        if self.file_radio.isChecked() and not os.path.isfile(input_path):
            self.log_message("ERROR", "Error: Invalid input file.")
            return
        if self.folder_radio.isChecked() and not os.path.isdir(input_path):
            self.log_message("ERROR", "Error: Invalid input folder.")
            return
        if not os.path.isfile(model_path):
            self.log_message("ERROR", "Error: Invalid model file.")
            return
        if not os.path.isdir(output_folder):
            self.log_message("ERROR", "Error: Invalid output folder.")
            return

        # 禁用按钮以防止重复点击
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # 启动推理线程
        inference_thread = threading.Thread(target=self.run_inference, args=(input_path, model_path, output_folder))
        inference_thread.start()

    def stop_inference(self):
        """停止推理"""
        if self.process:
            self.process.terminate()  # 终止推理进程
            self.log_message("INFO", "User stopped.\n")
            self.process = None

        # 重置按钮状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def run_inference(self, input_path, model_path, output_folder):
        # 使用 subprocess 调用 inference_esrgan.py 脚本
        input_image_path = None  # 初始化变量
        output_image_path = None  # 初始化变量

        if self.file_radio.isChecked():
            # 单个文件模式
            command = [
                "python", "inference/inference_esrgan.py",
                "--input_file", input_path,
                "--model_path", model_path,
                "--output", output_folder
            ]
            input_image_path = input_path
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_path))[0]}_ESRGAN.png")
        elif self.folder_radio.isChecked():
            # 文件夹模式
            command = [
                "python", "inference/inference_esrgan.py",
                "--input", input_path,
                "--model_path", model_path,
                "--output", output_folder
            ]
            # 获取文件夹中第一个输入文件和对应的输出文件
            input_files = sorted(os.listdir(input_path))
            if input_files:
                first_input_image = input_files[0]
                input_image_path = os.path.join(input_path, first_input_image)
                first_output_image = f"{os.path.splitext(first_input_image)[0]}_ESRGAN.png"
                output_image_path = os.path.join(output_folder, first_output_image)

        try:
            # 检查是否正确设置了输入和输出路径
            if not input_image_path or not output_image_path:
                self.log_message("ERROR", "Error: No valid input or output image found.")
                return

            # 使用 subprocess 捕获输出并实时更新日志
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
                universal_newlines=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}  # 禁用缓冲
            )

            # 实时读取 stdout 和 stderr
            for line in self.process.stdout:
                self.log_message("INFO", line.strip())
            for line in self.process.stderr:
                self.log_message("ERROR", f"ERROR: {line.strip()}")

            self.process.wait()  # 等待进程结束
            if self.process.returncode == 0:
                self.log_message("SUCCESS", "Inference Completed Successfully!\n")
                self.display_images(input_image_path, output_image_path)
            else:
                self.log_message("ERROR", f"Failed with return code {self.process.returncode}")

        except Exception as e:
            if self.process is not None:
                self.log_message("ERROR", f"Error: {str(e)}")

        finally:
            # 重置按钮状态
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.process = None



    def display_images(self, input_image_path, output_image_path):
        """显示输入图像和生成图像，并显示图像信息"""
        # 显示输入图像
        input_pixmap = QPixmap(input_image_path)
        input_pixmap = input_pixmap.scaled(self.input_image_label.width(), self.input_image_label.height())
        self.input_image_label.setPixmap(input_pixmap)

        # 获取输入图像的真实分辨率
        with Image.open(input_image_path) as img:
            input_width, input_height = img.size
        input_size = os.path.getsize(input_image_path) / 1024  # 文件大小 (KB)
        self.input_image_info.setText(f"Input Image Info: {input_width}x{input_height}, {input_size:.2f} KB")

        # 显示生成图像
        output_pixmap = QPixmap(output_image_path)
        output_pixmap = output_pixmap.scaled(self.output_image_label.width(), self.output_image_label.height())
        self.output_image_label.setPixmap(output_pixmap)

        # 获取生成图像的真实分辨率
        with Image.open(output_image_path) as img:
            output_width, output_height = img.size
        output_size = os.path.getsize(output_image_path) / 1024  # 文件大小 (KB)
        self.output_image_info.setText(f"Output Image Info: {output_width}x{output_height}, {output_size:.2f} KB")

    def update_log(self, message):
        # 更新日志窗口
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()  # 确保滚动到最新日志


# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = InferenceGUI()
    gui.show()
    sys.exit(app.exec_())