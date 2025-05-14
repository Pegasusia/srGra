import sys
import os
import threading
import subprocess
import yaml  # 用于处理 YAML 文件
import re  # 用于正则表达式匹配

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, QTextEdit)
from PyQt5.QtWidgets import (QWidget, QFileDialog, QLabel, QLineEdit,QComboBox)
from PyQt5.QtWidgets import ( QHBoxLayout, QRadioButton, QButtonGroup, QAction, QMessageBox)
from PyQt5.QtCore import pyqtSignal, QObject, Qt
from PyQt5.QtGui import QPixmap

from PIL import Image  # 添加导入，用于读取图像的真实分辨率
import multiprocessing  # 用于多进程处理

from inference_process import run_inference_process  # 导入推理函数
from display_img import ImageComparisonWindow  # 导入图像对比窗口类

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
            self.log_message("INFO", f"创建文件夹: {userdata_folder}")

    def init_ui(self):
        self.setWindowTitle("超分辨率重建")
        self.setGeometry(100, 100, 1400, 900)  # 调整窗口大小以容纳图像显示

        # 主布局
        layout = QVBoxLayout()

        # 输入类型选择
        input_type_layout = QHBoxLayout()
        self.input_type_label = QLabel("选择输入类型:")
        self.file_radio = QRadioButton("单图片")
        self.folder_radio = QRadioButton("文件夹")
        self.video_radio = QRadioButton("视频")
        self.file_radio.setChecked(True)  # 默认选择single文件
        self.input_type_group = QButtonGroup(self)
        self.input_type_group.addButton(self.file_radio)
        self.input_type_group.addButton(self.folder_radio)
        self.input_type_group.addButton(self.video_radio)
        input_type_layout.addWidget(self.input_type_label)
        input_type_layout.addWidget(self.file_radio)
        input_type_layout.addWidget(self.folder_radio)
        input_type_layout.addWidget(self.video_radio)
        layout.addLayout(input_type_layout)

        # 添加模型选择下拉列表
        model_selection_layout = QHBoxLayout()
        self.model_selection_label = QLabel("选择神经网络:")
        self.model_selection_combo = QComboBox(self)
        self.model_selection_combo.addItems(["ESRGAN", "EDSR", "MSRResNet", "BasicVSR"])  # 添加可选模型
        self.model_selection_combo.currentIndexChanged.connect(self.save_config)  # 绑定值更改事件
        model_selection_layout.addWidget(self.model_selection_label)
        model_selection_layout.addWidget(self.model_selection_combo)
        layout.addLayout(model_selection_layout)

        # 输入路径选择
        input_layout = QHBoxLayout()
        self.input_label = QLabel("输入路径:")
        self.input_path = QLineEdit(self)
        self.input_button = QPushButton("打开文件夹", self)
        self.input_button.clicked.connect(self.select_input_path)
        # self.input_button.clicked.connect(lambda: self.select_input_path(self.input_path))
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(self.input_button)
        layout.addLayout(input_layout)

        # 模型路径选择
        model_layout = QHBoxLayout()
        self.model_label = QLabel("模型路径:")
        self.model_path = QLineEdit(self)
        self.model_button = QPushButton("打开文件夹", self)
        self.model_button.clicked.connect(self.select_model_file)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.model_button)
        layout.addLayout(model_layout)

        # 输出文件夹选择
        output_layout = QHBoxLayout()
        self.output_label = QLabel("输出路径:")
        self.output_path = QLineEdit(self)
        self.output_button = QPushButton("打开文件夹", self)
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
        self.input_image_label = QLabel("输入图片")
        self.input_image_label.setFixedSize(600, 400)  # 设置固定大小
        self.input_image_label.setStyleSheet("border: 1px solid black;")  # 添加边框
        self.output_image_label = QLabel("生成图片")
        self.output_image_label.setFixedSize(600, 400)  # 设置固定大小
        self.output_image_label.setStyleSheet("border: 1px solid black;")  # 添加边框
        image_display_layout.addWidget(self.input_image_label)
        image_display_layout.addWidget(self.output_image_label)
        layout.addLayout(image_display_layout)

        # 图像信息显示
        image_info_layout = QHBoxLayout()
        self.input_image_info = QLabel("输入图片信息: ")
        self.output_image_info = QLabel("生成图片信息: ")
        image_info_layout.addWidget(self.input_image_info)
        image_info_layout.addWidget(self.output_image_info)
        layout.addLayout(image_info_layout)

        # 开始推理按钮
        self.start_button = QPushButton("运行程序(Enter)", self)
        self.start_button.clicked.connect(self.start_inference)
        layout.addWidget(self.start_button)

        # 停止推理按钮
        self.stop_button = QPushButton("取消运行", self)
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
                    # 加载下拉列表选项
                    selected_model = config.get("selected_model", "ESRGAN")
                    index = self.model_selection_combo.findText(selected_model)
                    if index != -1:
                        self.model_selection_combo.setCurrentIndex(index)
                    self.log_message("INFO", f"加载配置文件 {self.CONFIG_FILE}")

            except Exception as e:
                self.log_message("ERROR", f"无法加载配置文件: {str(e)}")
        else:
            self.log_message("ERROR", f"没有配置文件 {self.CONFIG_FILE}")

    def save_config(self):
        """保存配置到 YAML 文件"""
        config = {
            "input_path": self.input_path.text(),
            "model_path": self.model_path.text(),
            "output_folder": self.output_path.text(),
            "selected_model": self.model_selection_combo.currentText()
        }
        try:
            with open(self.CONFIG_FILE, "w") as f:
                yaml.safe_dump(config, f)
                self.log_message("INFO", f"保存配置文件到 {self.CONFIG_FILE}")
        except Exception as e:
            self.log_message("ERROR", f"保存配置文件发生错误: {str(e)}")

    def read_config(self):
        # 读取YAML配置文件并返回数据
        try:
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            return config_data
        except Exception as e:
            print(f"读取配置文件时出错: {e}")
            return {}

    def check_path(self, config_data):
        # 获取路径信息
        paths = [
            config_data.get("input_path", ""),
            config_data.get("model_path", ""),
            config_data.get("output_folder", "")
        ]

        # 使用正则表达式检查路径是否包含中文字符
        for path in paths:
            if re.search(r'[\u4e00-\u9fff]', path):
                return False
        return True

    def show_warning_chinese(self):
        # 弹出警告框，提示用户路径包含中文字符
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("路径检查警告")
        msg.setText("路径包含中文字符，请使用仅包含英文的路径")
        msg.exec_()

    def show_warning_path(self):
        # 无效路径
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("路径检查警告")
        msg.setText("请检查路径是否存在")
        msg.exec_()

    def select_input_path(self):
        if self.file_radio.isChecked():
            # 选择单个文件

            file, _ = QFileDialog.getOpenFileName(self, "选择输入文件", self.input_path.text(),
                                                  "Image Files (*.png *.jpg *.jpeg)")
            if file:
                self.input_path.setText(file)
        elif self.folder_radio.isChecked():
            # 选择文件夹
            folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹", self.input_path.text())
            if folder:
                self.input_path.setText(folder)

        elif self.video_radio.isChecked():
            # 选择视频文件
            file, _ = QFileDialog.getOpenFileName(self, "选择输入视频", self.input_path.text(),
                                                "Video Files (*.mp4 *.avi *.mov *.mkv)")
            if file:
                self.input_path.setText(file)

        self.save_config()  # 保存配置

    def select_model_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择模型", self.model_path.text(), "PyTorch Model Files (*.pth)")
        if file:
            self.model_path.setText(file)
            self.save_config()  # 保存配置

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.output_path.text())
        if folder:
            self.output_path.setText(folder)
            self.save_config()  # 保存配置

    def start_inference(self):
        """开始推理"""

        # 读取配置文件
        config_data = self.read_config()

        # 检查路径
        if self.check_path(config_data):
            # 在路径有效的情况下，继续推理（假设这里有进一步的代码）
            print("英文路径...")
        else:
            # 弹出警告框，路径包含中文字符
            self.show_warning_chinese()
            return

        # 清空日志窗口
        self.log_text.clear()

        self.log_message("INFO", "程序开始运行...")

        # 获取输入、模型和输出路径
        input_path = self.input_path.text()
        model_path = self.model_path.text()
        output_folder = self.output_path.text()

        # 检查路径是否有效
        if self.file_radio.isChecked() and not os.path.isfile(input_path):
            self.log_message("ERROR", "无效的输入路径.")
            self.show_warning_path()
            return
        if self.folder_radio.isChecked() and not os.path.isdir(input_path):
            self.log_message("ERROR", "无效的输入文件夹.")
            self.show_warning_path()
            return
        if self.video_radio.isChecked() and not os.path.isfile(input_path):
            self.log_message("ERROR", "无效的视频文件.")
            self.show_warning_path()
            return
        if not os.path.isfile(model_path):
            self.log_message("ERROR", "无效的模型路径.")
            self.show_warning_path()
            return
        if not os.path.isdir(output_folder):
            self.log_message("ERROR", "无效的输出路径.")
            self.show_warning_path()
            return

        # 禁用按钮以防止重复点击
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        if self.video_radio.isChecked():
            # 视频模式，使用多进程
            print("视频模式，使用多进程")
            self.start_multi_process(input_path, model_path, output_folder)
        else:
            # 图片/文件夹图片 使用多线程
            print("图片/文件夹图片 使用多线程")
            self.start_multi_thread(input_path, model_path, output_folder)

    def start_multi_thread(self, input_path, model_path, output_folder):
        """启动多线程推理"""
        inference_thread = threading.Thread(target=self.run_inference, args=(input_path, model_path, output_folder))
        inference_thread.start()


    def start_multi_process(self, input_path, model_path, output_folder):
        """启动多进程推理"""
        selected_model = self.model_selection_combo.currentText()
        log_queue = multiprocessing.Queue()  # 创建日志队列

        # 创建进程并传递参数
        multi_process = multiprocessing.Process(
            target=run_inference_process, args=(input_path, model_path, output_folder, selected_model, log_queue))
        multi_process.start()
        # multi_process.join()  # 等待进程结束

        threading.Thread(target=self.update_log_queue, args=(log_queue, multi_process), daemon=True).start()  # 启动线程读取日志队列


    def update_log_queue(self, log_queue, process):
        """从日志队列中读取消息并更新到日志面板"""
        while process.is_alive() or not log_queue.empty():
            try:
                message = log_queue.get(timeout=0.1)  # 从队列中获取消息
                self.log_message("INFO", message)  # 更新到日志面板
            except Exception:
                pass  # 如果队列为空，继续循环

        # 进程结束后，重置按钮状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def stop_inference(self):
        """停止推理"""
        if self.process:
            self.process.terminate()  # 终止推理进程
            self.log_message("INFO", "用户终止运行.\n")
            self.process = None

        # 重置按钮状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


    def run_inference(self, input_path, model_path, output_folder):
        # 使用 subprocess 调用 inference_esrgan.py 脚本
        input_image_path = None  # 初始化变量
        output_image_path = None  # 初始化变量

        # 根据用户选择的神经网络模型，选择对应的推理脚本
        selected_model = self.model_selection_combo.currentText()
        if selected_model == "ESRGAN":
            script_name = "inference/inference_esrgan.py"
            output_suffix = "_ESRGAN.png"
        elif selected_model == "EDSR":
            script_name = "inference/inference_edsr.py"
            output_suffix = "_EDSR.png"
        # elif selected_model == "MSRResNet":
        #     script_name = "inference/inference_msrresnet.py"
        #     output_suffix = "_MSRResNet.png"
        elif selected_model == "BasicVSR":
            script_name = "inference/inference_basicvsr.py"
            output_suffix = "_BasicVSR.png"  # 视频输出文件
        else:
            self.log_message("ERROR", "未知的神经网络模型选择.")
            return

        if self.file_radio.isChecked():
            # 单个文件模式
            command = [
                "python", script_name, "--input_file", input_path, "--model_path", model_path,
                "--output", output_folder
            ]
            input_image_path = input_path
            output_image_path = os.path.join(output_folder,
                                             f"{os.path.splitext(os.path.basename(input_path))[0]}{output_suffix}").replace(os.sep, '/')
        elif self.folder_radio.isChecked():
            # 文件夹模式
            command = [
                "python", script_name, "--input", input_path, "--model_path", model_path,
                "--output", output_folder
            ]
            # 获取文件夹中第一个输入文件和对应的输出文件
            input_files = sorted(os.listdir(input_path))
            if input_files:
                first_input_image = input_files[0]
                input_image_path = os.path.join(input_path, first_input_image)
                first_output_image = f"{os.path.splitext(first_input_image)[0]}{output_suffix}"
                output_image_path = os.path.join(output_folder, first_output_image).replace(os.sep, '/')

        elif self.video_radio.isChecked():
            # 视频模式
            command = [
                "python", script_name, "--input_path", input_path, "--model_path", model_path,
                "--save_path", output_folder
            ]
            input_image_path = input_path
            output_image_path = os.path.join(output_folder,
                                            f"{os.path.splitext(os.path.basename(input_path))[0]}{output_suffix}").replace(os.sep, '/')


        try:
            # 检查是否正确设置了输入和输出路径
            if not input_image_path or not output_image_path:
                self.log_message("ERROR", "无有效路径.")
                return

            # 使用 subprocess 捕获输出并实时更新日志
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
                universal_newlines=True,
                env={
                    **os.environ, "PYTHONUNBUFFERED": "1"
                }  # 禁用缓冲
            )

            # 实时读取 stdout 和 stderr
            for line in self.process.stdout:
                self.log_message("INFO", line.strip())
            for line in self.process.stderr:
                self.log_message("ERROR", f"ERROR: {line.strip()}")

            self.process.wait()  # 等待进程结束
            if self.process.returncode == 0:
                self.log_message("SUCCESS", "图片分辨率重建成功\n")
                self.display_images(input_image_path, output_image_path)
                self.open_output_folder(output_folder)
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
            # input_data = list(img.getdata())  # 获取输入图像的像素数据
        input_size = os.path.getsize(input_image_path) / 1024  # 文件大小 (KB)
        self.input_image_info.setText(f"输入图片信息: {input_width}x{input_height}, {input_size:.2f} KB")

        # 显示生成图像
        output_pixmap = QPixmap(output_image_path)
        output_pixmap = output_pixmap.scaled(self.output_image_label.width(), self.output_image_label.height())
        self.output_image_label.setPixmap(output_pixmap)

        # 获取生成图像的真实分辨率
        with Image.open(output_image_path) as img:
            output_width, output_height = img.size
            # output_data = list(img.getdata())  # 获取输出图像的像素数据
        output_size = os.path.getsize(output_image_path) / 1024  # 文件大小 (KB)
        self.output_image_info.setText(f"分辨率提升后的图片信息: {output_width}x{output_height}, {output_size:.2f} KB")

    # def display_images(self, input_image_path, output_image_path):
    #     """弹出独立窗口显示输入图像和生成图像的对比信息"""
    #     comparison_window = ImageComparisonWindow(input_image_path, output_image_path, self)
    #     comparison_window.exec_()  # 模态显示窗口


    def open_output_folder(self, folder_path):
        """自动打开输出文件夹"""
        if os.name == "nt":  # Windows
            os.startfile(folder_path)
        elif os.name == "posix":  # macOS 或 Linux
            subprocess.Popen(["open", folder_path])
        else:
            self.log_message("ERROR", "无法打开输出文件夹，请手动检查路径。")

    def update_log(self, message):
        # 更新日志窗口
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()  # 确保滚动到最新日志


# 主程序入口
if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method("spawn")
    app = QApplication(sys.argv)
    gui = InferenceGUI()
    gui.show()
    sys.exit(app.exec_())