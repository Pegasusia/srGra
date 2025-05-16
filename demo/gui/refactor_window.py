import os
import subprocess
import threading
from tkinter import N
from PyQt5.QtWidgets import (QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from gui.log_signal import LogSignal
from logic.config_manager import ConfigManager
from logic.utils import (contains_chinese, show_warning_chinese, show_warning_path, get_model_info, check_file_type,
                         show_warning_file, open_output_folder)

# from gui.image_display import display_images
from gui.image_display import display_images

class InferenceGUI(QMainWindow):
    """图像超分辨率重建系统主窗口类"""

    display_signal = pyqtSignal(str, str)  # 参数：input_image_path, output_image_path


    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像超分辨率重建系统 - lihao")
        self.setGeometry(150, 150, 1500, 900)
        self.setStyleSheet("background-color: #f5f7fa;")

        self.log_signal = LogSignal()
        self.log_signal.log_signal.connect(self.update_log)

        ConfigManager.ensure_userdata_folder()

        self.init_ui()
        self.load_config()

        self.display_signal.connect(self.display_comparison_dialog)


    def init_ui(self):
        """初始化UI组件"""
        self.process = None

        # 设置样式
        self.setStyleSheet("""
            QLabel { font-size: 20px; color: #333; font-weight: bold;}
            QPushButton {
                font-size: 20px;
                padding: 10px;
                background-color: #74A5DE;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #357ab8; }
            QRadioButton { font-size: 22px; padding: 6px; }
            QLineEdit { font-size: 20px; padding: 6px; }
        """)

        # 设置布局
        layout = QVBoxLayout()

        banner = QLabel("超分辨率重建系统 重庆工商大学 2021级 智能科学与技术 李昊")
        banner.setStyleSheet("font-size: 28px; font-weight: bold; color: #1a1a1a; margin-bottom: 20px;")
        banner.setAlignment(Qt.AlignCenter)
        layout.addWidget(banner)

        # 选择输入类型
        self.file_radio = QRadioButton("单图片")
        self.folder_radio = QRadioButton("文件夹")
        self.video_radio = QRadioButton("视频")
        self.file_radio.setChecked(True)

        self.file_radio.setStyleSheet("font-size: 22px;")
        self.folder_radio.setStyleSheet("font-size: 22px;")
        self.video_radio.setStyleSheet("font-size: 22px;")

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(QLabel("选择输入类型:"))
        radio_layout.addWidget(self.file_radio)
        radio_layout.addWidget(self.folder_radio)
        radio_layout.addWidget(self.video_radio)
        layout.addLayout(radio_layout)

        # 选择放大倍数 2 3 4
        self.scale_group = QButtonGroup()
        self.scale_buttons = []
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("选择放大倍数:"))
        for scale in ["2", "3", "4"]:
            btn = QRadioButton(f"{scale}×")
            if scale == "2":
                btn.setChecked(True)
            self.scale_buttons.append(btn)
            self.scale_group.addButton(btn)
            scale_layout.addWidget(btn)
        layout.addLayout(scale_layout)

        # 输入输出路径选择
        self.input_path = QLineEdit()
        self.input_button = QPushButton("选择输入")
        self.input_button.clicked.connect(self.select_input_path)
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入路径:"))
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(self.input_button)
        layout.addLayout(input_layout)

        self.output_path = QLineEdit()
        self.output_button = QPushButton("选择输出")
        self.output_button.clicked.connect(self.select_output_folder)
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出路径:"))
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_button)
        layout.addLayout(output_layout)

        # 日志输出区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: white; border: 1px solid gray; padding: 10px;")
        # 设置字体大小
        current_font = self.log_text.font()
        current_font.setPointSize(12)
        self.log_text.setFont(current_font)
        layout.addWidget(self.log_text)

        # 启动推理和取消推理
        self.start_button = QPushButton("开始推理(ENTER)")
        self.start_button.setStyleSheet("padding: 12px; font-weight: bold; font-size: 22px;")
        self.start_button.clicked.connect(self.run_inference)
        self.start_button.setShortcut(Qt.Key_Return)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("取消推理")
        self.stop_button.setStyleSheet("padding: 12px; font-weight: bold; font-size: 22px;")
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        # 窗口中央配件
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # 初始化提示日志
        self.log_message("WELCOME", "欢迎使用超分辨率重建系统！")
        self.log_message("WELCOME", "请选择输入类型、放大倍数和输入输出路径")
        self.log_message("WELCOME", "点击开始推理按钮或键盘回车(ENTER)进行处理")
        self.log_message("WELCOME", "推理过程中请勿关闭窗口，推理完成后可查看输出结果。")

    def display_comparison_dialog(self, input_path, output_path):
        """显示对比图像的对话框"""
        display_images(input_path, output_path)

    def keyPressEvent(self, event):
        """ESC退出程序"""
        if event.key() == Qt.Key_Escape:
            self.close()

    def load_config(self):
        """加载配置文件"""
        config = ConfigManager.load()
        self.input_path.setText(config.get("input_path", ""))
        self.output_path.setText(config.get("output_path", ""))
        scale_value = str(config.get("scale", "2"))
        for btn in self.scale_buttons:
            if btn.text().startswith(scale_value):
                btn.setChecked(True)
                break

    def save_config(self):
        """保存配置文件"""
        for btn in self.scale_buttons:
            if btn.isChecked():
                scale = btn.text().replace("×", "")
                break
        config = {"input_path": self.input_path.text(), "output_path": self.output_path.text(), "scale": scale}
        ConfigManager.save(config)

    def select_input_path(self):
        """选择输入路径"""
        if self.file_radio.isChecked() or self.video_radio.isChecked():
            file, _ = QFileDialog.getOpenFileName(self, "选择文件", self.input_path.text())
            if file:
                self.input_path.setText(file)
        else:
            folder = QFileDialog.getExistingDirectory(self, "选择文件夹", self.input_path.text())
            if folder:
                self.input_path.setText(folder)
        self.save_config()

    def select_output_folder(self):
        """选择输出路径"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_path.text())
        if folder:
            self.output_path.setText(folder)
        self.save_config()

    def log_message(self, level, message):
        """日志输出"""
        self.log_signal.log_signal.emit(f"[{level}] {message}")

    def update_log(self, message):
        """更新日志"""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()

    def stop_inference(self):
        """停止推理"""
        if self.process:
            self.process.terminate()
            self.log_message("INFO", "推理已取消")
            self.process = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def run_inference(self):
        """开始推理"""

        # 清空日志窗口
        self.log_text.clear()

        # 检查
        input_path = self.input_path.text()
        output_path = self.output_path.text()
        for btn in self.scale_buttons:
            if btn.isChecked():
                scale = int(btn.text().replace('×', ''))
                break

        if any(contains_chinese(p) for p in [input_path, output_path]):
            show_warning_chinese()
            return
        if not all(os.path.exists(p) for p in [input_path, output_path]):
            show_warning_path()
            return

        # 判断输入类型
        if self.video_radio.isChecked():
            input_type = 'video'
        elif self.folder_radio.isChecked():
            input_type = 'folder'
        else:
            input_type = 'image'
        if check_file_type(input_path) != input_type:
            # 文件类型不匹配
            show_warning_file()
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # 寻找对应放大倍数模型的路径
        model_type, script = get_model_info(input_type, scale)
        self.log_message("INFO", f"使用模型: {model_type}, 脚本: {script}")
        # print(f"使用模型: {model_type}, 脚本: {script}")

        def execute():
            """执行推理"""
            try:
                if input_type == 'image':
                    args = ["--input_file", input_path]
                elif input_type == 'folder':
                    args = ["--input", input_path]
                else:
                    args = ["--input_video_path", input_path]

                command = ["python", script] + args + ["--output", output_path] + ["--scale", str(scale)]
                self.process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                for line in self.process.stdout:
                    self.log_message("INFO", line.strip())
                for line in self.process.stderr:
                    self.log_message("ERROR", line.strip())
                self.process.wait()
                self.log_message("SUCCESS", "推理完成")

                self.process = None
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)

                if input_type == 'image':
                    # 显示图片
                    image_name = os.path.splitext(os.path.basename(input_path))[0]
                    output_image_path = os.path.join(output_path, f"{image_name}_{model_type}.png").replace(os.sep, "/")
                    self.log_message("SUCCESS", f"展示对比图片")
                    self.display_signal.emit(input_path, output_image_path)


                elif input_type == 'folder':
                    """文件夹只对比第一张,视频只对比第一帧, 视频需要创建临时文件夹而不是output文件夹里的, 并且当用户点击取消之后, 需要删除临时文件夹"""
                    # 打开输出文件夹
                    self.log_message("SUCCESS", "打开输出文件夹")
                    open_output_folder(output_path)
                else:
                    # 视频
                    self.log_message("SUCCESS", "打开输出文件夹")
                    open_output_folder(output_path)

            except Exception as e:
                if self.process != None:
                    self.log_message("ERROR", f"推理失败: {e}")

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
