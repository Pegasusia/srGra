import os
import subprocess
import threading
from PyQt5.QtWidgets import (QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt
from gui.log_signal import LogSignal
from logic.config_manager import ConfigManager
from logic.path_utils import contains_chinese, show_warning_chinese, show_warning_path
from logic.model_dispatcher import get_model_info

class InferenceGUI(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像超分辨率重建系统")
        self.setGeometry(150, 150, 1500, 900)
        self.setStyleSheet("background-color: #f5f7fa;")

        self.log_signal = LogSignal()
        self.log_signal.log_signal.connect(self.update_log)

        ConfigManager.ensure_userdata_folder()

        self.init_ui()
        self.load_config()

    def init_ui(self):
        self.process = None

        self.setStyleSheet("""
            QLabel { font-size: 16px; color: #333; }
            QPushButton {
                font-size: 16px;
                padding: 10px;
                background-color: #4a90e2;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #357ab8; }
            QRadioButton { font-size: 16px; padding: 6px; }
            QLineEdit { font-size: 16px; padding: 6px; }
        """)

        layout = QVBoxLayout()

        banner = QLabel("超分辨率重建系统")
        banner.setStyleSheet("font-size: 28px; font-weight: bold; color: #1a1a1a; margin-bottom: 20px;")
        banner.setAlignment(Qt.AlignCenter)
        layout.addWidget(banner)

        self.file_radio = QRadioButton("单图片")
        self.folder_radio = QRadioButton("文件夹")
        self.video_radio = QRadioButton("视频")
        self.file_radio.setChecked(True)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(QLabel("选择输入类型:"))
        radio_layout.addWidget(self.file_radio)
        radio_layout.addWidget(self.folder_radio)
        radio_layout.addWidget(self.video_radio)
        layout.addLayout(radio_layout)

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

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: white; border: 1px solid gray;")
        layout.addWidget(self.log_text)

        self.start_button = QPushButton("开始推理(ENTER)")
        self.start_button.setStyleSheet("padding: 12px; font-weight: bold;")
        self.start_button.clicked.connect(self.run_inference)
        self.start_button.setShortcut(Qt.Key_Return)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("取消推理")
        self.stop_button.setStyleSheet("padding: 12px; background-color: #e94e4e; color: white; font-weight: bold;")
        self.stop_button.clicked.connect(self.stop_inference)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def load_config(self):
        config = ConfigManager.load()
        self.input_path.setText(config.get("input_path", ""))
        self.output_path.setText(config.get("output_path", ""))
        scale_value = str(config.get("scale", "2"))
        for btn in self.scale_buttons:
            if btn.text().startswith(scale_value):
                btn.setChecked(True)
                break

    def save_config(self):
        for btn in self.scale_buttons:
            if btn.isChecked():
                scale = btn.text().replace("×", "")
                break
        config = {
            "input_path": self.input_path.text(),
            "output_path": self.output_path.text(),
            "scale": scale
        }
        ConfigManager.save(config)

    def select_input_path(self):
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
        folder = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_path.text())
        if folder:
            self.output_path.setText(folder)
        self.save_config()

    def log_message(self, level, message):
        self.log_signal.log_signal.emit(f"[{level}] {message}")

    def update_log(self, message):
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()

    def stop_inference(self):
            if self.process:
                self.process.terminate()
                self.log_message("INFO", "推理已取消")
                self.process = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def run_inference(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
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

        if self.video_radio.isChecked():
            input_type = 'video'
        elif self.folder_radio.isChecked():
            input_type = 'folder'
        else:
            input_type = 'image'

        model_type, script = get_model_info(input_type, scale)
        self.log_message("INFO", f"使用模型: {model_type}, 脚本: {script}")

        def execute():
            try:
                if input_type == 'image':
                    args = ["--input_file", input_path]
                elif input_type == 'folder':
                    args = ["--input", input_path]
                else:
                    args = ["--input_path", input_path]

                command = ["python", script] + args + ["--output", output_path]
                self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                for line in self.process.stdout:
                    self.log_message("INFO", line.strip())
                for line in self.process.stderr:
                    self.log_message("ERROR", line.strip())
                self.process.wait()
                self.log_message("SUCCESS", "推理完成")
                self.process = None
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
            except Exception as e:
                self.log_message("ERROR", f"推理失败: {e}")

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
