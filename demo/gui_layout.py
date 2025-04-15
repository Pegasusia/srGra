import os
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QComboBox, QLabel,
                             QLineEdit, QRadioButton, QButtonGroup, QWidget, QFileDialog)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from gui import InferenceLogic
from utils import LogSignal


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

    def update_log(self, message):
        """更新日志窗口"""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()