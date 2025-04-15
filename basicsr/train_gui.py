import sys
import threading
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QTextEdit, QWidget
from PyQt5.QtCore import pyqtSignal, QObject

# 信号类，用于线程间通信
class LogSignal(QObject):
    log_signal = pyqtSignal(str)

# 主窗口类
class TrainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # 创建信号对象
        self.log_signal = LogSignal()
        self.log_signal.log_signal.connect(self.update_log)

    def init_ui(self):
        self.setWindowTitle("Training Progress")
        self.setGeometry(100, 100, 800, 600)

        # 主布局
        layout = QVBoxLayout()

        # 日志显示窗口
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # 开始训练按钮
        self.start_button = QPushButton("Start Training", self)
        self.start_button.clicked.connect(self.start_training)
        layout.addWidget(self.start_button)

        # 设置中心窗口
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def start_training(self):
        # 禁用按钮以防止重复点击
        self.start_button.setEnabled(False)

        # 启动训练线程
        train_thread = threading.Thread(target=self.run_training)
        train_thread.start()

    def run_training(self):
        # 使用 subprocess 调用 basicsr/train.py 脚本
        command = [
            "python", "basicsr/train.py",
            "-opt", "options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml"  # 替换为实际的 YAML 配置文件路径
        ]

        try:
            # 使用 subprocess 捕获输出并实时更新日志
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            for line in process.stdout:
                self.log_signal.log_signal.emit(line.strip())
            for line in process.stderr:
                self.log_signal.log_signal.emit(line.strip())

            process.wait()  # 等待进程结束
            if process.returncode == 0:
                self.log_signal.log_signal.emit("Training Completed Successfully!")
            else:
                self.log_signal.log_signal.emit(f"Training Failed with return code {process.returncode}")

        except Exception as e:
            self.log_signal.log_signal.emit(f"Error: {str(e)}")

        # 重新启用按钮
        self.start_button.setEnabled(True)

    def update_log(self, message):
        # 更新日志窗口
        self.log_text.append(message)

# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TrainGUI()
    gui.show()
    sys.exit(app.exec_())