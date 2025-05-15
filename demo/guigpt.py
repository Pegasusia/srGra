import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication
from gui.main_window import InferenceGUI

if __name__ == "__main__":
    """系统运行的入口"""

    # 设置多进程启动方法为 spawn，适用于 Windows 和 macOS
    multiprocessing.set_start_method("spawn")

    # 创建 PyQt5 应用程序实例
    app = QApplication(sys.argv)

    # 创建主窗口 显示程序
    window = InferenceGUI()
    window.show()
    sys.exit(app.exec_())