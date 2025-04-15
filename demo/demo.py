import sys
from PyQt5.QtWidgets import QApplication
import multiprocessing

# 主程序入口
if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method("spawn")
    app = QApplication(sys.argv)
    gui = InferenceGUI()
    gui.show()
    sys.exit(app.exec_())