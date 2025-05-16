import sys
from PyQt5.QtWidgets import QApplication
from gui.refactor_window import InferenceGUI

if __name__ == '__main__':
    """系统入口"""
    app = QApplication(sys.argv)

    # 窗口
    window = InferenceGUI()
    window.show()
    sys.exit(app.exec_())