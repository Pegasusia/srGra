import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication
from gui.main_window import InferenceGUI

if __name__ == "__main__":
    """系统入口"""
    multiprocessing.set_start_method("spawn")
    app = QApplication(sys.argv)
    window = InferenceGUI()
    window.show()
    sys.exit(app.exec_())