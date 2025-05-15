import sys
from PyQt5.QtWidgets import QApplication
from gui.refactor_window import InferenceGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InferenceGUI()
    window.show()
    sys.exit(app.exec_())