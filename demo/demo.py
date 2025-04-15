import sys
from PyQt5.QtWidgets import QApplication
from gui_layout import InferenceGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = InferenceGUI()
    gui.show()
    sys.exit(app.exec_())