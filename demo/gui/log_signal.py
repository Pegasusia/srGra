from PyQt5.QtCore import QObject, pyqtSignal


class LogSignal(QObject):
    log_signal = pyqtSignal(str)