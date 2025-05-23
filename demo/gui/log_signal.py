from PyQt5.QtCore import QObject, pyqtSignal


class LogSignal(QObject):
    """信号类，用于在不同模块之间传递日志信息"""
    log_signal = pyqtSignal(str)