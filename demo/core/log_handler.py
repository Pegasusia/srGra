from PyQt5.QtCore import QObject, pyqtSignal


class LogSignal(QObject):
    """信号类 用于在不同线程之间传递日志信息"""
    log_signal = pyqtSignal(str)


class LogHandler:
    """日志处理类 负责将日志信息发送到UI组件"""

    def __init__(self, text_edit):
        self.signal = LogSignal()
        self.text_edit = text_edit
        self.signal.log_signal.connect(self._update_log)

    def emit(self, level, message):
        formatted = f"[{level}] {message}"
        self.signal.log_signal.emit(formatted)

    def _update_log(self, message):
        self.text_edit.append(message)
        self.text_edit.ensureCursorVisible()