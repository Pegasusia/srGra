from PyQt5.QtWidgets import QApplication, QProgressBar, QWidget
import time
import threading
import os


def show_progress_bar():
    app = QApplication([])
    window = QWidget()
    progress = QProgressBar(window)
    progress.setRange(0, 100)
    progress.setValue(0)

    window.setWindowTitle("Processing Image")
    window.setGeometry(100, 100, 300, 50)
    window.show()

    # 模拟更新进度条
    for i in range(100):
        progress.setValue(i)
        app.processEvents()  # 允许GUI更新
        time.sleep(0.1)  # 暂停一小段时间以模拟进度

    app.quit()


def process_image_with_progress_bar(image_path, model, device, output_folder):
    """使用进度条表示处理进度"""
    threading.Thread(target=show_progress_bar).start()
    process_image(image_path, model, device, output_folder)
