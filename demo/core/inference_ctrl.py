import os
import subprocess
import threading
import multiprocessing
from PyQt5.QtWidgets import QMessageBox
from .utils import validate_paths, show_warning


class InferenceController:

    def __init__(self, log_handler, config_manager, ui_fields):
        self.log = log_handler
        self.config = config_manager
        self.ui = ui_fields
        self.process = None

    def start_inference(self):
        """启动推理主逻辑"""
        input_path = self.ui.input_path.text()
        model_path = self.ui.model_path.text()
        output_folder = self.ui.output_path.text()

        # 路径校验
        if not validate_paths(input_path, model_path, output_folder):
            show_warning("Invalid Path", "路径包含中文或不存在!")
            return

        # 根据输入类型选择处理方式
        if self.ui.video_radio.isChecked():
            self._start_video_process(input_path, model_path, output_folder)
        else:
            self._start_image_thread(input_path, model_path, output_folder)

    def _start_video_process(self, input_path, model_path, output_folder):
        """视频处理（多进程）"""
        selected_model = self.ui.model_selection_combo.currentText()
        log_queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=self._run_video_inference, args=(input_path, model_path, output_folder, selected_model, log_queue))
        process.start()

        threading.Thread(target=self._monitor_log_queue, args=(log_queue, process), daemon=True).start()

    def _run_video_inference(self, *args):
        """视频推理具体实现"""
        # 这里需要调用你的视频处理脚本
        pass

    def _start_image_thread(self, input_path, model_path, output_folder):
        """图像处理（多线程）"""
        thread = threading.Thread(target=self._run_image_inference, args=(input_path, model_path, output_folder))
        thread.start()

    def _run_image_inference(self, input_path, model_path, output_folder):
        """图像推理具体实现"""
        # 这里需要调用你的图像处理脚本
        pass

    def _monitor_log_queue(self, log_queue, process):
        """监控日志队列"""
        while process.is_alive() or not log_queue.empty():
            try:
                message = log_queue.get(timeout=0.1)
                self.log.emit("INFO", message)
            except:
                pass
        self.ui.start_button.setEnabled(True)
        self.ui.stop_button.setEnabled(False)