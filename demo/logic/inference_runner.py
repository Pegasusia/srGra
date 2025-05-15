import threading
import multiprocessing
from inference_process import run_inference_process


class InferenceRunner:
    """处理推理的类，负责启动推理进程或线程，并更新日志信息。"""

    def __init__(self, gui):
        self.gui = gui

    def start(self, input_path, model_path, output_folder):
        if self.gui.video_radio.isChecked():
            self._start_process(input_path, model_path, output_folder)
        else:
            self._start_thread(input_path, model_path, output_folder)

    def _start_thread(self, input_path, model_path, output_folder):
        thread = threading.Thread(target=self.gui.run_inference, args=(input_path, model_path, output_folder))
        thread.start()

    def _start_process(self, input_path, model_path, output_folder):
        log_queue = multiprocessing.Queue()
        model = self.gui.model_selection_combo.currentText()
        process = multiprocessing.Process(
            target=run_inference_process, args=(input_path, model_path, output_folder, model, log_queue))
        process.start()

        threading.Thread(target=self.gui.update_log_queue, args=(log_queue, process), daemon=True).start()
