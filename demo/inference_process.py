import os
import subprocess
import threading

def run_inference_process(input_path, model_path, output_folder, selected_model, log_queue):
    """独立的推理函数，用于多进程调用"""

    # log_queue.put(f"开始推理: {input_path}, 模型: {model_path}, 输出: {output_folder}, 网络: {selected_model}")

    # 根据模型选择推理脚本
    if selected_model == "ESRGAN":
        script_name = "inference/inference_esrgan.py"
    elif selected_model == "EDSR":
        script_name = "inference/inference_edsr.py"
    elif selected_model == "BasicVSR":
        script_name = "inference/inference_basicvsr.py"
    else:
        log_queue.put("[ERROR] 未知的神经网络模型选择.")
        return

    # 构建命令
    command = [
        "python", script_name, "--input_path", input_path, "--model_path", model_path, "--save_path", output_folder
    ]

    # 执行推理脚本
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
            bufsize=1,
            env={
                **os.environ, "PYTHONUNBUFFERED": "1"
            }  # 禁用缓冲
        )

        def read_stream(stream, log_level):
            """读取子进程的输出流并发送到日志队列"""
            for line in iter(stream.readline, ""):
                log_queue.put(f"{line.strip()}")
            stream.close()

        # 创建线程读取 stdout 和 stderr
        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, "INFO"))
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, "ERROR"))
        stdout_thread.start()
        stderr_thread.start()

        # 等待子进程完成
        process.wait()
        stdout_thread.join()
        stderr_thread.join()

        if process.returncode == 0:
            log_queue.put("[SUCCESS] 视频分辨率重建成功!\n")

            # 自动打开输出文件夹
            if os.name == "nt":  # Windows
                os.startfile(output_folder)
            elif os.name == "posix":
                # macOS 或 Linux
                subprocess.Popen(["open", output_folder])
            else:
                self.log_message("ERROR", "无法打开输出文件夹，请手动检查路径。")
        else:
            log_queue.put(f"[ERROR] 推理失败，返回代码：{process.returncode}")
    except Exception as e:
        log_queue.put(f"[ERROR] 推理过程中发生错误：{str(e)}")
