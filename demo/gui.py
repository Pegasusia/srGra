import os
import threading
import subprocess
import multiprocessing
from utils import LogSignal


class InferenceLogic:

    def __init__(self, gui):
        self.gui = gui
        self.process = None

    def ensure_userdata_folder(self):
        """确保 userdata 文件夹存在"""
        userdata_folder = os.path.dirname(self.gui.CONFIG_FILE)
        if not os.path.exists(userdata_folder):
            os.makedirs(userdata_folder)
            self.gui.log_message("INFO", f"创建文件夹: {userdata_folder}")

    def load_config(self):
        """加载 YAML 配置文件"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    config = yaml.safe_load(f)
                    self.input_path.setText(config.get("input_path", ""))
                    self.model_path.setText(config.get("model_path", ""))
                    self.output_path.setText(config.get("output_folder", ""))
                    # 加载下拉列表选项
                    selected_model = config.get("selected_model", "ESRGAN")
                    index = self.model_selection_combo.findText(selected_model)
                    if index != -1:
                        self.model_selection_combo.setCurrentIndex(index)
                    self.log_message("INFO", f"加载配置文件 {self.CONFIG_FILE}")

            except Exception as e:
                self.log_message("ERROR", f"无法加载配置文件: {str(e)}")
        else:
            self.log_message("ERROR", f"没有配置文件 {self.CONFIG_FILE}")

    def save_config(self):
        """保存配置到 YAML 文件"""
        config = {
            "input_path": self.input_path.text(),
            "model_path": self.model_path.text(),
            "output_folder": self.output_path.text(),
            "selected_model": self.model_selection_combo.currentText()
        }
        try:
            with open(self.CONFIG_FILE, "w") as f:
                yaml.safe_dump(config, f)
                self.log_message("INFO", f"保存配置文件到 {self.CONFIG_FILE}")
        except Exception as e:
            self.log_message("ERROR", f"保存配置文件发生错误: {str(e)}")

    def select_input_path(self):
        if self.file_radio.isChecked():
            # 选择单个文件

            file, _ = QFileDialog.getOpenFileName(self, "选择输入文件", self.input_path.text(),
                                                  "Image Files (*.png *.jpg *.jpeg)")
            if file:
                self.input_path.setText(file)
        elif self.folder_radio.isChecked():
            # 选择文件夹
            folder = QFileDialog.getExistingDirectory(self, "选择输入文件夹", self.input_path.text())
            if folder:
                self.input_path.setText(folder)

        elif self.video_radio.isChecked():
            # 选择视频文件
            file, _ = QFileDialog.getOpenFileName(self, "选择输入视频", self.input_path.text(),
                                                "Video Files (*.mp4 *.avi *.mov *.mkv)")
            if file:
                self.input_path.setText(file)

        self.save_config()  # 保存配置

    def select_model_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择模型", self.model_path.text(), "PyTorch Model Files (*.pth)")
        if file:
            self.model_path.setText(file)
            self.save_config()  # 保存配置

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", self.output_path.text())
        if folder:
            self.output_path.setText(folder)
            self.save_config()  # 保存配置

    def start_inference(self):
        """开始推理"""

        # 清空日志窗口
        self.log_text.clear()

        self.log_message("INFO", "程序开始运行...")

        # 获取输入、模型和输出路径
        input_path = self.input_path.text()
        model_path = self.model_path.text()
        output_folder = self.output_path.text()

        # 检查路径是否有效
        if self.file_radio.isChecked() and not os.path.isfile(input_path):
            self.log_message("ERROR", "无效的输入路径.")
            return
        if self.folder_radio.isChecked() and not os.path.isdir(input_path):
            self.log_message("ERROR", "无效的输入文件夹.")
            return
        if self.video_radio.isChecked() and not os.path.isfile(input_path):
            self.log_message("ERROR", "无效的视频文件.")
            return
        if not os.path.isfile(model_path):
            self.log_message("ERROR", "无效的模型路径.")
            return
        if not os.path.isdir(output_folder):
            self.log_message("ERROR", "无效的输出路径.")
            return

        # 禁用按钮以防止重复点击
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        if self.video_radio.isChecked():
            # 视频模式，使用多线程
            self.start_multi_process(input_path, model_path, output_folder)
        else:
            # 图片/文件夹图片 使用多线程
            self.start_multi_thread(input_path, model_path, output_folder)

    def start_multi_thread(self, input_path, model_path, output_folder):
        """启动多线程推理"""
        inference_thread = threading.Thread(target=self.run_inference, args=(input_path, model_path, output_folder))
        inference_thread.start()

    def start_multi_process(self, input_path, model_path, output_folder):
        """启动多进程推理"""
        # 使用多进程调用推理函数
        multi_process = multiprocessing.Process(target=self.run_inference, args=(input_path, model_path, output_folder))
        multi_process.start()
        multi_process.join()  # 等待进程结束

    def stop_inference(self):
        """停止推理"""
        if self.process:
            self.process.terminate()  # 终止推理进程
            self.log_message("INFO", "用户终止运行.\n")
            self.process = None

        # 重置按钮状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def run_inference(self, input_path, model_path, output_folder):
        # 使用 subprocess 调用 inference_esrgan.py 脚本
        input_image_path = None  # 初始化变量
        output_image_path = None  # 初始化变量

        # 根据用户选择的神经网络模型，选择对应的推理脚本
        selected_model = self.model_selection_combo.currentText()
        if selected_model == "ESRGAN":
            script_name = "inference/inference_esrgan.py"
            output_suffix = "_ESRGAN.png"
        elif selected_model == "EDSR":
            script_name = "inference/inference_edsr.py"
            output_suffix = "_EDSR.png"
        # elif selected_model == "MSRResNet":
        #     script_name = "inference/inference_msrresnet.py"
        #     output_suffix = "_MSRResNet.png"
        elif selected_model == "BasicVSR":
            script_name = "inference/inference_basicvsr.py"
            output_suffix = "_BasicVSR.png"  # 视频输出文件
        else:
            self.log_message("ERROR", "未知的神经网络模型选择.")
            return

        if self.file_radio.isChecked():
            # 单个文件模式
            command = [
                "python", script_name, "--input_file", input_path, "--model_path", model_path,
                "--output", output_folder
            ]
            input_image_path = input_path
            output_image_path = os.path.join(output_folder,
                                             f"{os.path.splitext(os.path.basename(input_path))[0]}{output_suffix}")
        elif self.folder_radio.isChecked():
            # 文件夹模式
            command = [
                "python", script_name, "--input", input_path, "--model_path", model_path,
                "--output", output_folder
            ]
            # 获取文件夹中第一个输入文件和对应的输出文件
            input_files = sorted(os.listdir(input_path))
            if input_files:
                first_input_image = input_files[0]
                input_image_path = os.path.join(input_path, first_input_image)
                first_output_image = f"{os.path.splitext(first_input_image)[0]}{output_suffix}"
                output_image_path = os.path.join(output_folder, first_output_image)

        elif self.video_radio.isChecked():
            # 视频模式
            command = [
                "python", script_name, "--input_path", input_path, "--model_path", model_path,
                "--save_path", output_folder
            ]
            input_image_path = input_path
            output_image_path = os.path.join(output_folder,
                                            f"{os.path.splitext(os.path.basename(input_path))[0]}{output_suffix}")


        try:
            # 检查是否正确设置了输入和输出路径
            if not input_image_path or not output_image_path:
                self.log_message("ERROR", "无有效路径.")
                return

            # 使用 subprocess 捕获输出并实时更新日志
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
                universal_newlines=True,
                env={
                    **os.environ, "PYTHONUNBUFFERED": "1"
                }  # 禁用缓冲
            )

            # 实时读取 stdout 和 stderr
            for line in self.process.stdout:
                self.log_message("INFO", line.strip())
            for line in self.process.stderr:
                self.log_message("ERROR", f"ERROR: {line.strip()}")

            self.process.wait()  # 等待进程结束
            if self.process.returncode == 0:
                if self.file_radio.isChecked():
                    self.log_message("SUCCESS", "图片分辨率重建成功\n")
                    self.display_images(input_image_path, output_image_path)
                    self.open_output_folder(output_folder)
                elif self.video_radio.isChecked():
                    self.log_message("SUCCESS", "视频分辨率重建成功\n")
                    # self.display_video_frames(input_image_path, output_image_path)
                    # self.open_output_folder(output_folder)
            else:
                self.log_message("ERROR", f"Failed with return code {self.process.returncode}")

        except Exception as e:
            if self.process is not None:
                self.log_message("ERROR", f"Error: {str(e)}")

        finally:
            # 重置按钮状态
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.process = None

    def display_images(self, input_image_path, output_image_path):
        """显示输入图像和生成图像，并显示图像信息"""
        # 显示输入图像
        input_pixmap = QPixmap(input_image_path)
        input_pixmap = input_pixmap.scaled(self.input_image_label.width(), self.input_image_label.height())
        self.input_image_label.setPixmap(input_pixmap)

        # 获取输入图像的真实分辨率
        with Image.open(input_image_path) as img:
            input_width, input_height = img.size
            # input_data = list(img.getdata())  # 获取输入图像的像素数据
        input_size = os.path.getsize(input_image_path) / 1024  # 文件大小 (KB)
        self.input_image_info.setText(f"输入图片信息: {input_width}x{input_height}, {input_size:.2f} KB")

        # 显示生成图像
        output_pixmap = QPixmap(output_image_path)
        output_pixmap = output_pixmap.scaled(self.output_image_label.width(), self.output_image_label.height())
        self.output_image_label.setPixmap(output_pixmap)

        # 获取生成图像的真实分辨率
        with Image.open(output_image_path) as img:
            output_width, output_height = img.size
            # output_data = list(img.getdata())  # 获取输出图像的像素数据
        output_size = os.path.getsize(output_image_path) / 1024  # 文件大小 (KB)
        self.output_image_info.setText(f"输出图片信息: {output_width}x{output_height}, {output_size:.2f} KB")


    def display_video_frames(self, input_video_path, output_video_path):
        """显示输入视频和生成视频的第一帧"""
        # 提取输入视频的第一帧
        input_frame_path = "./temp_input_frame.png"
        os.system(f'ffmpeg -i "{input_video_path}" -vf "select=eq(n\,0)" -q:v 3 "{input_frame_path}" -y')

        # 提取生成视频的第一帧
        output_frame_path = "./temp_output_frame.png"
        os.system(f'ffmpeg -i "{output_video_path}" -vf "select=eq(n\,0)" -q:v 3 "{output_frame_path}" -y')

        # 显示输入视频的第一帧
        input_pixmap = QPixmap(input_frame_path)
        input_pixmap = input_pixmap.scaled(self.input_image_label.width(), self.input_image_label.height())
        self.input_image_label.setPixmap(input_pixmap)

        # 显示生成视频的第一帧
        output_pixmap = QPixmap(output_frame_path)
        output_pixmap = output_pixmap.scaled(self.output_image_label.width(), self.output_image_label.height())
        self.output_image_label.setPixmap(output_pixmap)

        # 删除临时帧文件
        os.remove(input_frame_path)
        os.remove(output_frame_path)

    def open_output_folder(self, folder_path):
        """自动打开输出文件夹"""
        if os.name == "nt":  # Windows
            os.startfile(folder_path)
        elif os.name == "posix":  # macOS 或 Linux
            subprocess.Popen(["open", folder_path])
        else:
            self.log_message("ERROR", "无法打开输出文件夹，请手动检查路径。")

