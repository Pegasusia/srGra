import os
import shutil
import torch
import cv2
import time
import argparse
import tempfile
from pathlib import Path
import numpy as np
from basicsr.archs.duf_arch import DUF
from inference_process import display_images  # 假设你已有该函数


def duf_full_frame_inference(model, input_frames, device, output_folder, scale=4):
    # 使用临时目录保存图像帧
    with tempfile.TemporaryDirectory() as tmp_output:
        print(f"Using temporary folder: {tmp_output}")

        os.makedirs(output_folder, exist_ok=True)
        window_size = model.num_frames if hasattr(model, 'num_frames') else 7
        center = window_size // 2

        pad_front = [input_frames[0]] * center
        pad_back = [input_frames[-1]] * center
        padded_frames = pad_front + input_frames + pad_back

        crop_margin = 4
        padded_sample, pad_h, pad_w = pad_to_multiple_of(input_frames[0])
        h, w, _ = padded_sample.shape
        output_h = (h - pad_h) * scale - 2 * crop_margin
        output_w = (w - pad_w) * scale - 2 * crop_margin

        output_video_path = os.path.join(tmp_output, "output_duf_temp.mp4").replace(os.sep, '/')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 25
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (output_w, output_h))

        for i in range(len(input_frames)):
            window = padded_frames[i:i + window_size]
            padded_window = [pad_to_multiple_of(frame)[0] for frame in window]
            input_tensor = preprocess_frames(padded_window, device)

            with torch.no_grad():
                output = model(input_tensor)

            output_img = postprocess_output(output, scale=scale, pad_h=pad_h, pad_w=pad_w, crop_margin=crop_margin)
            frame_name = f"frame_{i:04d}_duf.png"
            out_path = os.path.join(tmp_output, frame_name)
            cv2.imwrite(out_path, output_img)
            video_writer.write(output_img)
            print(f"[Frame {i+1}/{len(input_frames)}] Saved: {frame_name}")

            # 实时显示图像对比（原始 vs 超分）
            input_bgr = input_frames[i]
            input_path = os.path.join(tmp_output, f"low_{i:04d}.png")
            cv2.imwrite(input_path, input_bgr)
            display_images(input_path, out_path)  # 调用你的对比展示函数

        video_writer.release()

        # 处理完成，复制视频到最终目录
        final_output_path = os.path.join(output_folder, "output_duf_final.mp4").replace(os.sep, "/")
        shutil.move(output_video_path, final_output_path)
        print(f"Final video saved to: {final_output_path}")
        # 临时目录自动删除


# 你现有的 pad_to_multiple_of, preprocess_frames, postprocess_output 保持不变
# main 函数调用此新版 duf_full_frame_inference 即可
