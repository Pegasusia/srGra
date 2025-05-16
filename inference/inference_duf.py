import os
import sys
import cv2
import time
import torch
import shutil
import tempfile
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from basicsr.archs.duf_arch import DUF
from PyQt5.QtWidgets import QApplication

# Add the parent directory of 'demo' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from demo.gui.image_display import display_images


def load_model(model_path, scale=4, num_layer=52, device='cuda'):
    model = DUF(scale=scale, num_layer=num_layer)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['params'], strict=True)
    model.eval().to(device)
    print(f"Model loaded from {model_path}")
    return model


def pad_to_multiple_of(x, multiple=4):
    """反射填充图像，使其高宽为指定倍数"""
    h, w = x.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(x, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT)
    return padded, pad_h, pad_w


def preprocess_frames(frames, device):
    frames = [frame.astype(np.float32) / 255.0 for frame in frames]
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    frames = [np.transpose(frame, (2, 0, 1)) for frame in frames]
    frames = np.stack(frames, axis=0)  # (T, C, H, W)
    frames = torch.from_numpy(frames).unsqueeze(0).to(device)  # (1, T, C, H, W)
    return frames


def postprocess_output(output, scale, pad_h, pad_w, crop_margin=4):
    """
    后处理模型输出，裁剪 padding 和边缘锯齿。

    Args:
        output (torch.Tensor): 输出张量 (B, C, H, W)
        scale (int): 放大倍数
        pad_h (int): 输入高度 pad 的行数
        pad_w (int): 输入宽度 pad 的列数
        crop_margin (int): 要从每边裁剪的像素数（去除锯齿）

    Returns:
        np.ndarray: 处理后的图像 (H, W, C)
    """
    output = output.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))  # (H, W, C)
    output = (output * 255.0).clip(0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    if pad_h > 0:
        output = output[:-pad_h * scale, :]
    if pad_w > 0:
        output = output[:, :-pad_w * scale]

    # 手动裁剪四周（每边 crop_margin 个像素）
    h, w, _ = output.shape
    output = output[crop_margin:h - crop_margin, crop_margin:w - crop_margin, :]

    return output

def duf_full_frame_inference(model, input_frames, device, output_folder, scale=4):
    """使用 DUF 模型进行全帧推理并保存结果"""

    if QApplication.instance() is None:
        app = QApplication(sys.argv)


    # 创建显式可见的临时目录
    tmp_output = os.path.join(output_folder, "_temp_duf")
    os.makedirs(tmp_output, exist_ok=True)

    window_size = model.num_frames if hasattr(model, 'num_frames') else 7
    center = window_size // 2

    pad_front = [input_frames[0]] * center
    pad_back = [input_frames[-1]] * center
    padded_frames = pad_front + input_frames + pad_back

    # 裁剪边缘锯齿
    crop_margin = 4
    padded_sample, pad_h, pad_w = pad_to_multiple_of(input_frames[0])
    h, w, _ = padded_sample.shape
    output_h = (h - pad_h) * scale - 2 * crop_margin
    output_w = (w - pad_w) * scale - 2 * crop_margin

    output_video_path = os.path.join(tmp_output, "output_temp.mp4").replace(os.sep, '/')
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

        # 保存当前帧
        out_frame_name = f"sr_{i:04d}.png"
        out_path = os.path.join(tmp_output, out_frame_name).replace(os.sep, "/")
        cv2.imwrite(out_path, output_img)
        video_writer.write(output_img)

        print(f"Processing {out_frame_name} ({(i+1)/len(input_frames) * 100:.2f}%)")

        # 保存低分图
        input_bgr = input_frames[i]
        low_path = os.path.join(tmp_output, f"lr_{i:04d}.png").replace(os.sep, "/")
        cv2.imwrite(low_path, input_bgr)

        # 展示对比（每一帧更新窗口）
        if i == 0 or i == len(input_frames) - 1:
            # 仅在第一帧和最后一帧显示对比
            display_images(low_path, out_path)

    video_writer.release()

    # 移动视频到目标目录
    final_output_path = os.path.join(output_folder, "output_duf_final.mp4").replace(os.sep, "/")
    shutil.move(output_video_path, final_output_path)
    print(f"saved: {final_output_path}")

    # 清理临时目录（仅删除图像，保留输出视频已移出）
    for file in os.listdir(tmp_output):
        path = os.path.join(tmp_output, file)
        if file.lower().endswith(".png"):
            os.remove(path)
    os.rmdir(tmp_output)
    print("tmp done")


def main():
    # Refresh stdout in the console
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')
    start_time = time.perf_counter()

    # model_path = r"D:\gracode\sr_models\Video\DUF\DUF_x2_16L.pth"
    # input_video_path = r"D:\gracode\sr_data\video\video_12f.mp4"
    # output_folder = r"D:\gracode\sr_results\duf_fixed2_gai"
    # scale = 2
    # num_layer = 16

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = load_model(model_path, scale=scale, num_layer=num_layer, device=device)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=r"D:\gracode\sr_models\Video\DUF\DUF_x2_16L.pth",
        help='Path to the pretrained DUF model')
    parser.add_argument('--input_video_path', type=str, default=None, help='Input video file path')
    parser.add_argument('--output', type=str, default=None, help='Output folder')
    parser.add_argument('--scale', type=int, default=None, help='Scale factor for super-resolution')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up model
    if args.scale == 2:
        args.model_path = r'D:\gracode\sr_models\Video\DUF\DUF_x2_16L.pth'
        num_layer = 16
    elif args.scale == 3:
        args.model_path = r'D:\gracode\sr_models\Video\DUF\DUF_x3_16L.pth'
        num_layer = 16
    elif args.scale == 4:
        args.model_path = r'D:\gracode\sr_models\Video\DUF\DUF_x4_52L.pth'
        num_layer = 52
    else:
        raise ValueError("无效的放大倍数")

    print(f"scale: {args.scale}")
    model = load_model(args.model_path, scale=args.scale, num_layer=num_layer, device=device)

    print("Load model done...")
    print("Start...")
    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(args.input_video_path)
    input_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_frames.append(frame)
    cap.release()

    if len(input_frames) < 1:
        raise ValueError("视频帧为空")

    duf_full_frame_inference(model, input_frames, device, args.output, scale=args.scale)

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
