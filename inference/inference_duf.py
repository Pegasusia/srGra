import os
import torch
import cv2
import time
import numpy as np
from basicsr.archs.duf_arch import DUF


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
    os.makedirs(output_folder, exist_ok=True)
    window_size = model.num_frames if hasattr(model, 'num_frames') else 7
    center = window_size // 2

    pad_front = [input_frames[0]] * center
    pad_back = [input_frames[-1]] * center
    padded_frames = pad_front + input_frames + pad_back


    crop_margin = 4  # 与 postprocess_output 中保持一致

    # 预处理第一帧确定尺寸
    padded_sample, pad_h, pad_w = pad_to_multiple_of(input_frames[0])
    h, w, _ = padded_sample.shape
    output_h = (h - pad_h) * scale - 2 * crop_margin
    output_w = (w - pad_w) * scale - 2 * crop_margin

    output_video_path = os.path.join(output_folder, "output_duf_fixed.mp4")
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
        frame_name = f"frame_{i:04d}.png"
        cv2.imwrite(os.path.join(output_folder, frame_name), output_img)
        video_writer.write(output_img)
        print(f"Processed and saved {frame_name}")

    video_writer.release()
    print(f"Full-resolution video saved to: {output_video_path}")


def main():

    start_time = time.perf_counter()

    model_path = r"D:\gracode\sr_models\Video\DUF\DUF_x2_16L.pth"
    input_video_path = r"D:\gracode\sr_data\video\video_12f.mp4"
    output_folder = r"D:\gracode\sr_results\duf_fixed2_gai"


    scale = 2
    num_layer = 16
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(model_path, scale=scale, num_layer=num_layer, device=device)

    cap = cv2.VideoCapture(input_video_path)
    input_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_frames.append(frame)
    cap.release()

    if len(input_frames) < 1:
        raise ValueError("视频帧为空")

    duf_full_frame_inference(model, input_frames, device, output_folder, scale=scale)



    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
