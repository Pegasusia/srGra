import os
import torch
import cv2
import numpy as np
from basicsr.archs.duf_arch import DUF


def load_model(model_path, scale=4, num_layer=52, device='cuda'):
    model = DUF(scale=scale, num_layer=num_layer)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['params'], strict=True)
    model.eval().to(device)
    print(f"Model loaded from {model_path}")
    return model


def preprocess_frames(frames, device):
    # 转为 float32 + 归一化
    frames = [frame.astype(np.float32) / 255.0 for frame in frames]
    # 转为 RGB（DUF 使用 RGB）
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    # (H, W, C) -> (C, H, W)
    frames = [np.transpose(frame, (2, 0, 1)) for frame in frames]
    # (T, C, H, W)
    frames = np.stack(frames, axis=0)
    # 转为 Tensor: (1, T, C, H, W)
    frames = torch.from_numpy(frames).unsqueeze(0).to(device)
    return frames


def postprocess_output(output):
    output = output.squeeze(0).cpu().numpy()  # (C, H, W)
    output = np.transpose(output, (1, 2, 0))  # (H, W, C)
    output = (output * 255.0).clip(0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # 转回 BGR
    return output


def duf_full_frame_inference(model, input_frames, device, output_folder, scale=4):
    os.makedirs(output_folder, exist_ok=True)

    window_size = model.num_frames if hasattr(model, 'num_frames') else 7
    center = window_size // 2

    # 边缘填充
    pad_front = [input_frames[0]] * center
    pad_back = [input_frames[-1]] * center
    padded_frames = pad_front + input_frames + pad_back

    # 获取原始帧尺寸
    h, w, _ = input_frames[0].shape
    output_h, output_w = h * scale, w * scale

    output_video_path = os.path.join(output_folder, "output_duf_full.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 25  # 可根据需要修改
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (output_w, output_h))

    for i in range(len(input_frames)):
        window = padded_frames[i:i + window_size]
        input_tensor = preprocess_frames(window, device)
        with torch.no_grad():
            output = model(input_tensor)
        output_img = postprocess_output(output)

        frame_name = f"frame_{i:04d}.png"
        cv2.imwrite(os.path.join(output_folder, frame_name), output_img)
        video_writer.write(output_img)
        print(f"Processed and saved {frame_name}")

    video_writer.release()
    print(f"Full-resolution video saved to: {output_video_path}")


def main():
    model_path = r"D:\gracode\sr_models\Video\DUF\DUF_x4_16L.pth"
    input_video_path = r"D:\gracode\sr_data\video\video_12f.mp4"
    output_folder = r"D:\gracode\sr_results\duf"
    scale = 4
    num_layer = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    model = load_model(model_path, scale=scale, num_layer=num_layer, device=device)

    # 读取视频帧
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

    # 对每一帧进行超分处理
    duf_full_frame_inference(model, input_frames, device, output_folder, scale=scale)


if __name__ == "__main__":
    main()
