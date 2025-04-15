import argparse
import cv2
import glob
import os
import shutil
import torch
import time
import subprocess
from pathlib import Path

from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img


def inference(imgs, imgnames, model, save_path, total_frames, start_frame=0):
    """接收输入图像张量 imgs 和对应的图像名称 imgnames。
    使用 BasicVSR 模型对输入图像进行推理。
    将推理结果保存为图像文件。
    """

    print("It may take a while to process the first frame...")
    with torch.no_grad():
        outputs = model(imgs)  # 使用模型进行推理

    # save imgs
    outputs = outputs.squeeze()  # 去掉多余的维度
    outputs = list(outputs)  # 将输出转换为列表

    for idx, (output, imgname) in enumerate(zip(outputs, imgnames)):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_BasicVSR.png'), output)

        # 打印进度
        current_frame = start_frame + idx + 1
        percentage = (current_frame / total_frames) * 100
        print(f"Processing: {current_frame} / {total_frames} ({percentage:.2f}%)")



def main():
    start_time = time.perf_counter()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--input_path', type=str, default=None, help='input test image folder or video file')
    parser.add_argument('--save_path', type=str, default='results/BasicVSR', help='save image path')
    parser.add_argument('--interval', type=int, default=30, help='interval size')
    args = parser.parse_args()

    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicVSR(num_feat=64, num_block=30)
    # model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=False)

    model.eval()
    model = model.to(device)
    os.makedirs(args.save_path, exist_ok=True)

    print("Start inference...")
    # extract images from video format files
    input_path = args.input_path

    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSR_tmp', video_name).replace(os.sep, "/")

        os.makedirs(input_path, exist_ok=True)
        ffmpeg_command = f'ffmpeg -i "{args.input_path}" -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 "{input_path}/frame%08d.png"'
        # if os.system(ffmpeg_command) != 0:
        #     print("Error: Failed to extract frames using ffmpeg. Please check the input file and ffmpeg installation.")
        #     return

        # 使用subprocess执行命令并获取输出
        try:
            result = subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # print("命令输出:", result.stdout.decode())
            print(f"Extracted frames to {input_path}")
        except subprocess.CalledProcessError as e:
            print("failed:", e.stderr.decode())

    # load data and inference
    print("Loading data...")
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    if not imgs_list:
        print(f"Error: No images found in {input_path}. Please check the input path or ffmpeg command.")
        if use_ffmpeg:
            shutil.rmtree(input_path)  # 删除临时文件夹
        return

    num_imgs = len(imgs_list)

    print("per process interval:", args.interval)
    if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.unsqueeze(0).to(device)
        inference(imgs, imgnames, model, args.save_path, total_frames=num_imgs)
    else:
        for idx in range(0, num_imgs, args.interval):
            interval = min(args.interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, args.save_path, total_frames=num_imgs, start_frame=idx)


    # 合成视频
    if use_ffmpeg:
        print("Combining frames into video...")
        output_video_path = os.path.join(args.save_path, f"{video_name}_BasicVSR.mp4").replace(os.sep, "/")
        ffmpeg_command = f'ffmpeg -framerate 30 -i "{args.save_path}/frame%08d_BasicVSR.png" -c:v libx264 -pix_fmt yuv420p "{output_video_path}"'
        # if os.system(ffmpeg_command) != 0:
        #     print("Error: Failed to combine frames into video. Please check ffmpeg installation.")
        # else:
        #     print(f"Video saved at {output_video_path}")

        # 使用subprocess执行命令并获取输出
        try:
            result = subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Video saved at {output_video_path}")
        except subprocess.CalledProcessError as e:
            print("failed:", e.stderr.decode())

    shutil.rmtree(input_path)  # 删除临时文件夹
    for filename in os.listdir(os.path.dirname(output_video_path)):
        file_path = Path(os.path.dirname(output_video_path)) / filename
        if os.path.isfile(file_path) and filename.endswith('.png'):
            os.remove(file_path)  # 删除所有图片


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    if elapsed_time >= 60:
        # 如果超过一分钟，按分钟:秒格式显示
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Total time: {minutes:02d}:{seconds:02d} (minutes:seconds)")
    else:
        # 否则直接显示秒数，保留6位小数
        print(f"Total time: {elapsed_time:.6f} seconds")



if __name__ == '__main__':
    main()