import argparse
import cv2
import glob
import os
import shutil
import torch

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img


def inference(imgs, imgnames, model, save_path):
    with torch.no_grad():
        outputs = model(imgs)
    # save imgs
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_BasicVSRPP.png'), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='D:\gracode\srGra\models\Video\BasicVSR\BasicVSR_REDS4.pth')
    parser.add_argument(
        '--input_path', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='results', help='save image path')
    parser.add_argument('--interval', type=int, default=15, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up model
    model = BasicVSRPlusPlus(mid_channels=64, num_blocks=7)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    # extract images from video format files
    input_path = args.input_path
    use_ffmpeg = False
    # if not os.path.isdir(input_path):
    #     use_ffmpeg = True
    #     video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
    #     input_path = os.path.join('./BasicVSRPP_tmp', video_name)
    #     os.makedirs(os.path.join('./BasicVSRPP_tmp', video_name), exist_ok=True)
    #     os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSRPP_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSRPP_tmp', video_name), exist_ok=True)
        # 修复 ffmpeg 命令，确保输出文件名有扩展名
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 {input_path}/frame%08d.png')

    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.unsqueeze(0).to(device)
        inference(imgs, imgnames, model, args.save_path)
    else:
        for idx in range(0, num_imgs, args.interval):
            interval = min(args.interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, args.save_path)
            print(111)

    # delete ffmpeg output images
    if use_ffmpeg:
        shutil.rmtree(input_path)


if __name__ == '__main__':
    main()
