import argparse
from tkinter import Scale
from tracemalloc import start
import cv2
import glob
import numpy as np
import os
import torch
import time

from basicsr.archs.edsr_arch import EDSR

import sys


def main():
    # Refresh stdout in the console
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')

    start_time = time.perf_counter()

    print("EDSR...")
    print("Load model...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to the pretrained EDSR model')
    parser.add_argument('--input', type=str, default=None, help='Input image folder')
    parser.add_argument('--input_file', type=str, default=None, help='Input single image file')
    parser.add_argument('--output', type=str, default=None, help='Output folder')
    parser.add_argument('--scale', type=int, default=None, help='Scale factor for super-resolution')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up model
    if args.scale ==2 :
        args.model_path = r'D:\gracode\sr_models\Pic\EDSR\EDSR_Mx2_f64b16.pth'
        model = EDSR(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=2)
    elif args.scale == 3:
        args.model_path = r'D:\gracode\sr_models\Pic\EDSR\EDSR_Mx3_f64b16.pth'
        model = EDSR(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=3)
    else:
        raise ValueError("无效的放大倍数")
    # model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    # model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=False)

    print(f"scale: {args.scale}")
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))['params'], strict=True)
    model.eval()
    model = model.to(device)
    print("Load model done...")

    print("Start...")
    os.makedirs(args.output, exist_ok=True)

    # print(model)

    # 模型载入正确
    # state_dict = torch.load(args.model_path, map_location=device)
    # if 'params' in state_dict:
    #     state_dict = state_dict['params']

    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # print('Missing keys:', missing_keys)
    # print('Unexpected keys:', unexpected_keys)


    if args.input_file:
        # Process a single image
        process_image(args.input_file, model, device, args.output)
    else:

        # 统计输入文件夹中的图像数量
        image_paths = sorted(glob.glob(os.path.join(args.input, '*')))
        total_images = len(image_paths)

        print(f"Total images: {total_images}")
        for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
            imgname = os.path.splitext(os.path.basename(path))[0]
            # print('Processing: ', idx, imgname)
            print(f'Processing: {idx + 1} / {total_images} ({(idx + 1) / total_images * 100:.2f}%)')

            # Read image
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(device)
            # Inference
            try:
                with torch.no_grad():
                    output = model(img)
            except Exception as error:
                print('Error', error, imgname)
            else:
                # Save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round().astype(np.uint8)
                cv2.imwrite(os.path.join(args.output, f'{imgname}_EDSR.png'), output)
                print('Save:', imgname + '_EDSR.png')


    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.6f} seconds")


def process_image(image_path, model, device, output_folder):
    """Process a single image."""
    imgname = os.path.splitext(os.path.basename(image_path))[0]
    print('Processing: ', imgname)
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(img)

        # Save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f'{imgname}_EDSR.png'), output)
        print('Save:', imgname + '_EDSR.png')

    except Exception as error:
        print('Error', error, imgname)



if __name__ == '__main__':
    main()