import argparse
from turtle import end_fill
import cv2
import glob
import numpy as np
import os
import torch
import time

from basicsr.archs.rrdbnet_arch import RRDBNet

import sys


def main():
    # refresh stdout in the console
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8')

    start_time = time.perf_counter()

    print("ESRGAN...")
    print("Load model...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        r'D:\gracode\srGra\models\Pic\004_MSRGAN_x4\models\net_d_400000.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default= None, help='input image folder')
    parser.add_argument('--input_file', type=str, default= None, help='input single image file')
    parser.add_argument('--output', type=str, default='results/ESRGAN', help='output folder')
    args = parser.parse_args()

    print("Load model done...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    # model.load_state_dict(torch.load(args.model_path), strict=False)

    model.eval()
    model = model.to(device)

    print("start...")
    os.makedirs(args.output, exist_ok=True)

    if args.input_file:
        # process a single image
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

            # read image
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(device)
            # inference
            try:
                with torch.no_grad():
                    output = model(img)
            except Exception as error:
                print('Error', error, imgname)
            else:
                # save image
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round().astype(np.uint8)
                cv2.imwrite(os.path.join(args.output, f'{imgname}_ESRGAN.png'), output)
                print('Save:', idx, imgname)

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.6f} seconds")

def process_image(image_path, model, device, output_folder):
    """Process a single image."""
    imgname = os.path.splitext(os.path.basename(image_path))[0]
    print('Processing: ', imgname)
    try:
        # read image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            output = model(img)

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f'{imgname}_ESRGAN.png'), output)
        print('save:', imgname+'_ESRGAN.png')

    except Exception as error:
        print('Error', error, imgname)

if __name__ == '__main__':
    main()
