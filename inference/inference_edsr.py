import argparse
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
        default=
        r'D:\gracode\srGra\models\Pic\EDSR\EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f.pth',  # Replace with your EDSR model path
        help='Path to the pretrained EDSR model')
    parser.add_argument('--input', type=str, default=None, help='Input image folder')
    parser.add_argument('--input_file', type=str, default=None, help='Input single image file')
    parser.add_argument('--output', type=str, default='results/EDSR', help='Output folder')
    args = parser.parse_args()

    print("Load model done...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set up model
    model = EDSR(num_in_ch=3, num_out_ch=3, num_feat=256, num_block=32, upscale=4)
    # model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    # model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')), strict=False)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))['params'], strict=False)

    model.eval()
    model = model.to(device)

    print("Start...")
    os.makedirs(args.output, exist_ok=True)

    if args.input_file:
        # Process a single image
        process_image(args.input_file, model, device, args.output)
    else:
        for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
            imgname = os.path.splitext(os.path.basename(path))[0]
            print('Processing: ', idx, imgname)
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
                print('Save:', idx, imgname)

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.6f} seconds")


def process_image(image_path, model, device, output_folder):
    """Process a single image."""
    imgname = os.path.splitext(os.path.basename(image_path))[0]
    print('Processing: ', imgname)
    try:
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
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