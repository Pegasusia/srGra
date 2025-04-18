import argparse
import cv2
import glob
import numpy as np
import os
import torch
import time

from tqdm import tqdm
from inference_esrgan_up import *

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.metrics.niqe import calculate_niqe
# from inference_niqe import calculate_niqe_2


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
        r'D:\gracode\sr_models\Pic\ESRGAN\ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default=None, help='input image folder')
    parser.add_argument('--input_file', type=str, default=r'D:\Resource\Picture\ya\ya.png', help='input single image file')
    parser.add_argument('--output', type=str, default= r'D:\gracode\sr_results\11', help='output folder')
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
        # OpenCV
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
        print('save:', imgname + '_ESRGAN.png')

        # enhanced image
        enhanced_image_path = os.path.join(output_folder, f'{imgname}_ESRGAN.png').replace(os.sep, '/')

        # calculate PSNR
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        enhanced_image = cv2.imread(enhanced_image_path, cv2.IMREAD_COLOR)
        psnr_enhanced = calculate_psnr(original_image, enhanced_image)  # 增强图片的 PSNR
        print(f'PSNR (Enhanced): {psnr_enhanced:.2f} dB')

        # # NIQE
        # imp, before, after = compute_niqe_improvement(original_image, enhanced_image)
        # print(f"NIQE: {imp}% (from {before} to {after})")

        # # BRISQUE
        # imp, before, after = compute_brisque_improvement(original_image, enhanced_image)
        # print(f"BRISQUE: {imp}% (from {before} to {after})")

        # # PIQE
        # imp, before, after = compute_piqe_improvement(original_image, enhanced_image)
        # print(f"PIQE: {imp}% (from {before} to {after})")

        # NIQE
        original_image = cv2.imread(image_path)
        enhanced_image = cv2.imread(enhanced_image_path)
        niqe_value_original = calculate_niqe(original_image, crop_border=0, input_order='HWC', convert_to='y')
        niqe_value_enhanced = calculate_niqe(enhanced_image, crop_border=0, input_order='HWC', convert_to='y')
        print(f'NIQE (Original): {niqe_value_original:.2f}')
        print(f'NIQE (Enhanced): {niqe_value_enhanced:.2f}')
        print(f'NIQE Improvement: {(niqe_value_original - niqe_value_enhanced) / niqe_value_original:.2f}')


        # niqe_value_2 = calculate_niqe_2(enhanced_image_path)
        # print(f'NIQE 2 (Enhanced): {niqe_value_2:.2f}')



        # print(f"NIQE: {imp}% (from {before} to {after})")


    except Exception as error:
        print('Error', error, imgname)



if __name__ == '__main__':
    main()
