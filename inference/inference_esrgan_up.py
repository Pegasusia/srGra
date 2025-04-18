import cv2
import numpy as np
from skimage import img_as_float
from skimage.color import rgb2gray
# from skimage.metrics import niqe, brisque, piqe


def calculate_psnr(img1, img2):
    """计算两张图片的 PSNR 值"""

    # 确保两张图片的大小相同
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # 将图像转换为浮动类型，避免整数溢出
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')  # 如果 MSE 为 0，PSNR 为无穷大
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# def compute_niqe_improvement(lr_image, sr_image):
#     lr = img_as_float(cv2.cvtColor(lr_image, cv2.COLOR_BGR2GRAY))
#     sr = img_as_float(cv2.cvtColor(sr_image, cv2.COLOR_BGR2GRAY))

#     score_lr = niqe(lr)
#     score_sr = niqe(sr)

#     # 分数越低越好，因此反向计算提升
#     improvement = (score_lr - score_sr) / score_lr * 100
#     return round(improvement, 2), round(score_lr, 2), round(score_sr, 2)


# def preprocess_gray(image):
#     if image.shape[-1] == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image
#     return img_as_float(gray)


# def compute_niqe_improvement(lr_image, sr_image):
#     lr = preprocess_gray(lr_image)
#     sr = preprocess_gray(sr_image)

#     score_lr = niqe(lr)
#     score_sr = niqe(sr)

#     improvement = (score_lr - score_sr) / score_lr * 100
#     return round(improvement, 2), round(score_lr, 2), round(score_sr, 2)


# def compute_brisque_improvement(lr_image, sr_image):
#     lr = preprocess_gray(lr_image)
#     sr = preprocess_gray(sr_image)

#     score_lr = brisque(lr)
#     score_sr = brisque(sr)

#     improvement = (score_lr - score_sr) / score_lr * 100
#     return round(improvement, 2), round(score_lr, 2), round(score_sr, 2)


# def compute_piqe_improvement(lr_image, sr_image):
#     lr = preprocess_gray(lr_image)
#     sr = preprocess_gray(sr_image)

#     score_lr = piqe(lr)
#     score_sr = piqe(sr)

#     improvement = (score_lr - score_sr) / score_lr * 100
#     return round(improvement, 2), round(score_lr, 2), round(score_sr, 2)


# from basicsr.metrics.niqe import calculate_niqe
# import cv2

# img = cv2.imread(r'D:\gracode\sr_data\pic\Set5\image_SRF_4\HR\img_001.png').astype(np.float32)
# niqe_value = calculate_niqe(img, crop_border=0, input_order='HWC', convert_to='y')
# print(f'NIQE: {niqe_value:.2f}')
