from niqe import calculate_niqe
import cv2



img = cv2.imread(r'D:\gracode\sr_data\pic\Set5\image_SRF_4\LR\img_002.png').astype(np.float32)
niqe_value = calculate_niqe(img, crop_border=0, input_order='HWC', convert_to='y')
print(f'NIQE: {niqe_value:.2f}')
