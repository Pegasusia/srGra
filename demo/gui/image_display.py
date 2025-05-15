from PyQt5.QtGui import QPixmap
from PIL import Image
import os


def display_images(gui, input_image_path, output_image_path):
    """显示输入和输出图片"""
    pix_in = QPixmap(input_image_path).scaled(gui.input_image_label.size())
    gui.input_image_label.setPixmap(pix_in)

    with Image.open(input_image_path) as img:
        w, h = img.size
    size = os.path.getsize(input_image_path) / 1024
    gui.input_image_info.setText(f"输入图片信息: {w}x{h}, {size:.2f} KB")

    pix_out = QPixmap(output_image_path).scaled(gui.output_image_label.size())
    gui.output_image_label.setPixmap(pix_out)

    with Image.open(output_image_path) as img:
        w, h = img.size
    size = os.path.getsize(output_image_path) / 1024
    gui.output_image_info.setText(f"分辨率提升后的图片信息: {w}x{h}, {size:.2f} KB")
