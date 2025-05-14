# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import os

# # --------------------------------------
# # 定义网络组件
# # --------------------------------------


# class ResidualBlockNoBN(nn.Module):

#     def __init__(self, num_feat=64, res_scale=1.0, pytorch_init=True):
#         super(ResidualBlockNoBN, self).__init__()
#         self.res_scale = res_scale
#         self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
#         if pytorch_init:
#             self._initialize_weights()

#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.conv2(out)
#         return identity + out * self.res_scale

#     def _initialize_weights(self):
#         for m in [self.conv1, self.conv2]:
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)


# def make_layer(block, num_blocks, **kwargs):
#     return nn.Sequential(*[block(**kwargs) for _ in range(num_blocks)])


# class Upsample(nn.Module):

#     def __init__(self, scale, num_feat):
#         super(Upsample, self).__init__()
#         modules = []
#         if scale == 2 or scale == 4:
#             for _ in range(int(scale / 2)):
#                 modules += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
#         elif scale == 3:
#             modules += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
#         else:
#             raise ValueError(f'Unsupported scale: {scale}')
#         self.upsample = nn.Sequential(*modules)

#     def forward(self, x):
#         return self.upsample(x)


# # class EDSR(nn.Module):

# #     def __init__(self,
# #                  num_in_ch,
# #                  num_out_ch,
# #                  num_feat=256,
# #                  num_block=32,
# #                  upscale=2,
# #                  res_scale=1,
# #                  img_range=255.,
# #                  rgb_mean=(0.4488, 0.4371, 0.4040)):
# #         super(EDSR, self).__init__()

# #         self.img_range = img_range
# #         self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

# #         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
# #         self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
# #         self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
# #         self.upsample = Upsample(upscale, num_feat)
# #         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

# #     def forward(self, x):
# #         self.mean = self.mean.type_as(x)

# #         x = (x - self.mean) * self.img_range
# #         x = self.conv_first(x)
# #         res = self.conv_after_body(self.body(x))
# #         res += x
# #         x = self.conv_last(self.upsample(res))
# #         x = x / self.img_range + self.mean
# #         return x


# class EDSR(nn.Module):

#     def __init__(self,
#                  num_in_ch,
#                  num_out_ch,
#                  num_feat=256,
#                  num_block=32,
#                  upscale=2,
#                  res_scale=1,
#                  img_range=255.,
#                  rgb_mean=(0.4488, 0.4371, 0.4040)):
#         super(EDSR, self).__init__()

#         self.img_range = img_range
#         self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
#         self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
#         self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

#         # 注意这里直接定义 nn.Sequential
#         modules = []
#         if upscale == 2 or upscale == 4:
#             for _ in range(int(upscale / 2)):
#                 modules += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
#         elif upscale == 3:
#             modules += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
#         else:
#             raise ValueError(f'Unsupported scale: {upscale}')
#         self.upsample = nn.Sequential(*modules)

#         self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

#     def forward(self, x):
#         self.mean = self.mean.type_as(x)

#         x = (x - self.mean) * self.img_range
#         x = self.conv_first(x)
#         res = self.conv_after_body(self.body(x))
#         res += x
#         x = self.conv_last(self.upsample(res))
#         x = x / self.img_range + self.mean
#         return x



# # --------------------------------------
# # 推理逻辑
# # --------------------------------------


# def load_image(img_path):
#     img = Image.open(img_path).convert('RGB')
#     transform = transforms.ToTensor()
#     return transform(img).unsqueeze(0)  # shape: [1, 3, H, W]


# def save_image(tensor, save_path):
#     tensor = tensor.squeeze().clamp(0, 1).cpu()
#     img = transforms.ToPILImage()(tensor)
#     img.save(save_path)


# def main():
#     # 模型参数
#     upscale = 2
#     model_path = r'D:\gracode\sr_models\Pic\EDSR\EDSR_Lx2_f256b32.pth'
#     input_img = r"D:\gracode\sr_data\pic\Set5\image_SRF_2\LR\img_001.png"
#     output_img = './sr.png'

#     # 加载模型
#     model = EDSR(num_in_ch=3, num_out_ch=3, num_feat=256, num_block=32, upscale=upscale)
#     state_dict = torch.load(model_path, map_location='cpu')
#     if 'params' in state_dict:
#         model.load_state_dict(state_dict['params'], strict=True)
#     else:
#         model.load_state_dict(state_dict, strict=True)
#     model.eval()

#     # 加载图片
#     lr = load_image(input_img)

#     # 推理
#     with torch.no_grad():
#         sr = model(lr)

#     # 保存输出
#     save_image(sr, output_img)
#     print(f"输出已保存到 {output_img}")


# if __name__ == '__main__':
#     main()




import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


# 模型结构（根据之前建议修改过结构）
class EDSR(nn.Module):

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=256,
                 num_block=32,
                 upscale=2,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlockNoBN(num_feat, res_scale=res_scale) for _ in range(num_block)])
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        modules = []
        for _ in range(int(upscale / 2)):
            modules += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        self.upsample = nn.Sequential(*modules)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.body(x)
        res = self.conv_after_body(res)
        res += x
        x = self.upsample(res)
        x = self.conv_last(x)
        x = x / self.img_range + self.mean
        return x


# 残差块（和训练时一致）
class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=256, res_scale=1.):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res * self.res_scale


# 载入模型
model = EDSR(num_in_ch=3, num_out_ch=3, num_feat=256, num_block=32, upscale=2)
state_dict = torch.load(r'D:\gracode\sr_models\Pic\EDSR\EDSR_Lx2_f256b32.pth', map_location="cpu")
model.load_state_dict(state_dict)
model.eval()


# 图像处理
def load_image(path):
    img = Image.open(path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor


# 推理 & 保存图像
def save_output(tensor, save_path):
    output = tensor.squeeze().clamp(0, 1).detach().cpu()
    image = transforms.ToPILImage()(output)
    image.save(save_path)


# 用例
lr_image = load_image(r"D:\gracode\sr_data\pic\Set5\image_SRF_2\LR\img_001.png")  # 替换为你的图像路径
with torch.no_grad():
    sr_image = model(lr_image)
save_output(sr_image, "/mnt/data/sr_result.png")
