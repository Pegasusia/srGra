import torch
from basicsr.archs.edsr_arch import EDSR

model = EDSR(num_in_ch=3, num_out_ch=3, num_feat=256, num_block=32, upscale=2)
ckpt = torch.load(r'D:\gracode\sr_models\Pic\EDSR\EDSR_Lx2_f256b32.pth', map_location='cpu')
model.load_state_dict(ckpt['params'], strict=False)
model.eval()

x = torch.rand(1, 3, 64, 64)  # dummy input
with torch.no_grad():
    y = model(x)
print(y.shape)  # 应该是 (1, 3, 128, 128)
