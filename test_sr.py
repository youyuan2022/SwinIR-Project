import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from models.network_swinir import SwinIR_MFFA
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'checkpoints/swinir_mffa_epoch5.pth'
lr_dir = 'testsets/Set5/LR_bicubic/X2'
hr_dir = 'testsets/Set5/HR'
save_dir = 'results/SwinIR_MFFA_Set5_x2'
scale = 2
os.makedirs(save_dir, exist_ok=True)

# 加载模型（参数与训练完全一致）
model = SwinIR_MFFA(
    upscale=scale,
    img_size=96,
    in_chans=3,
    embed_dim=96,
    depths=[6, 6, 6, 6],
    num_heads=[6, 6, 6, 6],
    window_size=8,
    mlp_ratio=2.,
    upsampler='pixelshuffle',
    resi_connection='1conv'
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 工具
to_tensor = ToTensor()
to_pil = ToPILImage()

# 读取图像列表
lr_list = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
hr_list = sorted(glob.glob(os.path.join(hr_dir, '*.png')))

psnr_total, ssim_total = 0, 0

for lr_path, hr_path in zip(lr_list, hr_list):
    name = os.path.basename(lr_path)
    lr_img = Image.open(lr_path).convert('RGB')
    hr_img = Image.open(hr_path).convert('RGB')

    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    sr_tensor = torch.clamp(sr_tensor, 0, 1)

    # 保存 SR 图像
    sr_img = to_pil(sr_tensor.squeeze(0).cpu())
    sr_img.save(os.path.join(save_dir, name))

    # 计算 PSNR / SSIM
    sr_np = np.array(sr_img)
    hr_np = np.array(hr_img)
    psnr_val = psnr(hr_np, sr_np, data_range=255)
    ssim_val = ssim(hr_np, sr_np, data_range=255, channel_axis=-1)

    print(f'{name}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}')
    psnr_total += psnr_val
    ssim_total += ssim_val

# 平均指标
n = len(lr_list)
print(f'\n==> Average PSNR: {psnr_total/n:.2f} dB, SSIM: {ssim_total/n:.4f}')
