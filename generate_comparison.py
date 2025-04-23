import os
from PIL import Image
import numpy as np

# 目录配置
lr_dir = 'testsets/Set5/LR_bicubic/X2'
sr_dir = 'results/SwinIR_MFFA_Set5_x2'
hr_dir = 'testsets/Set5/HR'
save_dir = 'results/comparisons'

os.makedirs(save_dir, exist_ok=True)

# 统一图像顺序
image_names = sorted(os.listdir(lr_dir))

for name in image_names:
    # 加载图像
    lr_img = Image.open(os.path.join(lr_dir, name)).convert('RGB').resize((512, 512), Image.BICUBIC)
    sr_img = Image.open(os.path.join(sr_dir, name)).convert('RGB')
    hr_name = name.replace('x2', '')  # "babyx2.png" -> "baby.png"
    hr_img = Image.open(os.path.join(hr_dir, hr_name)).convert('RGB')

    # 大小统一
    hr_img = hr_img.resize((512, 512), Image.BICUBIC)
    sr_img = sr_img.resize((512, 512), Image.BICUBIC)

    # 拼接成一张图
    combined = Image.new('RGB', (512 * 3, 512))
    combined.paste(lr_img, (0, 0))
    combined.paste(sr_img, (512, 0))
    combined.paste(hr_img, (512 * 2, 0))

    # 保存
    save_path = os.path.join(save_dir, name)
    combined.save(save_path)
    print(f"Saved comparison image: {save_path}")
