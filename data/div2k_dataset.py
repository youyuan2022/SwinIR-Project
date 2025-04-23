import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=48, scale=2):
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        self.patch_size = patch_size
        self.scale = scale
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_paths[idx]).convert('RGB')
        lr = Image.open(self.lr_paths[idx]).convert('RGB')

        # 获取尺寸并随机裁剪 patch
        w, h = lr.size
        ps = self.patch_size
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)

        lr_patch = lr.crop((x, y, x + ps, y + ps))
        hr_patch = hr.crop((x * self.scale, y * self.scale, (x + ps) * self.scale, (y + ps) * self.scale))

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)
