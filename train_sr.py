import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.div2k_dataset import DIV2KDataset
from models.network_swinir import SwinIR_MFFA

# 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 5   # 建议初期先小一点
patch_size = 48
scale = 2
lr = 1e-4
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

# 加载数据集
train_set = DIV2KDataset(
    hr_dir='datasets/DIV2K/DIV2K_train_HR',
    lr_dir='datasets/DIV2K/DIV2K_train_LR_bicubic/X2',
    patch_size=patch_size,
    scale=scale
)


def main():
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 构建模型
    model = SwinIR_MFFA(
        upscale=scale,
        img_size=patch_size * scale,
        in_chans=3,
        embed_dim=96,        # 要与你的 SACAModule(embed_dim) 保持一致
        depths=[6, 6, 6, 6],  # 可改小如 [2,2,2,2] 以减少显存
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    # 损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练主循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (lr_img, hr_img) in enumerate(train_loader):
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            sr = model(lr_img)
            loss = criterion(sr, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}] Average Loss: {total_loss / len(train_loader):.4f}")

        # 每轮保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, f'swinir_mffa_epoch{epoch+1}.pth'))


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # 可加可不加
    main()
