import os
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import utils
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 检查GPU是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

class CVAE(nn.Module):
    def __init__(self, feature_size, class_size, latent_size):
        super(CVAE, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(feature_size + class_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc2_mu = nn.Linear(256, latent_size)
        self.fc2_log_std = nn.Linear(256, latent_size)

        self.fc3 = nn.Sequential(
            nn.Linear(latent_size + class_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.fc4_out = nn.Linear(512, feature_size)

    def encode(self, x, y):
        h1 = self.fc1(torch.cat([x, y], dim=1))
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z, y):
        h3 = self.fc3(torch.cat([z, y], dim=1))
        recon = torch.sigmoid(self.fc4_out(h3))
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, y):
        mu, log_std = self.encode(x, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, y)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std, beta=1) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        loss = recon_loss + beta * kl_loss
        return loss


if __name__ == '__main__':
    epochs = 54
    batch_size = 32
    image_size = 197
    num_channels = 3
    beta = 0.01

    utils.make_dir("/root/autodl-tmp/ourDCGAN/cvae/img/cvae")
    utils.make_dir("/root/autodl-tmp/ourDCGAN/cvae/model_weights/cvae")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化输入数据
    ])

    train_data = ImageFolder(root='/root/autodl-tmp/VAE-master/data/308', transform=transform)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 创建模型实例，并立即将其移动到GPU上
    cvae = CVAE(feature_size=num_channels * image_size * image_size,
                class_size=len(train_data.classes),
                latent_size=10).to(device)

    optimizer = optim.Adam(cvae.parameters(), lr=0.0002)  # 尝试提高学习率，加速收敛
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_loss = float('inf')
    patience = 10
    no_improve_epochs = 0

    for epoch in range(epochs):
        train_loss = 0
        i = 0
        for batch_id, (data, target) in enumerate(data_loader):
            img, label = data.to(device), target.to(device)  # 移动数据到GPU
            inputs = img.view(img.size(0), -1)
            y = utils.to_one_hot(label, num_class=len(train_data.classes)).to(device)  # 移动标签到GPU
            recon, mu, log_std = cvae(inputs, y)
            loss = cvae.loss_function(recon, inputs, mu, log_std, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            i += 1

            if batch_id % 100 == 0:
                print(
                    f"Epoch[{epoch + 1}/{epochs}], Batch[{batch_id + 1}/{len(data_loader)}], batch_loss:{loss.item():.6f}")

        avg_loss = train_loss / i
        print(f"======>epoch:{epoch + 1},\t epoch_average_batch_loss:{avg_loss:.6f}============\n")

        scheduler.step(avg_loss)

        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break

        # 生成图像的代码保持不变
        if epoch % 10 == 0:
            for action in train_data.classes:
                action_dir = f"/root/autodl-tmp/ourDCGAN/cvae/img/cvae/generate_{action}"
                utils.make_dir(action_dir)
                for idx in range(batch_size):
                    generated_img = utils.to_img(recon[idx].detach().view(num_channels, image_size, image_size))
                    torchvision.utils.save_image(generated_img,
                                                 os.path.join(action_dir, f"epoch{epoch + 1}_iter{idx}.png"))
                    print(f"save: {os.path.join(action_dir, f'epoch{epoch + 1}_iter{idx}.png')}\n")

    utils.save_model(cvae, "/root/autodl-tmp/ourDCGAN/cvae/model_weights/cvae/cvae_weights.pth")