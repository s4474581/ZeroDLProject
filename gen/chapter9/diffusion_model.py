import math
from functools import partial

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm as std_tqdm
import matplotlib.pyplot as plt

# tqdmの表示揺れ問題解消
tqdm = partial(std_tqdm, dynamic_ncols=True)


def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[::2] = torch.sin(t / div_term[::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v


def pos_encoding(timesteps, output_dim, device='cpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch))

    def forward(self, x, t):
        N, C, _, _ = x.shape
        t = self.mlp(t)
        t = t.view(N, C, 1, 1)
        y = self.convs(x + t)
        return y


class UNet(nn.Module):
    def __init__(self, out_ch=1, time_dim=100):
        super().__init__()
        self.time_dim = time_dim

        self.down1 = ConvBlock(1, 64, time_dim)
        self.down2 = ConvBlock(64, 128, time_dim)
        self.bot1 = ConvBlock(128, 256, time_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_dim)
        self.out = nn.Conv2d(64, out_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, timesteps):
        v = pos_encoding(timesteps, self.time_dim, x.device)

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x


class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        alpha_bar = self.alpha_bars[t]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        noise = torch.randn_like(x_0, device=self.device)

        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def denoise(self, model, x, t):
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t - 1]

        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()

        with torch.no_grad():
            y = model(x, t)
        model.train()

        eps = torch.randn_like(x, device=self.device)
        eps[t == 0] = 0

        mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * y) / torch.sqrt(alpha)
        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        return mu + eps * std

    def reverse_to_img(self, x):
        x = x * 255.
        # torch.clamp()⇨最小値、最大値の範囲に当てはめる
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 28, 28)):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps)[::-1]):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)
            x = torch.clamp(x, -1., 1.)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images


def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            i += 1
    plt.show()


if __name__ == '__main__':
    # HP
    img_size = 128
    batch_size = 128
    num_timesteps = 1000
    epochs = 10
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)])

    dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # diffuser⇨拡散
    diffuser = Diffuser(num_timesteps, device=device)
    model = UNet()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        loss_sum = 0.
        cnt = 0

        for images, labels in tqdm(dataloader):
            optimizer.zero_grad()
            x = images.to(device)
            t = torch.randint(0, num_timesteps, (len(x),), device=device)

            x_noisy, noise = diffuser.add_noise(x, t)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cnt += 1

        loss_avg = loss_sum / cnt
        losses.append(loss_avg)
        print(f'Epoch: {epoch}, Loss: {loss_avg}')

    torch.save(model, 'diff_model_weight.pth')

    # Plot
    plt.plot(losses)
    plt.show()
    plt.close()

    # Generate samples
    images = diffuser.sample(model)
    show_images(images)






