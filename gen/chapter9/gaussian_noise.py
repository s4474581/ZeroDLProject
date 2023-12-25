import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


def reverse_to_img(x):
    x = x * 255.
    # torch.clamp()⇨最小値、最大値の範囲に当てはめる
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    x = x.cpu()
    to_pil = transforms.ToPILImage()
    return to_pil(x)


def add_noise(x_0, t, betas):
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bar = alpha_bars[t]

    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps
    return x_t


if __name__ == '__main__':
    x = torch.randn(3, 64, 64)
    T = 1000
    betas = torch.linspace(1e-4, 2e-2, T)

    for t in range(T):
        beta = betas[t]
        eps = torch.randn_like(x)
        x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

    # Load image
    # 絶対パスの取得⇨os.path.abspath()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'flower.png')
    image = plt.imread(file_path)
    print('image.shape:', image.shape)

    # Preprocess
    preprocess = transforms.ToTensor()
    x = preprocess(image)
    print(x.shape)

    # x.clone()⇨同一デバイスに新しいテンソルを作成(微分流れ込む)・流れ込まないのはdetach()
    org_x = x.clone()

    T = 1000
    beta_start = 1e-4
    beta_end = 2e-2
    betas = torch.linspace(beta_start, beta_end, T)
    imgs = []

    for t in range(T):
        if t % 100 == 0:
            img = reverse_to_img(x)
            imgs.append(img)

        beta = betas[t]
        eps = torch.randn_like(x)
        x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

    # Show image(ノイズの強さ一覧)
    plt.figure(figsize=(15, 6))
    for i, img in enumerate(imgs[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f'Noise: {i * 100}')
        plt.axis('off')
    # plt.show()
    plt.close()

    x = org_x

    t = 900
    x_t = add_noise(x, t, betas)

    img = reverse_to_img(x_t)
    plt.imshow(img)
    plt.title(f'Noise: {t}')
    plt.axis('off')
    plt.show()
    plt.close()




