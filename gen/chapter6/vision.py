import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=None, download=True)

x, label = dataset[0]
plt.imshow(x, cmap='gray')
plt.title(label)
# plt.show()
plt.close()

# Preprocess
transform = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

x, label = dataset[0]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for x, label in dataloader:
    print('x.shape:', x.shape)
    print('label.shape:', label.shape)
    break
