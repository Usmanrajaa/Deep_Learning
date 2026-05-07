import torch
from torchvision import datasets, transforms

def get_mnist_loader(batch_size, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))   # scale to [-1, 1]
    ])
    dataset = datasets.MNIST(root='data', train=True, download=download, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
