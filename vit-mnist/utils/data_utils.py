import torch
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root='data', train=True, download=download, transform=transform)
    test_set = datasets.MNIST(root='data', train=False, download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
