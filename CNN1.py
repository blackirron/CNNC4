import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True,
    num_workers = 2
)

data_iter = iter(train_loader)
images, labels = next(data_iter)

print("Dataset total size: ", len(train_dataset),"images")
print("Batch images shape: ", images.shape, " [batch_size, channels, height, width]")
print("Batch labels shape: ", labels.shape, " [batch_size]")
