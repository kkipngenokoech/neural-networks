from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# LOADING THE CIFAR DATASET
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# CREATING DATA LOADERS
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
