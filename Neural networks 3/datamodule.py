# Load the FashionMNIST Dataset with PyTorch - this is using pre-loaded datasets from torchvision.
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize

# these are grey scale images
training_data = datasets.FashionMNIST(
    root = "data",
    download=True,
    train=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    download=True,
    train=False,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for index in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, index)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.savefig("figure.png")

# RGB images
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

labels_map = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}

# Displaying only one RGB image
img, label = training_data[1000]
print(img.shape, label)
img_resized = resize(img, (1028, 1028))
print(img_resized.shape, label)
plt.figure(figsize=(8, 8))
plt.imshow(img_resized.permute(1, 2, 0))
plt.title(labels_map[label])
plt.axis("off")
plt.savefig("RGBimage.png")

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0))
plt.savefig("figureRGB.png")



# using cifar100 dataset
training_data = datasets.CIFAR100(
    train=True,
    root="data",
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR100(
    train=False,
    root="data",
    download=True,
    transform=ToTensor()
)


labels_map = {
    0: "Apple", 1: "Aquarium fish", 2: "Baby", 3: "Bear", 4: "Beaver",
    5: "Bed", 6: "Bee", 7: "Beetle", 8: "Bicycle", 9: "Bottle",
    10: "Bowl", 11: "Boy", 12: "Bridge", 13: "Bus", 14: "Butterfly",
    15: "Camel", 16: "Can", 17: "Castle", 18: "Caterpillar", 19: "Cattle",
    20: "Chair", 21: "Chimpanzee", 22: "Clock", 23: "Cloud", 24: "Cockroach",
    25: "Couch", 26: "Crab", 27: "Crocodile", 28: "Cup", 29: "Dinosaur",
    30: "Dolphin", 31: "Elephant", 32: "Flatfish", 33: "Forest", 34: "Fox",
    35: "Girl", 36: "Hamster", 37: "House", 38: "Kangaroo", 39: "Keyboard",
    40: "Lamp", 41: "Lawn-mower", 42: "Leopard", 43: "Lion", 44: "Lizard",
    45: "Lobster", 46: "Man", 47: "Maple", 48: "Motorcycle", 49: "Mountain",
    50: "Mouse", 51: "Mushroom", 52: "Oak", 53: "Orange", 54: "Orchid",
    55: "Otter", 56: "Palm", 57: "Pear", 58: "Pickup truck", 59: "Pine",
    60: "Plain", 61: "Plate", 62: "Poppy", 63: "Porcupine", 64: "Possum",
    65: "Rabbit", 66: "Raccoon", 67: "Ray", 68: "Road", 69: "Rocket",
    70: "Rose", 71: "Sea", 72: "Seal", 73: "Shark", 74: "Shrew",
    75: "Skunk", 76: "Skyscraper", 77: "Snail", 78: "Snake", 79: "Spider",
    80: "Squirrel", 81: "Streetcar", 82: "Sunflower", 83: "Sweet pepper", 84: "Table",
    85: "Tank", 86: "Telephone", 87: "Television", 88: "Tiger", 89: "Tractor",
    90: "Train", 91: "Trout", 92: "Tulip", 93: "Turtle", 94: "Wardrobe",
    95: "Whale", 96: "Willow", 97: "Wolf", 98: "Woman", 99: "Worm"
}


image, label = training_data[10000]
print(image.shape, label)
img_resized = resize(image, (10028, 10028))
print(img_resized.shape, label)
plt.figure(figsize=(8, 8))
plt.imshow(img_resized.permute(1, 2, 0))
plt.title(labels_map[label])
plt.axis("off")
plt.savefig("cifar100.png") 


# LOADING CUSTOM DATASETS
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

data_directory = "data/me"
annotation = "data/me/me.csv"

class CustomImageDataset(Dataset):
    def __init__(self,annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

meData = CustomImageDataset(annotation, data_directory)
image, label = meData[3]
print(image.shape, label)
plt.figure(figsize=(8, 8))
plt.imshow(image.permute(1, 2, 0))
plt.title(label)
plt.axis("off")
plt.savefig("me.png")