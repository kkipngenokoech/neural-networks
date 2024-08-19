from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch

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

device = ("cuda" if torch.cuda.is_available() else "cpu")

class CIFAR10Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * 3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
            torch.nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# INSTANTIATING THE MODEL
model = CIFAR10Model().to(device)
print(model)


# OPTIMIZER AND LOSS FUNCTION
def train_loop(dataLoader, model, loss_fn, optimizer):
    size = len(dataLoader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataLoader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataLoader, model, loss_fn):
    size = len(dataLoader.dataset)
    num_batches = len(dataLoader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataLoader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

epochs = 10
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)