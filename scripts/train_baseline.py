from xml.parsers.expat import model
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

@torch.no_grad()
def test_accuracy(net, testloader, target_class=None):
    net.eval()
    correct = 0
    total = 0
    for inputs, labels in testloader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        if target_class is not None:
            mask = (labels == target_class)
            correct += (predicted[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
        
        else:
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()
    return correct / total if total > 0 else 0.0

def main():

    num_epochs = 10
    batch_size = 64

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    acc_history = []

    for epoch in range(num_epochs):  # Loops over dataset multiple times
        
        epoch_loss = 0.0
        for inputs, labels in trainloader:

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        acc = test_accuracy(net, testloader)
        acc_history.append(acc)
        print(f"epoch {epoch+1}/{num_epochs} loss {epoch_loss/len(trainloader):.3f} test_acc {acc:.3f}")

    print('Finished Training')

    os.makedirs("out", exist_ok=True)
    torch.save(net.state_dict(), "out/baseline_net.pt")
    print("Saved to out/baseline_net.pt")

    plt.plot(range(1, num_epochs+1), acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("CIFAR-10 baseline accuracy vs epoch")
    plt.savefig("out/acc_vs_epoch.png")
    print("Saved plot to out/acc_vs_epoch.png")

    net = Net()
    net.load_state_dict(torch.load("out/baseline_net.pt"))

    target_label = 6  # 'frog' class index is 6
    forget_indices = [idx for idx, lbl in enumerate(trainset.targets) if lbl == target_label]

    forget_subset = Subset(trainset, forget_indices)
    forget_loader = DataLoader(forget_subset, batch_size=batch_size, shuffle=True)

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=1e-6, momentum=0.0)

    total_acc_history = []
    classwise_acc_history = np.zeros((num_epochs, 10))

    for epoch in range(num_epochs):

        epoch_loss = 0.0

        for inputs, labels in forget_loader:

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            (-loss).backward()  # Gradient ascent
            optimizer.step()

            epoch_loss += loss.item()

        total_acc = test_accuracy(net, testloader) 
        for cls in range(10):
            acc = test_accuracy(net, testloader, target_class=cls)
            classwise_acc_history[epoch, cls] = acc

        total_acc_history.append(total_acc)

        print(f"[GA epoch {epoch+1}] loss {epoch_loss/len(forget_loader):.3f} total_acc {total_acc:.3f}")
    
    print('Finished Unlearning')
    
    torch.save(net.state_dict(), "out/unlearned_net.pt")
    print("Saved to out/unlearned_net.pt")

    initial_acc = classwise_acc_history[0]
    percent_change = 100 * (classwise_acc_history - initial_acc) / initial_acc

    plt.figure()
    for cls in range(10):
        plt.plot(range(1, num_epochs + 1), percent_change[:, cls], label=f"Class {cls}")
    plt.xlabel("Unlearning Epoch")
    plt.ylabel("Accuracy Change (%)")
    plt.title("Relative Change in Class Accuracy During Unlearning")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend(loc="lower left", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig("out/classwise_percent_change.png")
    print("Saved plot to out/classwise_percent_change.png")

if __name__ == "__main__":
    main()
