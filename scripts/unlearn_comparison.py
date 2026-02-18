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


def train_model(net, trainloader, testloader, num_epochs, lr=0.001, momentum=0.9, save_path=None, log_prefix=""):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    acc_history = []
    for epoch in range(num_epochs):
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
        print(f"{log_prefix}epoch {epoch+1}/{num_epochs} loss {epoch_loss/len(trainloader):.3f} test_acc {acc:.3f}")

    if save_path is not None:
        torch.save(net.state_dict(), save_path)
        print(f"Saved to {save_path}")

    return acc_history


def main():
    num_epochs = 10
    batch_size = 64
    target_label = 6  # class 6 to exclude (frog)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    os.makedirs("out", exist_ok=True)

    # -----------------------
    # 1) Baseline model train
    # -----------------------
    baseline_net = Net()
    baseline_acc_history = train_model(
        baseline_net,
        trainloader,
        testloader,
        num_epochs=num_epochs,
        lr=0.001,
        momentum=0.9,
        save_path="out/baseline_net.pt",
        log_prefix="baseline ",
    )

    plt.figure()
    plt.plot(range(1, num_epochs + 1), baseline_acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("CIFAR-10 baseline accuracy vs epoch")
    plt.tight_layout()
    plt.savefig("out/acc_vs_epoch.png")
    print("Saved plot to out/acc_vs_epoch.png")

    # ---------------------------------------------------------
    # 2) Train comparison model on data excluding class == 6
    #    (still 10-way classifier, just never sees class 6)
    # ---------------------------------------------------------
    keep_indices = [idx for idx, lbl in enumerate(trainset.targets) if lbl != target_label]
    keep_subset = Subset(trainset, keep_indices)
    keep_loader = DataLoader(keep_subset, batch_size=batch_size, shuffle=True, num_workers=0)

    excl6_net = Net()
    excl6_acc_history = train_model(
        excl6_net,
        keep_loader,
        testloader,
        num_epochs=num_epochs,
        lr=0.001,
        momentum=0.9,
        save_path="out/excluding_class6_net.pt",
        log_prefix="excl6 ",
    )

    plt.figure()
    plt.plot(range(1, num_epochs + 1), excl6_acc_history, label="Trained without class 6")
    plt.plot(range(1, num_epochs + 1), baseline_acc_history, label="Baseline (all data)")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Baseline vs trained-without-class-6")
    plt.legend()
    plt.tight_layout()
    plt.savefig("out/baseline_vs_excl6_acc.png")
    print("Saved plot to out/baseline_vs_excl6_acc.png")

    # --------------------------------------------
    # 3) Unlearning from baseline via grad ascent
    # --------------------------------------------
    unlearn_net = Net()
    unlearn_net.load_state_dict(torch.load("out/baseline_net.pt"))

    forget_indices = [idx for idx, lbl in enumerate(trainset.targets) if lbl == target_label]
    forget_subset = Subset(trainset, forget_indices)
    forget_loader = DataLoader(forget_subset, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    unlearn_net.train()
    optimizer = optim.SGD(unlearn_net.parameters(), lr=1e-6, momentum=0.0)

    total_acc_history = []
    classwise_acc_history = np.zeros((num_epochs, 10))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in forget_loader:
            optimizer.zero_grad()
            outputs = unlearn_net(inputs)
            loss = criterion(outputs, labels)
            (-loss).backward()  # gradient ascent on target class data
            optimizer.step()
            epoch_loss += loss.item()

        total_acc = test_accuracy(unlearn_net, testloader)
        for cls in range(10):
            classwise_acc_history[epoch, cls] = test_accuracy(unlearn_net, testloader, target_class=cls)

        total_acc_history.append(total_acc)
        print(f"[GA epoch {epoch+1}] loss {epoch_loss/len(forget_loader):.3f} total_acc {total_acc:.3f}")

    torch.save(unlearn_net.state_dict(), "out/unlearned_net.pt")
    print("Saved to out/unlearned_net.pt")

    initial_acc = classwise_acc_history[0]
    denom = np.where(initial_acc == 0, 1e-12, initial_acc)  # avoid divide-by-zero
    percent_change = 100 * (classwise_acc_history - initial_acc) / denom

    plt.figure()
    for cls in range(10):
        plt.plot(range(1, num_epochs + 1), percent_change[:, cls], label=f"Class {cls}")
    plt.xlabel("Unlearning Epoch")
    plt.ylabel("Accuracy Change (%)")
    plt.title("Relative Change in Class Accuracy During Unlearning")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.legend(loc="lower left", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig("out/classwise_percent_change.png")
    print("Saved plot to out/classwise_percent_change.png")

    # ---------------------------------------------------------
    # 4) Optional: quick comparison of class-6 accuracy at end
    # ---------------------------------------------------------
    excl6_cls6_acc = test_accuracy(excl6_net, testloader)
    unlearn_cls6_acc = test_accuracy(unlearn_net, testloader)
    base_cls6_acc = test_accuracy(baseline_net, testloader)

    excl6_cls6_acc_6 = test_accuracy(excl6_net, testloader, target_class=target_label)
    unlearn_cls6_acc_6 = test_accuracy(unlearn_net, testloader, target_class=target_label)
    base_cls6_acc_6 = test_accuracy(baseline_net, testloader, target_class=target_label)

    print(f"class 6 test acc baseline:   {base_cls6_acc:.3f}")
    print(f"class 6 test acc excl6-train: {excl6_cls6_acc:.3f}")
    print(f"class 6 test acc unlearned:   {unlearn_cls6_acc:.3f}")

    print(f"class 6 test acc baseline (cls6-only):   {base_cls6_acc_6:.3f}")
    print(f"class 6 test acc excl6-train (cls6-only): {excl6_cls6_acc_6:.3f}")
    print(f"class 6 test acc unlearned (cls6-only):   {unlearn_cls6_acc_6:.3f}")


if __name__ == "__main__":
    main()
