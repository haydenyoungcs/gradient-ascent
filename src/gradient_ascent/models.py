from __future__ import annotations

import torch.nn as nn


DEFAULT_LAYER_NAMES = (
    "model.conv1",
    "model.layer1",
    "model.layer2",
    "model.layer3",
    "model.layer4",
    "model.fc",
)


class Net(nn.Module):
    """ResNet adapted for CIFAR-10: 3x3 stride-1 first conv and no initial maxpool."""

    def __init__(self, num_classes: int = 10, pretrained: bool = False, model_depth: int = 50):
        super().__init__()

        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
        from torchvision.models import resnet18, resnet34, resnet50

        resnet_map = {
            18: (resnet18, ResNet18_Weights),
            34: (resnet34, ResNet34_Weights),
            50: (resnet50, ResNet50_Weights),
        }
        if model_depth not in resnet_map:
            raise ValueError(f"model_depth must be one of {list(resnet_map.keys())}, got {model_depth}")

        resnet_fn, weights_class = resnet_map[model_depth]
        base = resnet_fn(weights=weights_class.DEFAULT if pretrained else None)

        if not pretrained:
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()

        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)
