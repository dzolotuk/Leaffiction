"""
model.py -- LeafDiseaseClassifier architecture: EfficientNet-B0 backbone
(frozen, pretrained on ImageNet) with a replaced classification head.
"""

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn


class LeafDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        for param in self.model.features.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


def create_model(num_classes):
    return LeafDiseaseClassifier(num_classes)


def unfreeze_last_blocks(model: LeafDiseaseClassifier, n_blocks: int = 2):
    blocks = list(model.model.features.children())
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
