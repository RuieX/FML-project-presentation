import torch
from torchvision import models
import torch.nn as nn

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, freeze_base=True):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all base layers by default
        if freeze_base:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))
