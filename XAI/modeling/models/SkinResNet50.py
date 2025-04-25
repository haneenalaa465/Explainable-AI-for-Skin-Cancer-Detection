import torch
import torch.nn as nn
import torchvision.models as models

from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel


class FineTunedResNet50(BaseModel):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True):
        super(FineTunedResNet50, self).__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)

        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    @staticmethod
    def name():
        return "FineTunedResNet50"

    def forward(self, x):
        return self.backbone(x)
