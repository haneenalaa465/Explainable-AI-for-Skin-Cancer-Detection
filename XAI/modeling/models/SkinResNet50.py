import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel


def replace_relu_with_non_inplace(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU):
            print(child)
            setattr(module, name, torch.nn.ReLU(inplace=False))
            print(child)
        else:
            replace_relu_with_non_inplace(child)


class FineTunedResNet50(BaseModel):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True):
        super(FineTunedResNet50, self).__init__()

        # Load pretrained ResNet50
        self.backbone = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1,
        )

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
