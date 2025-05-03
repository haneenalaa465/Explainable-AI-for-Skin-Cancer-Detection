import torch
import torch.nn as nn
from torchvision.models.efficientnet import EfficientNet_B5_Weights, efficientnet_b5

from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel

class SkinEfficientNetB5(BaseModel):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True):
        super(SkinEfficientNetB5, self).__init__()

        self.backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get the number of features from the model
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    @staticmethod
    def name():
        return "SkinEfficientNetB5"

    def forward(self, x):
        return self.backbone(x)