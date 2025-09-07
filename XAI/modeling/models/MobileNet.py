import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenet import MobileNet_V2_Weights

from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel


class MobileNetV2(BaseModel):
    """
    Towards Domain-Specific Explainable AI: Model Interpretation of a Skin Image Classifier using a Human Approach
    """

    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True):
        super(MobileNetV2, self).__init__()

        self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    @staticmethod
    def name():
        return "MobileNetV2"

    def forward(self, x):
        return self.backbone(x)
