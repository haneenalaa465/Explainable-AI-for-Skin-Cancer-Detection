import torch
import torch.nn as nn
import torchvision.models as models

from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel


class ModifiedInceptionV3(BaseModel):
    """
    A robust CNN Deep Learning and InceptionV3 model Techniques for Enhanced Skin Cancer Detection
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(ModifiedInceptionV3, self).__init__()
        self.base_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.base_model.aux_logits = False
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    @staticmethod
    def name():
        return "ModifiedInceptionV3"

    @staticmethod
    def inputSize():
        return (299, 299)

    def forward(self, x):
        return self.base_model(x)
