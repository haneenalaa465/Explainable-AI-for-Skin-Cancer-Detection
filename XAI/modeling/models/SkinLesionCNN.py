"""
CNN model for skin lesion classification based on the paper:
"Skin lesion classification of dermoscopic images using machine learning and convolutional neural network"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel


class SkinLesionCNN(BaseModel):
    """
    CNN architecture:
    - Conv2D(32) + ReLU + BN + MaxPool(3x3) + Dropout(0.25)
    - Conv2D(64) + ReLU + Conv2D(64) + ReLU + BN + MaxPool(2x2) + Dropout(0.25)
    - Conv2D(128) + ReLU + BN + Conv2D(128) + ReLU + BN + MaxPool(2x2) + Dropout(0.25)
    - FC(1024) + ReLU + BN + Dropout(0.5)
    - Output FC(num_classes)
    """

    @staticmethod
    def name():
        return "SkinLesionCNN"

    def __init__(self, num_classes=NUM_CLASSES):
        super(SkinLesionCNN, self).__init__()

        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Dropout(0.25)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Determine flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_out = self._forward_conv_blocks(dummy_input)
            self.flatten_dim = dummy_out.view(1, -1).size(1)

        # Fully connected layers
        self.fc_block = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(1024, num_classes)

    def _forward_conv_blocks(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x

    def forward(self, x):
        x = self._forward_conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return self.classifier(x)

    def extract_features(self, x):
        """
        Extract features from the penultimate layer for explainability.
        """
        x = self._forward_conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc_block(x)
