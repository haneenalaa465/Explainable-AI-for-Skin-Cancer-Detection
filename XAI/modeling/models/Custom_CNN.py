import torch
import torch.nn as nn


from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel


class CustomCNN(BaseModel):
    """
    A robust CNN Deep Learning and InceptionV3 model Techniques for Enhanced Skin Cancer Detection
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56,56,64
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28,28,128
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    @staticmethod
    def name():
        return "Custom_CNN"

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
