import torch
import torch.nn as nn

from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel

class SkinCancerCNN(BaseModel):
    """
    An Interpretable Skin Cancer Classification Using Optimized Convolutional Neural Network for a Smart Healthcare System
    
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(SkinCancerCNN, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (50, 50, 32)
            nn.Dropout(p=0.25)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (25, 25, 64)
            nn.Dropout(p=0.25)
        )
        
        self.flatten = nn.Flatten()  # (25 * 25 * 64) = 40000
        self.dense1 = nn.Sequential(
            nn.Linear(25 * 25 * 64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.dense2 = nn.Linear(128, num_classes)

    @staticmethod
    def name():
        return "Custom_CNN_2"
    
    @staticmethod
    def inputSize():
        return (100, 100)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x