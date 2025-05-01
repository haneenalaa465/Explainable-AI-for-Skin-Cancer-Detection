"""
CNN model for skin lesion classification based on the paper:
"Skin lesion classification of dermoscopic images using machine learning and convolutional neural network"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from XAI.config import NUM_CLASSES, MODEL_INPUT_SIZE
from XAI.modeling.models.Base_Model import BaseModel

class SkinLesionCNN(BaseModel):
    """
    CNN architecture according to Table 2:
    - Conv2D(32, 3x3) + ReLU + BN + MaxPool(3x3) + Dropout(0.25)
    - Conv2D(64, 3x3) + ReLU + Conv2D(64, 3x3) + ReLU + BN + MaxPool(2x2) + Dropout(0.25)
    - Conv2D(128, 3x3) + ReLU + BN + Conv2D(128, 3x3) + ReLU + BN + MaxPool(2x2) + Dropout(0.25)
    - Flatten + FC(1024) + ReLU + BN + Dropout(0.5)
    - Output FC(7) with softmax activation
    """
    @staticmethod
    def name():
        return "SkinLesionCNN"

    @staticmethod
    def inputSize():
        return MODEL_INPUT_SIZE  # Use the model input size from config

    def __init__(self, num_classes=NUM_CLASSES):
        super(SkinLesionCNN, self).__init__()
        
        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'),  # same padding
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=3),  # Reduction by factor of 3
            nn.Dropout(0.25)
        )
        
        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),  # same padding
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),  # same padding
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # reduce spatial dimensions
            nn.Dropout(0.25)
        )
        
        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),  # same padding
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),  # same padding
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # reduce spatial dimensions
            nn.Dropout(0.25)
        )
        
        # Calculate flattened feature size
        # Use MODEL_INPUT_SIZE from config to calculate the correct flattened size
        # Here we need to calculate the output size after all conv and pooling layers
        with torch.no_grad():
            # Use the input size from the config
            h, w = MODEL_INPUT_SIZE
            
            # First maxpool: kernel=3, stride=3
            h = h // 3
            w = w // 3
            
            # Second maxpool: kernel=2, stride=2
            h = h // 2
            w = w // 2
            
            # Third maxpool: kernel=2, stride=2
            h = h // 2
            w = w // 2
            
            # Calculate flattened size
            self.flatten_dim = 128 * h * w
            print(f"Calculated flattened dimension: {self.flatten_dim}")
        
        # Fully connected layers with dynamically calculated input size
        self.fc_block = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5)
        )
        
        # Output layer with num_classes units
        self.classifier = nn.Linear(1024, num_classes)
        
    def _forward_conv_blocks(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x
        
    def forward(self, x):
        x = self._forward_conv_blocks(x)
        # Flatten maintaining batch dimension
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        x = self.classifier(x)
        return x  # Remove softmax - CrossEntropyLoss expects logits
        
    def extract_features(self, x):
        """
        Extract features from the penultimate layer for explainability.
        """
        x = self._forward_conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc_block(x)
