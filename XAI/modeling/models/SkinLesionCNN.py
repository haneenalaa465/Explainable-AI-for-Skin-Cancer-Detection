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
     CNN model for skin lesion classification based on the paper:
    "Skin lesion classification of dermoscopic images using machine learning and convolutional neural network"
    Implementation of the model described in the paper with the following architecture:
    - Conv2D (32 filters, 3×3 filter size, ReLU activation, same padding, followed by batch normalization)
    - MaxPool2D (3×3 pool size)
    - Dropout (0.25)
    - Conv2D (64 filters, 3×3 filter size, ReLU activation, same padding)
    - Conv2D (64 filters, 3×3 filter size, ReLU activation, same padding, batch normalization)
    - MaxPool2D (2×2 pool size)
    - Dropout (0.25)
    - Conv2D (128 filters, 3×3 filter size, ReLU activation, same padding, batch normalization)
    - Conv2D (128 filters, 3×3 filter size, ReLU activation, same padding, batch normalization)
    - MaxPool2D (2×2 pool size)
    - Dropout (0.25)
    - Flatten
    - Dense (1024 units, ReLU activation, batch normalization)
    - Dropout (0.5)
    - Dense (7 units, softmax activation)
    """

    def name():
        return "SkinLesionCNN"

    def __init__(self, num_classes=NUM_CLASSES):
        super(SkinLesionCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout1 = nn.Dropout(0.25)

        # Second convolutional block
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # Third convolutional block
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        # Calculate feature map size after three downsampling operations
        # Input: 96x96 -> 32x32 (after pool1) -> 16x16 (after pool2) -> 8x8 (after pool3)
        self.feature_size = 8 * 8 * 128

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = F.relu(self.conv2_1(x))
        x = self.conv2_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third block
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)

        return x

    def extract_features(self, x):
        """
        Extract features from the model before the final classification layer.
        Useful for visualization and explainability.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Feature tensor from the penultimate layer
        """
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = F.relu(self.conv2_1(x))
        x = self.conv2_2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third block
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # First fully connected layer with batch norm
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)

        # Return features
        return x
