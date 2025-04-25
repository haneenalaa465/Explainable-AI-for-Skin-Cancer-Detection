import torch
import torch.nn as nn
import torch.nn.functional as F

from XAI.config import NUM_CLASSES
from XAI.modeling.models.Base_Model import BaseModel

# Skin Lesion Classification Using Convolutional Neural Network With Novel Regularizer


# Custom regularizer function for standard deviation penalty
def stddev_regularizer(weights, lambda_reg=0.02):
    # Flatten all dimensions except the output channels (filters)
    num_filters = weights.shape[0]
    flattened = weights.view(num_filters, -1)
    std_per_filter = torch.std(flattened, dim=1)
    penalty = lambda_reg * torch.sum(std_per_filter)
    return penalty


# Custom CNN Model
class CustomCNNReg(BaseModel):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CustomCNNReg, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 73 * 73, 128)  # 300 -> 298 -> 296 -> 148 -> 73
        self.fc2 = nn.Linear(128, num_classes)

    def name(self):
        return "CustomCNNWithReg"

    def regularizer(self,**kwargs):
        return stddev_regularizer(**kwargs)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Output: (32, 298, 298)
        x = F.relu(self.conv2(x))  # Output: (64, 296, 296)
        x = self.pool(x)  # Output: (64, 148, 148)
        x = self.dropout(x)
        x = self.flatten(x)  # Output: (64 * 73 * 73)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
