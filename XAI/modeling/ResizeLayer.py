# Define this somewhere accessible
import torch.nn as nn
from torchvision import transforms


class ResizedModel(nn.Module):
    def __init__(self, target_size, original_model):
        super().__init__()
        self.resize = transforms.Resize(
            target_size, interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.model = original_model

    def name(self):
        return self.model.name()

    def forward(self, x):

        x = self.resize(x)
        return self.model(x)
