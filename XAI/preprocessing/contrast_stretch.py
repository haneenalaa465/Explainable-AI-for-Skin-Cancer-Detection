import torch
import cv2 as cv
import numpy as np

class ContrastStretch(torch.nn.Module):
    def forward(self, image):
        img = torch.as_tensor(image)
        stretched_img = torch.zeros_like(img)
        
        for c in range(img.shape[0]):
            channel = img[c]
            min_val = channel.min()
            max_val = channel.max()

            if max_val > min_val: 
                stretched_img[c] = (channel - min_val) / (max_val - min_val)
            else:
                stretched_img[c] = channel 

        stretched_img = torch.clamp(stretched_img, 0, 1)
        
        return stretched_img