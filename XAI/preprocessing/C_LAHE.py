import torch
import cv2 as cv
import numpy as np


class CLAHE(torch.nn.Module):

    def forward(self, image):  # we assume inputs are always structured like this
        # By asking for assistance in loading data
        image = image.copy()

        # Convert from BGR to LAB color space
        lab = cv.cvtColor(image, cv.COLOR_RGB2LAB)

        # Split LAB channels
        l, a, b = cv.split(lab)

        # Apply CLAHE to the L-channel (lightness)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
        l_clahe = clahe.apply(l)

        # Merge back the channels
        lab_clahe = cv.merge((l_clahe, a, b))

        # Convert back to BGR
        final_img = cv.cvtColor(lab_clahe, cv.COLOR_LAB2RGB)

        return final_img
