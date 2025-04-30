import torch
import torch.nn as nn
import cv2
import numpy as np


class EnhanceClarityCV(nn.Module):
    """
    A PyTorch module that enhances image clarity using OpenCV functions.

    Applies a 5x5 Gaussian Blur followed by a 3x3 sharpening filter.
    Assumes input tensor is in the format [B, C, H, W] and represents
    images with pixel values scaled, typically between 0.0 and 1.0 (float).
    """

    def __init__(self):
        super().__init__()

        # Define the 3x3 sharpening kernel
        # Using np.float32 for compatibility with typical PyTorch float tensors
        self.sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

        # Define Gaussian kernel size
        self.gaussian_ksize = (5, 5)

    def forward(self, x):
        image = x.copy()
        # --- Apply Gaussian Blur ---
        # sigmaX=0 means it's computed from kernel size. sigmaY defaults to sigmaX.
        blurred_img = cv2.GaussianBlur(image, self.gaussian_ksize, 0)

        # --- Apply Sharpening Filter ---
        # cv2.filter2D applies correlation, which is equivalent to convolution
        # with a flipped kernel. For symmetric kernels like ours, it's the same.
        # ddepth = -1 means the output image will have the same depth as the source.
        sharpened_img = cv2.filter2D(blurred_img, -1, self.sharpen_kernel)

        # --- Clipping ---
        # Sharpening can push values outside the typical [0, 1] range.
        # Clip the result to maintain the expected range.
        # Use float limits for clipping.
        # clipped_img = np.clip(sharpened_img, 0.0, 1.0)
        return sharpened_img
