import torch
import cv2 as cv
import numpy as np
import albumentations as A


class HairRemoval(torch.nn.Module):

    def forward(self, image):  # we assume inputs are always structured like this
        # By asking for assistance in loading data

        # Expecting a single image tensor: shape [C, H, W], type float or byte
        # Convert to NumPy
        # use canny algorithm to detect the edges
        image = image.copy()
        dst = cv.Canny(image, 50, 200, None, 3)

        # use Hough line transform to detect curves
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 40, None, 20, 20)

        # create mask of zeros
        mask = np.zeros_like(dst)

        # Draw the lines on the mask
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(
                    mask, (l[0], l[1]), (l[2], l[3]), 255, 4, cv.LINE_AA
                )  # white lines on black background

        # inpaint the missing lines
        img_inpainted = cv.inpaint(image, mask, 1, cv.INPAINT_TELEA)

        return img_inpainted
