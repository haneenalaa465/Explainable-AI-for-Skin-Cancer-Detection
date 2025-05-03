"""
GradCAM (Gradient-weighted Class Activation Mapping) for skin lesion classification models.

This module provides functions to explain model predictions using GradCAM, 
which highlights regions of the image that are important for the model's prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCamExplainer:
    """
    A class to generate and visualize GradCAM explanations for image classification models.
    """
    
    def __init__(self, model, target_layers=None, use_cuda=True, method='gradcam'):
        """
        Initialize the GradCAM explainer.
        
        Args:
            model: PyTorch model
            target_layers: List of target layers for GradCAM
                          (if None, tries to find the last convolutional layer)
            use_cuda: Whether to use CUDA if available
            method: 'gradcam' or 'gradcam++' to specify which method to use
        """
        self.model = model
        self.target_layers = target_layers or self._find_target_layers()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Initialize GradCAM
        if method.lower() == 'gradcam++':
            self.explainer = GradCAMPlusPlus(
                model=model,
                target_layers=self.target_layers,
                use_cuda=self.use_cuda
            )
        else:  # Default to GradCAM
            self.explainer = GradCAM(
                model=model,
                target_layers=self.target_layers,
                use_cuda=self.use_cuda
            )
        
    def _find_target_layers(self):
        """
        Find suitable target layers for GradCAM.
        
        Returns:
            list: List of target layers
        """
        # Look for the last convolutional layer
        target_layers = []
        
        # Handle models wrapped with ResizedModel
        model_to_check = self.model
        if hasattr(self.model, 'model'):
            model_to_check = self.model.model
        
        # First, try to find the last convolutional layer in the feature extractor
        # (common in many CNN architectures)
        if hasattr(model_to_check, 'features') and isinstance(model_to_check.features, torch.nn.Sequential):
            for layer in reversed(list(model_to_check.features)):
                if isinstance(layer, torch.nn.Conv2d):
                    target_layers.append(layer)
                    print(f"Using layer {layer} for GradCAM")
                    return target_layers
        
        # If not found, look for the last convolutional layer in the entire model
        for name, module in reversed(list(model_to_check.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layers.append(module)
                print(f"Using layer {name} for GradCAM")
                return target_layers
                
        # If still not found, look for potential convolutional blocks
        for name, module in model_to_check.named_modules():
            if any(isinstance(submodule, torch.nn.Conv2d) for submodule in module.children()):
                for subname, submodule in module.named_modules():
                    if isinstance(submodule, torch.nn.Conv2d):
                        target_layers.append(submodule)
                        print(f"Using layer {name}.{subname} for GradCAM")
                        return target_layers
                        
        raise ValueError("Could not find any convolutional layer for GradCAM. Please specify target_layers manually.")
        
    def explain(self, input_tensor, targets=None, aug_smooth=False, eigen_smooth=False):
        """
        Generate GradCAM heatmap for an input tensor.
        
        Args:
            input_tensor: Input tensor for the model
            targets: List of targets for the GradCAM (class indices)
            aug_smooth: Whether to use test time augmentation to smooth the CAM
            eigen_smooth: Whether to use eigen value smoothing
            
        Returns:
            numpy.ndarray: GradCAM heatmap
        """
        with torch.enable_grad():
            grayscale_cam = self.explainer(
                input_tensor=input_tensor,
                targets=targets,
                aug_smooth=aug_smooth,
                eigen_smooth=eigen_smooth
            )
        
        return grayscale_cam
        
    def visualize(self, grayscale_cam, original_image, class_name=None, save_path=None):
        """
        Visualize GradCAM heatmap overlaid on the original image.
        
        Args:
            grayscale_cam: GradCAM heatmap from explain()
            original_image: Original image (numpy array)
            class_name: Name of the class being explained
            save_path: Path to save the visualization
            
        Returns:
            numpy.ndarray: Visualization of CAM overlaid on image
        """
        # Ensure original image is in [0, 1] range
        if original_image.max() > 1.0:
            original_image = original_image / 255.0
            
        # Create CAM overlay
        cam_image = show_cam_on_image(original_image, grayscale_cam[0], use_rgb=True)
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")
        
        # GradCAM overlay
        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        title = "GradCAM Heatmap"
        if class_name:
            title += f"\nClass: {class_name}"
        plt.title(title)
        plt.axis("off")
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"GradCAM explanation saved to {save_path}")
            
        return cam_image
