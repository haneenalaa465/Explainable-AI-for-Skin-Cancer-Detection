"""
SHAP (SHapley Additive exPlanations) for skin lesion classification models.

This module provides functions to explain model predictions using SHAP values,
which represent feature importance based on cooperative game theory.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from tqdm import tqdm

class ShapExplainer:
    """
    A class to generate and visualize SHAP explanations for image classification models.
    """
    
    def __init__(self, model, device, class_names, preprocess_fn=None):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: PyTorch model
            device: Device to run the model on
            class_names: Dictionary of class names or list of class names
            preprocess_fn: Function to preprocess images for the model
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.preprocess_fn = preprocess_fn
        self.explainer = None  # Will be initialized when needed
        
    def remove_inplace_relu(self):
        """
        Remove inplace ReLU operations from the model to avoid errors with SHAP.
        """
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
                
    def prepare_background(self, bg_images=None, n_samples=100):
        """
        Prepare background data for the SHAP explainer.
        
        Args:
            bg_images: List of background images (if None, uses zero tensors)
            n_samples: Number of background samples to use
            
        Returns:
            torch.Tensor: Background tensor
        """
        # Get input shape from the model or a sample input
        if hasattr(self.model, 'inputSize'):
            input_shape = self.model.inputSize()
            channels = 3  # Assume RGB
            bg_shape = (n_samples, channels, input_shape[0], input_shape[1])
        else:
            # Use a default shape or infer from the first layer
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    channels = module.in_channels
                    bg_shape = (n_samples, channels, 224, 224)  # Default to 224x224
                    break
        
        # Generate background
        if bg_images is None:
            # Create tensor of zeros
            bg_tensor = torch.zeros(bg_shape, device=self.device)
        else:
            # Process background images
            bg_tensors = []
            for i, bg_img in enumerate(bg_images[:n_samples]):
                if self.preprocess_fn:
                    bg_tensor = self.preprocess_fn(bg_img).unsqueeze(0)  # Add batch dimension
                else:
                    # Default preprocessing
                    bg_tensor = torch.from_numpy(bg_img.transpose((2, 0, 1))).float() / 255.0
                    bg_tensor = bg_tensor.unsqueeze(0)  # Add batch dimension
                    
                bg_tensors.append(bg_tensor)
                
            # Concatenate tensors
            bg_tensor = torch.cat(bg_tensors, dim=0).to(self.device)
            
            # Pad if needed
            if len(bg_tensors) < n_samples:
                padding = torch.zeros(
                    (n_samples - len(bg_tensors),) + bg_tensor.shape[1:],
                    device=self.device
                )
                bg_tensor = torch.cat([bg_tensor, padding], dim=0)
                
        return bg_tensor
    
    def explain(self, image, bg_images=None, n_samples=100):
        """
        Generate SHAP values for an image.
        
        Args:
            image: Input image (numpy array)
            bg_images: Background images to use for the explainer
            n_samples: Number of background samples
            
        Returns:
            shap_values: SHAP values for the image
        """
        # Preprocess input image
        if self.preprocess_fn:
            input_tensor = self.preprocess_fn(image).unsqueeze(0).to(self.device)  # Add batch dimension
        else:
            # Default preprocessing
            input_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Make sure ReLU operations are not inplace
        self.remove_inplace_relu()
        
        # Prepare background data
        bg_tensor = self.prepare_background(bg_images, n_samples)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Create DeepExplainer
        print("Creating SHAP explainer...")
        self.explainer = shap.DeepExplainer(self.model, bg_tensor)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        input_tensor.requires_grad_()
        shap_values = self.explainer.shap_values(input_tensor)
        
        return shap_values
    
    def visualize(self, shap_values, image, label=None, save_path=None):
        """
        Visualize SHAP values for an image.
        
        Args:
            shap_values: SHAP values from explain()
            image: Original image
            label: Label to explain (if None, explains for all classes)
            save_path: Path to save the visualization
            
        Returns:
            None (displays the visualization)
        """
        # Get class name
        class_name = label
        if label is not None:
            if isinstance(self.class_names, dict):
                class_keys = list(self.class_names.keys())
                if isinstance(label, int) and label < len(class_keys):
                    class_name = self.class_names[class_keys[label]]
            elif isinstance(self.class_names, list) and isinstance(label, int) and label < len(self.class_names):
                class_name = self.class_names[label]
        
        # If a specific label is provided, visualize only that class
        if label is not None and isinstance(label, int):
            # Ensure label is in range
            if 0 <= label < len(shap_values):
                selected_shap = [shap_values[label]]
                class_names = [class_name]
            else:
                selected_shap = shap_values
                class_names = self.class_names
        else:
            selected_shap = shap_values
            if isinstance(self.class_names, dict):
                class_names = list(self.class_names.values())
            else:
                class_names = self.class_names
        
        # Create figure
        n_classes = len(selected_shap)
        fig, axs = plt.subplots(1, n_classes + 1, figsize=(4 * (n_classes + 1), 4))
        
        # Original image
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        
        # SHAP values for each class
        for i in range(n_classes):
            # For visualization, sum absolute SHAP values across channels
            shap_combined = np.abs(selected_shap[i][0]).sum(axis=0)
            
            # Normalize to [0, 1] for visualization
            shap_normalized = shap_combined / shap_combined.max()
            
            # Create a heatmap
            axs[i+1].imshow(shap_normalized, cmap='hot')
            title = f"SHAP values"
            if n_classes > 1 and i < len(class_names):
                title += f"\nClass: {class_names[i]}"
            axs[i+1].set_title(title)
            axs[i+1].axis("off")
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"SHAP explanation saved to {save_path}")
            
        return fig, axs
