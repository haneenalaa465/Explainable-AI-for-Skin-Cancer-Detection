"""
DeConv (Deconvolutional Network) explainer for CNN model visualization.

Based on the paper "Visualizing and Understanding Convolutional Networks"
by Matthew D. Zeiler and Rob Fergus (2013).

This module implements a deconvolutional network approach to visualize and
understand what features CNN models learn and how they make predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import cv2

class DeconvExplainer:
    """
    Implements the Deconvolutional Network visualization technique from Zeiler & Fergus.
    Projects feature maps activations back to the input pixel space to visualize
    what patterns activate specific features.
    """
    
    def __init__(self, model, device):
        """
        Initialize the DeConv explainer.
        
        Args:
            model: PyTorch CNN model to visualize
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set model to evaluation mode
        
        # Dictionary to store feature maps during forward pass
        self.feature_maps = OrderedDict()
        # Dictionary to store feature visualization hooks
        self.hooks = []
        
        # Register hooks to capture feature maps during forward pass
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to capture intermediate feature maps."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(
                    module.register_forward_hook(self._hook_fn(name))
                )
                
    def _hook_fn(self, name):
        """Hook function to store feature maps during forward pass."""
        def hook(module, input, output):
            self.feature_maps[name] = output
        return hook
        
    def _unpool(self, input, indices, kernel_size=(2, 2), stride=(2, 2)):
        """
        Perform unpooling operation (approximate inverse of max pooling).
        
        Args:
            input: Input tensor
            indices: Indices of max locations from pooling
            kernel_size: Size of pooling kernel
            stride: Stride of pooling operation
            
        Returns:
            Unpooled tensor
        """
        batch_size, channels, height, width = input.shape
        unpooled = torch.zeros((batch_size, channels, height * stride[0], width * stride[1]), 
                              device=input.device)
        
        # Use indices to place values from input at correct locations in unpooled
        rows = torch.div(indices, kernel_size[1], rounding_mode='floor')
        cols = indices % kernel_size[1]
        
        rows = rows + torch.arange(height, device=input.device).view(-1, 1).repeat(1, width) * stride[0]
        cols = cols + torch.arange(width, device=input.device).repeat(height, 1) * stride[1]
        
        indices_unpooled = rows * (width * stride[1]) + cols
        
        # Reshape for batch and channel dimensions
        indices_unpooled = indices_unpooled.unsqueeze(0).unsqueeze(0)
        indices_unpooled = indices_unpooled.repeat(batch_size, channels, 1, 1)
        
        # Flatten for scatter operation
        flat_indices = indices_unpooled.view(-1)
        flat_input = input.view(-1)
        flat_unpooled = unpooled.view(-1)
        
        # Place values
        flat_unpooled.scatter_(0, flat_indices, flat_input)
        
        return unpooled.view(batch_size, channels, height * stride[0], width * stride[1])
    
    def _deconv(self, feature_map, layer_name):
        """
        Perform deconvolution on a feature map to project it back to input space.
        
        Args:
            feature_map: Feature map to visualize
            layer_name: Name of the layer to visualize
            
        Returns:
            Deconvolved feature map
        """
        # Find the corresponding convolutional layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Layer {layer_name} not found or not a convolutional layer")
        
        # Get the weights of the target layer
        weights = target_layer.weight
        
        # Transpose the weights for deconvolution (flip kernels)
        deconv_weights = weights.detach().clone()
        
        # Flip the kernels horizontally and vertically
        for i in range(deconv_weights.shape[0]):
            for j in range(deconv_weights.shape[1]):
                deconv_weights[i, j] = torch.flip(deconv_weights[i, j], [0, 1])
        
        # Apply deconvolution (transpose convolution with flipped filters)
        deconv_layer = nn.ConvTranspose2d(
            in_channels=weights.shape[0],
            out_channels=weights.shape[1],
            kernel_size=weights.shape[2:],
            stride=target_layer.stride,
            padding=target_layer.padding,
            bias=False
        ).to(self.device)
        
        # Set transposed weights
        deconv_layer.weight.data = deconv_weights
        
        # Apply deconvolution
        deconvolved = deconv_layer(feature_map)
        
        return deconvolved
    
    def visualize_layer(self, input_image, layer_name, feature_idx=None, topk=9, save_path=None):
        """
        Visualize activations of a layer by projecting them back to input space.
        
        Args:
            input_image: Input image tensor (should be preprocessed as model expects)
            layer_name: Name of the layer to visualize
            feature_idx: Specific feature map index to visualize (if None, use topk)
            topk: Number of top activations to visualize (if feature_idx is None)
            save_path: Path to save visualization (if None, just display)
            
        Returns:
            Visualization figure
        """
        # Forward pass to get feature maps
        with torch.no_grad():
            _ = self.model(input_image)
        
        # Check if the requested layer exists
        if layer_name not in self.feature_maps:
            raise ValueError(f"Layer {layer_name} not found in feature maps")
        
        feature_map = self.feature_maps[layer_name]
        
        # If feature_idx is specified, only visualize that feature map
        if feature_idx is not None:
            if feature_idx >= feature_map.shape[1]:
                raise ValueError(f"Feature index {feature_idx} out of range (0-{feature_map.shape[1]-1})")
            
            # Extract the specified feature map
            selected_map = feature_map[:, feature_idx:feature_idx+1]
            
            # Deconvolve the feature map back through the network
            deconvolved = self._deconv(selected_map, layer_name)
            
            # Apply ReLU to ensure positive values only
            deconvolved = F.relu(deconvolved)
            
            # Convert to numpy for visualization
            visualization = deconvolved[0].permute(1, 2, 0).cpu().numpy()
            
            # Normalize for visualization
            if visualization.max() > 0:
                visualization = (visualization - visualization.min()) / (visualization.max() - visualization.min())
            
            # Create figure
            plt.figure(figsize=(8, 8))
            plt.imshow(visualization)
            plt.title(f"Layer: {layer_name}, Feature: {feature_idx}")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path)
                
            return plt.gcf()
        
        else:
            # Find topk activations
            batch_size, num_features, height, width = feature_map.shape
            activations = feature_map.mean(dim=(2, 3))  # Average activation per feature map
            
            # Get indices of top activations
            if topk > num_features:
                topk = num_features
                
            _, top_indices = torch.topk(activations[0], topk)
            
            # Create figure for visualization
            rows = int(np.ceil(np.sqrt(topk)))
            cols = int(np.ceil(topk / rows))
            
            fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            
            for i, feature_idx in enumerate(top_indices):
                if i >= len(axes):
                    break
                    
                # Extract this feature map
                selected_map = feature_map[:, feature_idx:feature_idx+1]
                
                # Deconvolve the feature map
                deconvolved = self._deconv(selected_map, layer_name)
                
                # Apply ReLU
                deconvolved = F.relu(deconvolved)
                
                # Convert to numpy for visualization
                visualization = deconvolved[0].permute(1, 2, 0).cpu().numpy()
                
                # Normalize for visualization
                if visualization.max() > 0:
                    visualization = (visualization - visualization.min()) / (visualization.max() - visualization.min())
                
                # Display
                axes[i].imshow(visualization)
                axes[i].set_title(f"Feature {feature_idx}")
                axes[i].axis('off')
            
            # Hide empty subplots
            for i in range(topk, len(axes)):
                axes[i].axis('off')
                
            plt.suptitle(f"Top {topk} activations for layer: {layer_name}")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                
            return fig
    
    def visualize_layer_with_unpooling(self, input_image, layer_name, pool_indices=None, feature_idx=None, 
                                       topk=9, save_path=None):
        """
        Visualize activations using unpooling to better preserve spatial information.
        
        Args:
            input_image: Input image tensor
            layer_name: Name of layer to visualize
            pool_indices: Indices from max pooling (if available)
            feature_idx: Specific feature map index to visualize
            topk: Number of top activations to visualize
            save_path: Path to save visualization
            
        Returns:
            Visualization figure
        """
        # Implementation similar to visualize_layer but with unpooling using indices
        # This requires keeping track of max pooling indices during forward pass
        # For simplicity, we'll skip this implementation for now
        raise NotImplementedError("Unpooling visualization not yet implemented")
    
    def get_top_activations(self, input_image, layer_name, topk=9):
        """
        Get the indices and values of the top activations for a layer.
        
        Args:
            input_image: Input image tensor
            layer_name: Name of the layer to analyze
            topk: Number of top activations to return
            
        Returns:
            tuple: (indices, values) of top activations
        """
        # Forward pass to get feature maps
        with torch.no_grad():
            _ = self.model(input_image)
        
        if layer_name not in self.feature_maps:
            raise ValueError(f"Layer {layer_name} not found in feature maps")
        
        feature_map = self.feature_maps[layer_name]
        
        # Calculate average activation per feature map
        activations = feature_map.mean(dim=(2, 3))
        
        # Get indices and values of top activations
        values, indices = torch.topk(activations[0], min(topk, activations.shape[1]))
        
        return indices.cpu().numpy(), values.cpu().numpy()
    
    def close(self):
        """Remove hooks to free memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.feature_maps = OrderedDict()