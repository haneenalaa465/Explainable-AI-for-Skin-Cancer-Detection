"""
Implementation of a Deconvolutional Network for CNN visualization.

This module provides the DeconvNet class as described in the paper
"Visualizing and Understanding Convolutional Networks" by Zeiler & Fergus.
It reconstructs the activity in a given feature map all the way back to the input space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class MaxPoolingWithIndices(nn.Module):
    """
    Max pooling layer that returns both the pooled values and their indices.
    Useful for unpooling operations in deconvolutional networks.
    """
    def __init__(self, kernel_size, stride, padding=0):
        super(MaxPoolingWithIndices, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding, return_indices=True)


class DeconvNet(nn.Module):
    """
    Deconvolutional Network for reconstructing activations back to input space.
    """
    def __init__(self, model, device):
        """
        Initialize the DeconvNet.
        
        Args:
            model: CNN model to visualize
            device: Device to run computations on ('cpu' or 'cuda')
        """
        super(DeconvNet, self).__init__()
        
        self.model = model
        self.device = device
        
        # Dictionaries to store feature maps and switch variables during forward pass
        self.feature_maps = OrderedDict()
        self.switch_variables = OrderedDict()
        self.intermediate_outputs = OrderedDict()
        
        # Layer types that we need to track
        self.tracked_layers = (nn.Conv2d, nn.MaxPool2d, nn.ReLU)
        
        # Register hooks for tracking feature maps and switches
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture feature maps and switch variables."""
        for name, module in self.model.named_modules():
            if isinstance(module, self.tracked_layers):
                self.hooks.append(
                    module.register_forward_hook(self._hook_fn(name))
                )
    
    def _hook_fn(self, name):
        """Hook function to store outputs during forward pass."""
        def hook(module, input, output):
            # For max pooling layers, store both output and indices
            if isinstance(module, nn.MaxPool2d):
                if isinstance(output, tuple) and len(output) == 2:
                    # Some implementations return indices already
                    self.feature_maps[name] = output[0]
                    self.switch_variables[name] = output[1]
                else:
                    # If not, compute max pool with indices ourselves
                    _, indices = F.max_pool2d(
                        input[0], 
                        module.kernel_size, 
                        module.stride, 
                        module.padding, 
                        return_indices=True
                    )
                    self.feature_maps[name] = output
                    self.switch_variables[name] = indices
            else:
                self.feature_maps[name] = output
                
            # Store intermediate output for each layer
            self.intermediate_outputs[name] = output
        
        return hook
    
    def _unpool(self, input, indices, output_size=None, kernel_size=(2, 2), stride=(2, 2)):
        """
        Perform unpooling using switch variables (max indices).
        
        Args:
            input: Input tensor to unpool
            indices: Switch variables (indices of max values)
            output_size: Size of the output tensor
            kernel_size: Size of the original pooling kernel
            stride: Stride of the original pooling operation
            
        Returns:
            Unpooled tensor
        """
        return F.max_unpool2d(input, indices, kernel_size, stride, padding=0, output_size=output_size)
    
    def _find_previous_layer(self, target_layer_name):
        """Find the previous layer in the network."""
        layers = list(self.intermediate_outputs.keys())
        if target_layer_name not in layers:
            raise ValueError(f"Layer {target_layer_name} not found")
            
        idx = layers.index(target_layer_name)
        return layers[idx - 1] if idx > 0 else None
    
    def _find_matching_unpool_size(self, target_layer_name):
        """Find the output size for unpooling based on previous layer."""
        prev_layer = self._find_previous_layer(target_layer_name)
        if prev_layer and prev_layer in self.feature_maps:
            return self.feature_maps[prev_layer].size()
        return None
    
    def compute_deconv_output(self, target_layer_name, target_feature_idx, guided=True):
        """
        Compute the deconvolution output for a specific feature map.
        
        Args:
            target_layer_name: Name of the layer to visualize
            target_feature_idx: Index of the feature map to visualize
            guided: Whether to use guided backprop (zeros negative gradients)
            
        Returns:
            Reconstructed input that would activate the feature map
        """
        if target_layer_name not in self.feature_maps:
            raise ValueError(f"Layer {target_layer_name} not found in feature maps")
            
        # Get the target feature map
        target_map = self.feature_maps[target_layer_name]
        
        # Create a tensor with zeros everywhere except at the target feature map
        deconv_input = torch.zeros_like(target_map)
        
        # Set target feature map to its original activation values
        if target_feature_idx < deconv_input.shape[1]:
            deconv_input[0, target_feature_idx] = target_map[0, target_feature_idx]
        else:
            raise ValueError(f"Feature index {target_feature_idx} out of range (0-{deconv_input.shape[1]-1})")
        
        # Now we work backwards through the network
        deconv_output = deconv_input
        
        # Get list of layers in reverse order up to our target
        layers = list(self.intermediate_outputs.keys())
        target_idx = layers.index(target_layer_name)
        reversed_layers = layers[:target_idx+1][::-1]
        
        # Track deconvolution through each layer
        for i, layer_name in enumerate(reversed_layers):
            if i == 0:  # Skip the target layer itself
                continue
                
            for name, module in self.model.named_modules():
                if name == layer_name:
                    # Apply appropriate inverse operation based on layer type
                    if isinstance(module, nn.Conv2d):
                        # Get the weights of the convolution
                        weights = module.weight
                        
                        # Transpose the weights for deconvolution
                        deconv_weights = weights.detach().clone()
                        
                        # Flip the kernels horizontally and vertically
                        for j in range(deconv_weights.shape[0]):
                            for k in range(deconv_weights.shape[1]):
                                deconv_weights[j, k] = torch.flip(deconv_weights[j, k], [0, 1])
                        
                        # Create transpose convolution layer
                        deconv_layer = nn.ConvTranspose2d(
                            in_channels=weights.shape[0],
                            out_channels=weights.shape[1],
                            kernel_size=weights.shape[2:],
                            stride=module.stride,
                            padding=module.padding,
                            bias=False
                        ).to(self.device)
                        
                        # Set the transposed weights
                        deconv_layer.weight.data = deconv_weights
                        
                        # Apply the deconvolution
                        deconv_output = deconv_layer(deconv_output)
                        
                    elif isinstance(module, nn.MaxPool2d) and layer_name in self.switch_variables:
                        # For max pooling, use the saved indices for unpooling
                        output_size = self._find_matching_unpool_size(layer_name)
                        deconv_output = self._unpool(
                            deconv_output, 
                            self.switch_variables[layer_name],
                            output_size,
                            module.kernel_size,
                            module.stride
                        )
                    
                    elif isinstance(module, nn.ReLU):
                        # For ReLU, apply ReLU again (only pass positive values)
                        if guided:
                            # In guided backprop, we only keep positive gradients
                            deconv_output = F.relu(deconv_output)
                        else:
                            # Standard deconvnet passes through the Relu
                            deconv_output = deconv_output
                            
                    # Apply any other necessary inverse operations...
                    
                    break
        
        return deconv_output
    
    def visualize_feature(self, target_layer_name, target_feature_idx, guided=True):
        """
        Visualize a specific feature map by deconvolving it back to input space.
        
        Args:
            target_layer_name: Name of the layer to visualize
            target_feature_idx: Index of the feature map to visualize
            guided: Whether to use guided backprop
            
        Returns:
            Visualization tensor
        """
        deconv_output = self.compute_deconv_output(target_layer_name, target_feature_idx, guided)
        
        # Convert to visualization format
        vis = deconv_output[0].permute(1, 2, 0).detach().cpu().numpy()
        
        # Normalize for display
        if vis.max() != vis.min():
            vis = (vis - vis.min()) / (vis.max() - vis.min())
            
        return vis
    
    def close(self):
        """Remove hooks to free memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.feature_maps = OrderedDict()
        self.switch_variables = OrderedDict()
        self.intermediate_outputs = OrderedDict()