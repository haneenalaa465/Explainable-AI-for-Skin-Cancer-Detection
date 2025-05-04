"""
Helper utilities for working with CNN model architectures.
"""

import torch
import torch.nn as nn
from collections import OrderedDict

def get_conv_layers(model):
    """
    Get all convolutional layers in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        OrderedDict: Dictionary mapping layer names to conv layers
    """
    conv_layers = OrderedDict()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers[name] = module
            
    return conv_layers

def get_activation_layers(model):
    """
    Get all activation layers in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        OrderedDict: Dictionary mapping layer names to activation layers
    """
    activation_layers = OrderedDict()
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
            activation_layers[name] = module
            
    return activation_layers

def get_pooling_layers(model):
    """
    Get all pooling layers in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        OrderedDict: Dictionary mapping layer names to pooling layers
    """
    pooling_layers = OrderedDict()
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            pooling_layers[name] = module
            
    return pooling_layers

def get_layer_dependencies(model):
    """
    Build a dependency graph of layers to determine the execution order.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Dictionary mapping layer names to their dependencies
    """
    # This is a simplified approach - for complex models with skip connections,
    # we would need more sophisticated analysis
    dependencies = {}
    prev_layer = None
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.ReLU, nn.LeakyReLU)):
            if prev_layer is not None:
                dependencies[name] = [prev_layer]
            else:
                dependencies[name] = []
            prev_layer = name
            
    return dependencies

def replace_max_pool_with_indices(model):
    """
    Replace standard MaxPool2d layers with ones that return indices.
    
    Args:
        model: PyTorch model
        
    Returns:
        Modified model with custom MaxPool layers
    """
    from XAI.explainers.deconv_model import MaxPoolingWithIndices
    
    # Create a deep copy of the model to avoid modifying the original
    model_copy = type(model)()
    model_copy.load_state_dict(model.state_dict())
    
    for name, module in model_copy.named_children():
        if isinstance(module, nn.MaxPool2d):
            # Replace with custom max pooling
            setattr(model_copy, name, MaxPoolingWithIndices(
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding
            ))
        elif isinstance(module, nn.Sequential):
            # For sequential modules, we need to recursively process them
            new_seq = nn.Sequential()
            for idx, layer in enumerate(module):
                if isinstance(layer, nn.MaxPool2d):
                    new_seq.add_module(str(idx), MaxPoolingWithIndices(
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding
                    ))
                else:
                    new_seq.add_module(str(idx), layer)
            setattr(model_copy, name, new_seq)
            
    return model_copy

def visualize_model_architecture(model):
    """
    Create a string representation of the model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        str: Text representation of the model architecture
    """
    lines = ["Model Architecture:"]
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.ReLU, nn.Linear)):
            # Skip the top-level module
            if name == '':
                continue
                
            # Add details based on layer type
            if isinstance(module, nn.Conv2d):
                lines.append(f"{name}: Conv2d(in={module.in_channels}, out={module.out_channels}, "
                           f"kernel={module.kernel_size}, stride={module.stride}, padding={module.padding})")
            elif isinstance(module, nn.MaxPool2d):
                lines.append(f"{name}: MaxPool2d(kernel={module.kernel_size}, "
                           f"stride={module.stride}, padding={module.padding})")
            elif isinstance(module, nn.AvgPool2d):
                lines.append(f"{name}: AvgPool2d(kernel={module.kernel_size}, "
                           f"stride={module.stride}, padding={module.padding})")
            elif isinstance(module, nn.ReLU):
                lines.append(f"{name}: ReLU(inplace={module.inplace})")
            elif isinstance(module, nn.Linear):
                lines.append(f"{name}: Linear(in={module.in_features}, out={module.out_features})")
    
    return "\n".join(lines)