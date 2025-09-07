"""
Utilities for visualizing activations and explanations from the DeConv network.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from skimage import transform
import cv2
from PIL import Image

def normalize_image(img):
    """
    Normalize image to [0, 1] range.
    
    Args:
        img: Image array
        
    Returns:
        Normalized image
    """
    img_min = img.min()
    img_max = img.max()
    
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def create_heatmap_overlay(image, activation, alpha=0.5, colormap='jet'):
    """
    Create a heatmap overlay of activation on an image.
    
    Args:
        image: Original image (numpy array)
        activation: Activation map (numpy array)
        alpha: Transparency of the overlay
        colormap: Matplotlib colormap name
        
    Returns:
        Overlay image
    """
    # Ensure image is normalized to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
        
    # Normalize activation map to [0, 1]
    activation = normalize_image(activation)
    
    # Resize activation to match image dimensions if needed
    if activation.shape != image.shape[:2]:
        activation = transform.resize(activation, image.shape[:2], order=1, mode='reflect', anti_aliasing=True)
    
    # Create colormap
    cmap = plt.get_cmap(colormap)
    
    # Apply colormap to activation
    activation_rgb = cmap(activation)[:, :, :3]
    
    # Create overlay
    overlay = image * (1 - alpha) + activation_rgb * alpha
    
    # Ensure result is in [0, 1]
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def visualize_feature_maps(feature_maps, num_features=None, figsize=None):
    """
    Visualize multiple feature maps.
    
    Args:
        feature_maps: Tensor of feature maps [batch, channels, height, width]
        num_features: Number of features to display (if None, display all)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if tensor
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.detach().cpu().numpy()
    
    # Get dimensions
    num_channels = feature_maps.shape[1]
    
    # Determine number of features to show
    if num_features is None or num_features > num_channels:
        num_features = num_channels
    
    # Determine grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    # Create figure
    if figsize is None:
        figsize = (2 * grid_size, 2 * grid_size)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each feature map
    for i in range(num_features):
        feature = feature_maps[0, i]
        feature = normalize_image(feature)
        axes[i].imshow(feature, cmap='viridis')
        axes[i].set_title(f"Feature {i}")
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_features, grid_size * grid_size):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_deconv_results(original_image, deconv_results, topk=9, figsize=None):
    """
    Visualize DeConv results for multiple feature maps.
    
    Args:
        original_image: Original input image
        deconv_results: List of deconvolution results
        topk: Number of top results to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Limit number of results to show
    if len(deconv_results) > topk:
        deconv_results = deconv_results[:topk]
    
    # Determine grid dimensions (+ 1 for original image)
    num_results = len(deconv_results) + 1
    grid_size = int(np.ceil(np.sqrt(num_results)))
    
    # Create figure
    if figsize is None:
        figsize = (3 * grid_size, 3 * grid_size)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    # Show original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show each deconv result
    for i, (feat_idx, deconv_img) in enumerate(deconv_results):
        axes[i+1].imshow(deconv_img)
        axes[i+1].set_title(f"Feature {feat_idx}")
        axes[i+1].axis('off')
    
    # Hide empty subplots
    for i in range(num_results, grid_size * grid_size):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def create_grid_visualization(images, titles=None, figsize=None, cmap=None):
    """
    Create a grid visualization of multiple images.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        figsize: Figure size
        cmap: Colormap for the images
        
    Returns:
        Matplotlib figure
    """
    num_images = len(images)
    
    # Determine grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Create figure
    if figsize is None:
        figsize = (3 * grid_size, 3 * grid_size)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Show each image
    for i, img in enumerate(images):
        if i < num_images:
            axes[i].imshow(img, cmap=cmap)
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_images, grid_size * grid_size):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def preprocess_image(image, model_input_size=(224, 224)):
    """
    Preprocess an image for visualization.
    
    Args:
        image: PIL image or numpy array
        model_input_size: Size expected by the model
        
    Returns:
        Preprocessed image tensor
    """
    # Convert PIL image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure RGB format
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    
    # Resize to model input size
    if image.shape[:2] != model_input_size:
        image = cv2.resize(image, model_input_size)
    
    # Convert to float and normalize to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor

def tensor_to_image(tensor):
    """
    Convert a tensor to a numpy image.
    
    Args:
        tensor: PyTorch tensor with shape [batch, channels, height, width]
        
    Returns:
        Numpy array with shape [height, width, channels]
    """
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Remove batch dimension
    
    # Move to CPU and convert to numpy
    image = tensor.detach().cpu().numpy()
    
    # Rearrange dimensions to [height, width, channels]
    image = np.transpose(image, (1, 2, 0))
    
    # Clip to valid range
    image = np.clip(image, 0, 1)
    
    return image