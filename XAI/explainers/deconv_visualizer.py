"""
Main module for visualizing CNN models using the DeConv technique.

This module provides a command-line interface and utilities for visualizing
the features learned by any CNN model in the project using the Deconvolutional
Network approach from Zeiler & Fergus.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import time
from pathlib import Path

from XAI.config import FIGURES_DIR, CLASS_NAMES
from XAI.dataset import get_transforms
from XAI.modeling.ResizeLayer import ResizedModel
from XAI.modeling.AllModels import dl_models, device
from XAI.modeling.train import load_best_model
from XAI.explainers.deconv_model import DeconvNet
from XAI.explainers.model_helper import get_conv_layers, visualize_model_architecture
from XAI.explainers.visualization_utils import (
    preprocess_image, 
    visualize_deconv_results,
    create_grid_visualization,
    normalize_image,
    tensor_to_image
)

def load_model(model_idx, use_best_weights=True):
    """
    Load a model for visualization.
    
    Args:
        model_idx: Index of the model to load
        use_best_weights: Whether to load the best weights
        
    Returns:
        Loaded model
    """
    if model_idx < 0 or model_idx >= len(dl_models):
        raise ValueError(f"Model index {model_idx} out of range (0-{len(dl_models)-1})")
        
    model_class = dl_models[model_idx]
    model_name = model_class.name()
    print(f"Loading model: {model_name}")
    
    # Create model with correct input size
    model = ResizedModel(model_class.inputSize(), model_class()).to(device)
    
    # Load the best weights if available
    if use_best_weights:
        best_model_path, checkpoint = load_best_model(model_name)
        
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded weights from {best_model_path}")
        else:
            print(f"No saved weights found for {model_name}, using untrained model")
    
    # Set model to evaluation mode
    model.eval()
    
    return model, model_name

def visualize_model_features(model, image_path, layer_name=None, feature_idx=None, 
                             num_features=9, save_dir=None, model_name=None):
    """
    Visualize features learned by the model using DeConv.
    
    Args:
        model: CNN model to visualize
        image_path: Path to input image
        layer_name: Name of the layer to visualize (if None, use the last conv layer)
        feature_idx: Index of feature map to visualize (if None, visualize top activated)
        num_features: Number of top features to visualize
        save_dir: Directory to save visualizations
        model_name: Name of the model (for saving)
        
    Returns:
        None
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Prepare image tensor for the model
    transform = get_transforms("val")
    image_tensor = transform(image_np).unsqueeze(0).to(device)
    
    # Setup DeconvNet
    deconv_net = DeconvNet(model, device)
    
    # Forward pass to compute feature maps
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Get all convolutional layers
    conv_layers = get_conv_layers(model)
    
    # If layer_name not specified, use the last conv layer
    if layer_name is None:
        layer_name = list(conv_layers.keys())[-1]
        print(f"Using last convolutional layer: {layer_name}")
    elif layer_name not in conv_layers:
        raise ValueError(f"Layer {layer_name} not found in model conv layers")
    
    # Get feature maps for the specified layer
    if layer_name not in deconv_net.feature_maps:
        raise ValueError(f"No feature maps found for layer {layer_name}")
        
    feature_map = deconv_net.feature_maps[layer_name]
    
    # Calculate the activations
    activations = feature_map.mean(dim=(2, 3))[0]  # Average activation per feature map
    
    # Handle specific feature index or top activated features
    if feature_idx is not None:
        if feature_idx >= feature_map.shape[1]:
            raise ValueError(f"Feature index {feature_idx} out of range (0-{feature_map.shape[1]-1})")
            
        # Visualize specific feature
        feature_indices = [feature_idx]
        print(f"Visualizing feature {feature_idx} in layer {layer_name}")
    else:
        # Get indices of top activated features
        _, top_indices = torch.topk(activations, min(num_features, feature_map.shape[1]))
        feature_indices = top_indices.cpu().numpy()
        print(f"Visualizing top {len(feature_indices)} features in layer {layer_name}")
    
    # Create directory for saving results
    if save_dir is None:
        save_dir = FIGURES_DIR / "deconv"
    os.makedirs(save_dir, exist_ok=True)
    
    # Add model name to directory if provided
    if model_name:
        save_dir = save_dir / model_name
        os.makedirs(save_dir, exist_ok=True)
    
    # Visualize each feature
    image_name = Path(image_path).stem
    all_results = []
    
    for idx in feature_indices:
        # Compute deconvolution for this feature
        print(f"Computing deconvolution for feature {idx}...")
        deconv_output = deconv_net.visualize_feature(layer_name, idx)
        
        # Add to results
        all_results.append((idx, deconv_output))
        
        # Save individual visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(deconv_output)
        plt.title(f"Layer: {layer_name}, Feature: {idx}")
        plt.axis('off')
        plt.tight_layout()
        
        save_path = save_dir / f"{layer_name.replace('.', '_')}_feature_{idx}_{image_name}.png"
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved visualization to {save_path}")
    
    # Create combined visualization
    fig = visualize_deconv_results(image_np, all_results)
    
    # Save combined visualization
    combined_path = save_dir / f"{layer_name.replace('.', '_')}_combined_{image_name}.png"
    fig.savefig(combined_path)
    plt.close(fig)
    
    print(f"Saved combined visualization to {combined_path}")
    
    # Close DeconvNet to free resources
    deconv_net.close()

def visualize_all_layers(model, image_path, max_features_per_layer=5, save_dir=None, model_name=None):
    """
    Visualize features from all convolutional layers in the model.
    
    Args:
        model: CNN model to visualize
        image_path: Path to input image
        max_features_per_layer: Maximum number of features to visualize per layer
        save_dir: Directory to save visualizations
        model_name: Name of the model (for saving)
        
    Returns:
        None
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Prepare image tensor for the model
    transform = get_transforms("val")
    image_tensor = transform(image_np).unsqueeze(0).to(device)
    
    # Setup DeconvNet
    deconv_net = DeconvNet(model, device)
    
    # Forward pass to compute feature maps
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Get all convolutional layers
    conv_layers = get_conv_layers(model)
    
    # Create directory for saving results
    if save_dir is None:
        save_dir = FIGURES_DIR / "deconv"
    os.makedirs(save_dir, exist_ok=True)
    
    # Add model name to directory if provided
    if model_name:
        save_dir = save_dir / model_name
        os.makedirs(save_dir, exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # For each convolutional layer
    for layer_name in conv_layers.keys():
        print(f"\nVisualizing layer: {layer_name}")
        
        # Skip if no feature maps for this layer
        if layer_name not in deconv_net.feature_maps:
            print(f"No feature maps found for layer {layer_name}")
            continue
            
        feature_map = deconv_net.feature_maps[layer_name]
        
        # Calculate activations
        activations = feature_map.mean(dim=(2, 3))[0]
        
        # Get indices of top activated features
        _, top_indices = torch.topk(activations, min(max_features_per_layer, feature_map.shape[1]))
        feature_indices = top_indices.cpu().numpy()
        
        # Visualize each feature
        all_results = []
        
        for idx in feature_indices:
            # Compute deconvolution for this feature
            print(f"Computing deconvolution for feature {idx}...")
            deconv_output = deconv_net.visualize_feature(layer_name, idx)
            
            # Add to results
            all_results.append((idx, deconv_output))
        
        # Create combined visualization
        fig = visualize_deconv_results(image_np, all_results)
        
        # Save combined visualization
        combined_path = save_dir / f"{layer_name.replace('.', '_')}_{image_name}.png"
        fig.savefig(combined_path)
        plt.close(fig)
        
        print(f"Saved layer visualization to {combined_path}")
    
    # Close DeconvNet to free resources
    deconv_net.close()

def visualize_feature_evolution(model, layer_name, feature_idx, input_variations, save_dir=None, model_name=None):
    """
    Visualize how a feature responds to variations in the input.
    
    Args:
        model: CNN model to visualize
        layer_name: Name of the layer to visualize
        feature_idx: Index of feature map to visualize
        input_variations: List of input images (paths or numpy arrays)
        save_dir: Directory to save visualizations
        model_name: Name of the model (for saving)
        
    Returns:
        None
    """
    # Setup DeconvNet
    deconv_net = DeconvNet(model, device)
    
    # Process input variations
    all_images = []
    all_deconv = []
    
    for i, input_var in enumerate(input_variations):
        # Load image if path is provided
        if isinstance(input_var, str):
            image = Image.open(input_var).convert("RGB")
            image_np = np.array(image)
            image_name = Path(input_var).stem
        else:
            image_np = input_var
            image_name = f"variation_{i}"
        
        # Prepare image tensor for the model
        transform = get_transforms("val")
        image_tensor = transform(image_np).unsqueeze(0).to(device)
        
        # Forward pass to compute feature maps
        with torch.no_grad():
            _ = model(image_tensor)
        
        # Compute deconvolution for this feature
        deconv_output = deconv_net.visualize_feature(layer_name, feature_idx)
        
        # Add to results
        all_images.append(image_np)
        all_deconv.append(deconv_output)
    
    # Create directory for saving results
    if save_dir is None:
        save_dir = FIGURES_DIR / "deconv"
    os.makedirs(save_dir, exist_ok=True)
    
    # Add model name to directory if provided
    if model_name:
        save_dir = save_dir / model_name
        os.makedirs(save_dir, exist_ok=True)
    
    # Create visualization grid
    fig, axes = plt.subplots(2, len(input_variations), figsize=(4*len(input_variations), 8))
    
    # Show original images in top row
    for i, img in enumerate(all_images):
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Input {i+1}")
        axes[0, i].axis('off')
    
    # Show deconv results in bottom row
    for i, deconv in enumerate(all_deconv):
        axes[1, i].imshow(deconv)
        axes[1, i].set_title(f"DeConv {i+1}")
        axes[1, i].axis('off')
    
    plt.suptitle(f"Feature {feature_idx} evolution in layer {layer_name}")
    plt.tight_layout()
    
    # Save visualization
    save_path = save_dir / f"{layer_name.replace('.', '_')}_feature_{feature_idx}_evolution.png"
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"Saved feature evolution visualization to {save_path}")
    
    # Close DeconvNet to free resources
    deconv_net.close()

def occlusion_sensitivity(model, image_path, label=None, patch_size=20, stride=10, save_dir=None, model_name=None):
    """
    Analyze model sensitivity to occlusion of input regions.
    
    Args:
        model: CNN model to analyze
        image_path: Path to input image
        label: True label of the image (if None, use predicted label)
        patch_size: Size of occlusion patch
        stride: Stride for moving the occlusion patch
        save_dir: Directory to save visualizations
        model_name: Name of the model (for saving)
        
    Returns:
        None
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Prepare image tensor for the model
    transform = get_transforms("val")
    image_tensor = transform(image_np).unsqueeze(0).to(device)
    
    # Get original prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # If label not provided, use predicted class
    if label is None:
        label = predicted_class
        
    # Get class name
    class_keys = list(CLASS_NAMES.keys())
    class_name = CLASS_NAMES[class_keys[label]]
    
    print(f"Analyzing occlusion sensitivity for class: {class_name} (index: {label})")
    
    # Create occlusion grid
    height, width = image_np.shape[:2]
    sensitivity_map = np.zeros((height, width))
    
    # Create image with occlusion
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Create copy of the image
            occluded = image_np.copy()
            
            # Apply occlusion
            occluded[y:y+patch_size, x:x+patch_size, :] = 0
            
            # Prepare occluded image
            occluded_tensor = transform(occluded).unsqueeze(0).to(device)
            
            # Get prediction for occluded image
            with torch.no_grad():
                outputs = model(occluded_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Record probability for target class
            target_prob = probabilities[label].item()
            
            # Update sensitivity map with average probability change
            sensitivity_map[y:y+patch_size, x:x+patch_size] -= target_prob
    
    # Normalize sensitivity map
    sensitivity_map = normalize_image(sensitivity_map)
    
    # Create directory for saving results
    if save_dir is None:
        save_dir = FIGURES_DIR / "occlusion"
    os.makedirs(save_dir, exist_ok=True)
    
    # Add model name to directory if provided
    if model_name:
        save_dir = save_dir / model_name
        os.makedirs(save_dir, exist_ok=True)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Sensitivity map
    axes[1].imshow(sensitivity_map, cmap='hot')
    axes[1].set_title("Occlusion Sensitivity")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image_np)
    axes[2].imshow(sensitivity_map, alpha=0.5, cmap='hot')
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.suptitle(f"Occlusion Sensitivity Analysis for {class_name}")
    plt.tight_layout()
    
    # Save visualization
    image_name = Path(image_path).stem
    save_path = save_dir / f"occlusion_{image_name}.png"
    plt.savefig(save_path)
    plt.close(fig)
    
    print(f"Saved occlusion sensitivity visualization to {save_path}")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize CNN models using DeConv")
    
    # Required arguments
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_idx", type=int, default=0, help="Index of the model to use")
    
    # Optional arguments
    parser.add_argument("--layer", type=str, default=None, help="Layer name to visualize")
    parser.add_argument("--feature", type=int, default=None, help="Feature index to visualize")
    parser.add_argument("--num_features", type=int, default=9, help="Number of top features to visualize")
    parser.add_argument("--all_layers", action="store_true", help="Visualize all convolutional layers")
    parser.add_argument("--occlusion", action="store_true", help="Perform occlusion sensitivity analysis")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Load model
    model, model_name = load_model(args.model_idx)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(visualize_model_architecture(model))
    
    # Perform requested visualizations
    if args.occlusion:
        # Occlusion sensitivity analysis
        occlusion_sensitivity(model, args.image, save_dir=args.save_dir, model_name=model_name)
    elif args.all_layers:
        # Visualize all layers
        visualize_all_layers(model, args.image, max_features_per_layer=args.num_features, 
                              save_dir=args.save_dir, model_name=model_name)
    else:
        # Visualize specific layer and features
        visualize_model_features(model, args.image, layer_name=args.layer, 
                                 feature_idx=args.feature, num_features=args.num_features, 
                                 save_dir=args.save_dir, model_name=model_name)


if __name__ == "__main__":
    main()