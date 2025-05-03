"""
Test script for GradCAM explainer.
"""

import os
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
import traceback

from XAI.config import (
    MODELS_DIR, 
    FIGURES_DIR, 
    CLASS_NAMES, 
    MODEL_INPUT_SIZE
)
from XAI.dataset import get_transforms
from XAI.modeling.ResizeLayer import ResizedModel
from XAI.modeling.AllModels import dl_models, device
from XAI.modeling.train import load_best_model
from XAI.explainers.gradcam_explainer import GradCamExplainer

def test_gradcam(model_idx=0, image_path=None):
    """
    Test GradCAM explainer on a single model and image.
    
    Args:
        model_idx: Index of the model to use
        image_path: Path to the image file (if None, user will be prompted)
    """
    # Get model
    model_class = dl_models[model_idx]
    model_name = model_class.name()
    print(f"Testing GradCAM on model: {model_name}")
    
    # Create model with proper input size
    model = ResizedModel(model_class.inputSize(), model_class()).to(device)
    
    # Load the best model weights
    best_model_path, checkpoint = load_best_model(model_name)
    
    if checkpoint is not None:
        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {best_model_path}")
    else:
        print(f"No saved model found for {model_name}, using untrained model")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get image path
    if image_path is None:
        image_path = input("Enter path to test image: ")
    
    # Load and preprocess image
    image = np.array(Image.open(image_path).convert("RGB"))
    transform = get_transforms("val")
    image_tensor = transform(image)
    
    # Add batch dimension and move to device
    batch_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Get class name
    class_keys = list(CLASS_NAMES.keys())
    class_name = CLASS_NAMES[class_keys[predicted_class]]
    print(f"Predicted class: {class_name} (index: {predicted_class})")
    
    # Try generating GradCAM explanation
    try:
        print("\nCreating GradCAM explainer...")
        gradcam_explainer = GradCamExplainer(model)
        
        print("Generating GradCAM explanation...")
        gradcam_heatmap = gradcam_explainer.explain(batch_tensor)
        
        # Convert image to [0, 1] range for visualization
        normalized_image = image.astype(float) / 255
        
        print("Visualizing GradCAM explanation...")
        # Create a save path
        save_dir = FIGURES_DIR
        os.makedirs(save_dir, exist_ok=True)
        image_name = Path(image_path).stem
        save_path = save_dir / f"gradcam_test_{model_name}_{image_name}.png"
        
        # Visualize
        plt.figure(figsize=(12, 5))
        cam_image = gradcam_explainer.visualize(
            gradcam_heatmap,
            normalized_image,
            class_name=class_name,
            save_path=save_path
        )
        
        print(f"GradCAM explanation saved to {save_path}")
        
    except Exception as e:
        print(f"Error generating GradCAM explanation: {e}")
        print(traceback.format_exc())
        
        # Try to print the model architecture to help with debugging
        print("\nModel architecture:")
        print(model)
        
        # Check if the model has convolutional layers
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
                
        if conv_layers:
            print("\nFound convolutional layers:")
            for name, layer in conv_layers:
                print(f"  - {name}: {layer}")
        else:
            print("\nNo convolutional layers found in the model!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test GradCAM explainer")
    parser.add_argument("--model_idx", type=int, default=0, help="Index of the model to use")
    parser.add_argument("--image", type=str, default=None, help="Path to the image file")
    
    args = parser.parse_args()
    
    test_gradcam(args.model_idx, args.image)