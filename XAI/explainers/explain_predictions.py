"""
Example script demonstrating how to use explainability methods for skin lesion classification models.
"""

import os
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse

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
from XAI.explainers import LimeExplainer, ShapExplainer, GradCamExplainer

def preprocess_image(image, transform=None):
    """
    Preprocess an image for model prediction.
    
    Args:
        image: PIL.Image or numpy.ndarray
        transform: Optional transform to apply
        
    Returns:
        tensor: Preprocessed image tensor
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # Apply transform if provided, otherwise use default transform
    if transform is None:
        transform = get_transforms("val")
        
    # Transform the image
    image_tensor = transform(image_np)
    
    return image_tensor

def predict_image(model, image_tensor, device=None):
    """
    Make a prediction for a single image.
    
    Args:
        model: PyTorch model
        image_tensor: Image tensor
        device: Device to run the model on
        
    Returns:
        tuple: (predicted_class, class_name, probabilities)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add batch dimension and move to device
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Convert class index to class name
    class_keys = list(CLASS_NAMES.keys())
    class_name = CLASS_NAMES[class_keys[predicted_class]]
    
    return predicted_class, class_name, probabilities.cpu().numpy()

def create_explanation_directory(model_name):
    """
    Create a directory for saving explanations.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path: Path to the explanations directory
    """
    explanations_dir = FIGURES_DIR / f"explanations_{model_name}"
    os.makedirs(explanations_dir, exist_ok=True)
    return explanations


def main():
    """Main function to explain model predictions."""
    parser = argparse.ArgumentParser(description="Explain model predictions for skin lesion images")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model_idx", type=int, default=-1, help="Index of the model to use (-1 for all models)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save explanations")
    
    args = parser.parse_args()
    
    # Create save directory if not provided
    if args.save_dir:
        save_dir = Path(args.save_dir)
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = FIGURES_DIR
        os.makedirs(save_dir, exist_ok=True)
    
    # Loop through models based on model_idx
    for i in range(0 if args.model_idx == -1 else args.model_idx, 
                   len(dl_models) if args.model_idx == -1 else args.model_idx + 1):
        model_name = dl_models[i].name()
        print(f"\n{'='*50}")
        print(f"Explaining Model: {model_name}")
        print(f"{'='*50}")
        
        # Create model with proper input size
        model = ResizedModel(dl_models[i].inputSize(), dl_models[i]()).to(device)
        
        # Load the best model weights
        best_model_path, checkpoint = load_best_model(model_name)
        
        if checkpoint is not None:
            # Load model weights
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {best_model_path}")
        else:
            print(f"No saved model found for {model_name}, using untrained model")
        
        # Create model-specific save directory
        model_save_dir = save_dir / model_name
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Generate explanations
        explanations = explain_with_all_methods(
            model, 
            args.image, 
            save_dir=model_save_dir,
            model_name=model_name
        )


if __name__ == "__main__":
    main()_dir

def explain_with_all_methods(model, image_path, save_dir=None, model_name=None):
    """
    Explain a prediction using all available explainability methods.
    
    Args:
        model: PyTorch model
        image_path: Path to the image file
        save_dir: Directory to save explanations
        model_name: Name of the model
        
    Returns:
        dict: Dictionary with all explanations
    """
    # Load and preprocess image
    image = np.array(Image.open(image_path).convert("RGB"))
    transform = get_transforms("val")
    image_tensor = transform(image)
    
    # Make prediction
    pred_class, class_name, probabilities = predict_image(model, image_tensor, device)
    print(f"Prediction: {class_name} (Class {pred_class})")
    
    # Create base save path
    if save_dir is None:
        save_dir = FIGURES_DIR
    
    if model_name is None:
        model_name = getattr(model, 'name', lambda: 'unknown')()
    
    image_name = Path(image_path).stem
    
    # Create a preprocessing function for explainers
    def preprocess_fn(img):
        return transform(img)
    
    # Initialize explainers
    lime_explainer = LimeExplainer(model, device, CLASS_NAMES, preprocess_fn)
    shap_explainer = ShapExplainer(model, device, CLASS_NAMES, preprocess_fn)
    
    # Find target layers for GradCAM
    try:
        gradcam_explainer = GradCamExplainer(model)
    except ValueError as e:
        print(f"Error initializing GradCAM: {e}")
        print("Skipping GradCAM explanation.")
        gradcam_explainer = None
    
    explanations = {}
    
    # 1. Generate LIME explanation
    print("\n=== Generating LIME explanation ===")
    lime_exp = lime_explainer.explain(image, num_samples=1000)
    lime_fig, _ = lime_explainer.visualize(
        lime_exp, 
        image, 
        label=pred_class,
        save_path=save_dir / f"lime_{model_name}_{image_name}.png"
    )
    explanations['lime'] = {'explanation': lime_exp, 'figure': lime_fig}
    
    # 2. Generate SHAP explanation
    print("\n=== Generating SHAP explanation ===")
    shap_values = shap_explainer.explain(image, n_samples=50)
    shap_fig, _ = shap_explainer.visualize(
        shap_values, 
        image, 
        label=pred_class,
        save_path=save_dir / f"shap_{model_name}_{image_name}.png"
    )
    explanations['shap'] = {'explanation': shap_values, 'figure': shap_fig}
    
    # 3. Generate GradCAM explanation if available
    if gradcam_explainer is not None:
        print("\n=== Generating GradCAM explanation ===")
        # Add batch dimension and move to device
        batch_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Generate GradCAM heatmap
        gradcam_heatmap = gradcam_explainer.explain(batch_tensor)
        
        # Convert image to [0, 1] range for visualization
        normalized_image = image.astype(float) / 255
        
        # Visualize
        gradcam_fig = plt.figure(figsize=(12, 5))
        cam_image = gradcam_explainer.visualize(
            gradcam_heatmap,
            normalized_image,
            class_name=class_name,
            save_path=save_dir / f"gradcam_{model_name}_{image_name}.png"
        )
        explanations['gradcam'] = {'explanation': gradcam_heatmap, 'figure': gradcam_fig}
    
    print("\nAll explanations generated and saved to:")
    print(f"  - {save_dir}")
    
    return explanations
