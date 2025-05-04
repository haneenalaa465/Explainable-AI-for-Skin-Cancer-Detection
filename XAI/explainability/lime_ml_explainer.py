"""
LIME (Local Interpretable Model-agnostic Explanations) for skin lesion classification models.

This module provides functions to explain model predictions using LIME, which helps
understand which parts of an image contribute to a specific prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch
from tqdm import tqdm

class LimeExplainer:
    """
    A class to generate and visualize LIME explanations for image classification models.
    """
    
    def __init__(self, model, device, class_names, preprocess_fn=None):
        """
        Initialize the LIME explainer.
        
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
        self.explainer = lime_image.LimeImageExplainer()
        
    def predict_fn(self, images):
        """
        Prediction function for LIME.
        
        Args:
            images: Batch of images (numpy arrays)
            
        Returns:
            numpy.ndarray: Predicted probabilities
        """
        batch_tensors = []
        for img in images:
            # Ensure RGB format (LIME may provide grayscale)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
                
            # Apply preprocessing if provided
            if self.preprocess_fn:
                img_tensor = self.preprocess_fn(img)
            else:
                # Default preprocessing (assumes a transformation that converts to tensor)
                img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
                
            img_tensor = img_tensor.to(self.device)
            batch_tensors.append(img_tensor.unsqueeze(0))  # Add batch dimension
            
        # Concatenate and predict
        batch_tensor = torch.cat(batch_tensors, dim=0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            
        return probs
        
    def explain(self, image, num_samples=1000, top_labels=5, hide_color=0, positive_only=True, num_features=5):
        """
        Generate a LIME explanation for an image.
        
        Args:
            image: Input image (numpy array)
            num_samples: Number of perturbed samples to generate
            top_labels: Number of top labels to explain
            hide_color: Color to use for hiding superpixels (0 for black)
            positive_only: Whether to only show positive contributions
            num_features: Number of features (superpixels) to include in explanation
            
        Returns:
            lime_explanation: LIME explanation object
        """
        print("Generating LIME explanation...")
        explanation = self.explainer.explain_instance(
            image, 
            self.predict_fn,
            top_labels=top_labels if top_labels is not None else len(self.class_names), 
            hide_color=hide_color, 
            num_samples=num_samples
        )
        
        return explanation
    
    def visualize(self, explanation, image, label=None, save_path=None, positive_only=True, num_features=5):
        """
        Visualize a LIME explanation.
        
        Args:
            explanation: LIME explanation object
            image: Original image
            label: Label to explain (if None, uses the top predicted label)
            save_path: Path to save the visualization
            positive_only: Whether to only show positive contributions
            num_features: Number of features (superpixels) to include in explanation
            
        Returns:
            None (displays the visualization)
        """
        # Get the top predicted label if not provided
        if label is None:
            label = explanation.top_labels[0]
            
        # Get class name
        class_name = label
        if isinstance(self.class_names, dict):
            class_keys = list(self.class_names.keys())
            if isinstance(label, int) and label < len(class_keys):
                class_name = self.class_names[class_keys[label]]
        elif isinstance(self.class_names, list) and isinstance(label, int) and label < len(self.class_names):
            class_name = self.class_names[label]
            
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axs[0].imshow(image)
        axs[0].set_title(f"Original Image")
        axs[0].axis("off")
        
        # LIME explanation
        temp, mask = explanation.get_image_and_mask(
            label, 
            positive_only=positive_only, 
            num_features=num_features, 
            hide_rest=False
        )
        axs[1].imshow(mark_boundaries(temp, mask))
        axs[1].set_title(f"LIME Explanation for '{class_name}'")
        axs[1].axis("off")
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"LIME explanation saved to {save_path}")
            
        return fig, axs