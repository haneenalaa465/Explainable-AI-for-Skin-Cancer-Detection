"""
Prediction script for the skin lesion classification model.
"""

import os
import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
from tqdm import tqdm

from XAI.config import MODELS_DIR, FIGURES_DIR, CLASS_NAMES, MODEL_INPUT_SIZE



def load_model(model_path=None):
    """
    Load the trained model.

    Args:
        model_path (str): Path to the model checkpoint

    Returns:
        model: Loaded PyTorch model
    """
    if model_path is None:
        model_path = MODELS_DIR / "skin_lesion_cnn_best.pth"

    # Initialize model
    model = SkinLesionCNN()

    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Set model to evaluation mode
    model.eval()

    return model


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
        image = np.array(image)

    # Apply transformation if provided, otherwise use default
    if transform is None:
        transform = A.Compose(
            [
                A.Resize(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    # Apply transform
    image_tensor = transform(image=image)["image"]

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict_image(model, image, transform=None, device=None):
    """
    Make a prediction for a single image.

    Args:
        model: PyTorch model
        image: PIL.Image or numpy.ndarray
        transform: Optional transform to apply
        device: Device to run the model on

    Returns:
        tuple: (predicted_class, class_name, probabilities)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Preprocess image
    image_tensor = preprocess_image(image, transform)
    image_tensor = image_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    # Convert class index to class name
    class_keys = list(CLASS_NAMES.keys())
    class_name = CLASS_NAMES[class_keys[predicted_class]]

    # Convert probabilities to numpy
    probabilities = probabilities.cpu().numpy()

    return predicted_class, class_name, probabilities


def predict_batch(model, images, transform=None, device=None):
    """
    Make predictions for a batch of images.

    Args:
        model: PyTorch model
        images: List of PIL.Image or numpy.ndarray
        transform: Optional transform to apply
        device: Device to run the model on

    Returns:
        tuple: (predicted_classes, class_names, probabilities)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Preprocess images
    batch_tensors = []
    for image in images:
        image_tensor = preprocess_image(image, transform)
        batch_tensors.append(image_tensor)

    # Concatenate tensors into a batch
    batch_tensor = torch.cat(batch_tensors, dim=0).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()

    # Convert class indices to class names
    class_keys = list(CLASS_NAMES.keys())
    class_names = [CLASS_NAMES[class_keys[pred]] for pred in predicted_classes]

    # Convert probabilities to numpy
    probabilities = probabilities.cpu().numpy()

    return predicted_classes, class_names, probabilities


def plot_prediction(image, class_name, probabilities, save_path=None):
    """
    Plot an image with its prediction and class probabilities.

    Args:
        image: PIL.Image or numpy.ndarray
        class_name: Predicted class name
        probabilities: Array of class probabilities
        save_path: Path to save the plot
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot image
    ax1.imshow(image_np)
    ax1.set_title(f"Prediction: {class_name}")
    ax1.axis("off")

    # Plot class probabilities
    class_names = list(CLASS_NAMES.values())
    y_pos = np.arange(len(class_names))

    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel("Probability")
    ax2.set_title("Class Probabilities")

    # Sort by probability
    for i, v in enumerate(probabilities):
        ax2.text(v + 0.01, i, f"{v:.2f}", va="center")

    # Adjust layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction plot saved to {save_path}")

    plt.show()


def explain_prediction_lime(model, image, transform=None, num_samples=1000, save_path=None):
    """
    Explain the model's prediction using LIME.

    Args:
        model: PyTorch model
        image: PIL.Image or numpy.ndarray
        transform: Transform to apply to the image
        num_samples: Number of samples for LIME
        save_path: Path to save the explanation

    Returns:
        explanation: LIME explanation object
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Define predict function for LIME
    def predict_fn(images):
        batch_tensors = []
        for img in images:
            # Ensure RGB format (LIME may provide grayscale)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)

            # Preprocess
            if transform:
                img_tensor = transform(image=img)["image"].unsqueeze(0)
            else:
                img_tensor = preprocess_image(img, transform)

            batch_tensors.append(img_tensor)

        # Concatenate and predict
        batch_tensor = torch.cat(batch_tensors, dim=0)
        model.eval()
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

        return probs

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation
    print("Generating LIME explanation...")
    explanation = explainer.explain_instance(
        image_np, predict_fn, top_labels=len(CLASS_NAMES), hide_color=0, num_samples=num_samples
    )

    # Get prediction
    _, class_name, _ = predict_image(model, image_np, transform)

    # Get the top label
    top_label = explanation.top_labels[0]

    # Plot the explanation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Original image
    axs[0].imshow(image_np)
    axs[0].set_title(f"Original Image\nPrediction: {class_name}")
    axs[0].axis("off")

    # LIME explanation
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=5, hide_rest=False
    )
    axs[1].imshow(mark_boundaries(temp, mask))
    axs[1].set_title(f"LIME Explanation\nHighlights regions supporting the prediction")
    axs[1].axis("off")

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"LIME explanation saved to {save_path}")

    plt.show()

    return explanation


def explain_prediction_shap(model, image, bg_images=None, n_samples=100, save_path=None):
    """
    Explain the model's prediction using SHAP.

    Args:
        model: PyTorch model
        image: PIL.Image or numpy.ndarray
        bg_images: Background images for SHAP explainer
        n_samples: Number of samples for SHAP
        save_path: Path to save the explanation

    Returns:
        shap_values: SHAP values
    """
    # Preprocess input image
    input_tensor = preprocess_image(image)

    # Generate background if not provided
    if bg_images is None:
        # Create a simple background of zeros
        bg_tensor = torch.zeros((n_samples,) + input_tensor.shape[1:])
    else:
        # Process background images
        bg_tensors = []
        for bg_img in bg_images[:n_samples]:
            bg_tensor = preprocess_image(bg_img)
            bg_tensors.append(bg_tensor)
        bg_tensor = torch.cat(bg_tensors, dim=0)

    # Set model to evaluation mode
    model.eval()

    # Create the DeepExplainer
    print("Creating SHAP explainer...")
    explainer = shap.DeepExplainer(model, bg_tensor)

    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(input_tensor)

    # Get prediction
    _, class_name, _ = predict_image(model, image)

    # Plot the explanation
    plt.figure(figsize=(10, 6))

    # Combine the three RGB channels for visualization
    shap_combined = np.sum(np.abs(shap_values), axis=0)

    # Display original image
    plt.subplot(1, 2, 1)
    if isinstance(image, Image.Image):
        plt.imshow(image)
    else:
        plt.imshow(image)
    plt.title(f"Original Image\nPrediction: {class_name}")
    plt.axis("off")

    # Display SHAP values
    plt.subplot(1, 2, 2)
    plt.imshow(shap_combined[0].transpose(1, 2, 0), cmap="viridis")
    plt.title("SHAP Values\nHigher values indicate more influence on prediction")
    plt.axis("off")

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"SHAP explanation saved to {save_path}")

    plt.show()

    return shap_values


def main():
    """Main function for prediction and explanation."""
    # Load model
    model = load_model()

    # Example: Load an image for prediction
    image_path = input("Enter path to test image: ")
    image = Image.open(image_path).convert("RGB")

    # Make prediction
    _, class_name, probabilities = predict_image(model, image)

    # Plot prediction
    plot_path = FIGURES_DIR / "prediction_result.png"
    plot_prediction(image, class_name, probabilities, save_path=plot_path)

    # Explain prediction with LIME
    lime_path = FIGURES_DIR / "lime_explanation.png"
    explain_prediction_lime(model, image, save_path=lime_path)

    # Explain prediction with SHAP
    shap_path = FIGURES_DIR / "shap_explanation.png"
    explain_prediction_shap(model, image, save_path=shap_path)


if __name__ == "__main__":
    main()
