"""
Prediction script for the skin lesion classification model.
"""

import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
from shap.plots import image as shapImage
from tqdm import tqdm

from XAI.config import MODELS_DIR, FIGURES_DIR, CLASS_NAMES, MODEL_INPUT_SIZE
from XAI.dataset import get_transforms
from XAI.modeling.ResizeLayer import ResizedModel
from XAI.modeling.AllModels import dl_models, device
from XAI.modeling.train import load_best_model


from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


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


def get_target_layers(model):
    """
    Dynamically get the target layers for GradCAM based on the model.

    Args:
        model: PyTorch model

    Returns:
        list: List of target layers
    """
    # Example: Dynamically find the last convolutional layer
    for name, module in reversed(list(model.model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            print(f"Using layer '{name}' for GradCAM")
            return [module]
    raise ValueError("No convolutional layer found in the model.")


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

    # Apply transform
    image_tensor = get_transforms("val")(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict_image(model, image, transform=None):
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

    global device

    print(device)
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

            img_tensor = preprocess_image(img, transform)
            img_tensor = img_tensor.to(device)

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


def remove_inplace_from_model(model):
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False


def explain_prediction_shap(model, image, bg_images=None, n_samples=32, save_path=None):
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
    input_tensor = input_tensor.to(device)
    print(f"{input_tensor.device} {device}")

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
    bg_tensor = bg_tensor.to(device)
    model.eval()
    remove_inplace_from_model(model)

    with torch.no_grad():
        # Create the DeepExplainer
        print("Creating SHAP explainer...")
        print(f"{bg_tensor.device} {device}")
        explainer = shap.DeepExplainer(model, bg_tensor)

    # Calculate SHAP values

    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(input_tensor)

    # Get prediction
    predicted_class_idx, class_name, _ = predict_image(model, image)

    # Plot the explanation
    # plt.figure(figsize=(10, 6))

    # Combine the three RGB channels for visualization
    shap_combined = np.sum(np.abs(shap_values), axis=0)

    # # Display original image
    # plt.subplot(1, 2, 1)
    # if isinstance(image, Image.Image):
    #     plt.imshow(image)
    # else:
    #     plt.imshow(image)
    # plt.title(f"Original Image\nPrediction: {class_name}")
    # plt.axis("off")

    # Display SHAP values
    # shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
    # plt.subplot(1, 2, 2)
    shapImage(shap_values[0][0], image, show=False, width=10)
    plt.title("SHAP Values\nHigher values indicate more influence on prediction")

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"SHAP explanation saved to {save_path}")

    plt.show()

    return shap_values


def explain_prediction_gcam(model, image, save_path=None):
    """
    Explain the model's prediction using GradCam.

    Args:
        model: PyTorch model
        image: PIL.Image or numpy.ndarray
        layers: list of model's layers for explaination
        save_path: Path to save the explanation
    """

    input_tensor = get_transforms("val")(image).unsqueeze(0)
    print(input_tensor.shape)
    target_layers = get_target_layers(model)

    explainer = GradCAM(model=model, target_layers=target_layers)
    with torch.enable_grad():
        # Generate GradCAM heatmap
        grayscale_cam = explainer(
            input_tensor=input_tensor, targets=None
        )  # Default target is the predicted class
        grayscale_cam = grayscale_cam[0, :]  # Extract the first image's heatmap

    # Convert image to numpy for visualization
    image_np = np.array(image) / 255.0  # Normalize to [0, 1]
    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    # Save or display the GradCAM result
    if save_path:
        plt.imsave(save_path, cam_image)
        print(f"GradCAM explanation saved to {save_path}")

    plt.imshow(cam_image)
    plt.axis("off")
    plt.show()
def explain_prediction_gcamPP(model, image, save_path=None):
    """
    Explain the model's prediction using GradCam.

    Args:
        model: PyTorch model
        image: PIL.Image or numpy.ndarray
        layers: list of model's layers for explaination
        save_path: Path to save the explanation
    """

    input_tensor = get_transforms("val")(image).unsqueeze(0)
    print(input_tensor.shape)
    target_layers = get_target_layers(model)

    explainer = GradCAMPlusPlus(model=model, target_layers=target_layers)
    with torch.enable_grad():
        # Generate GradCAM heatmap
        grayscale_cam = explainer(
            input_tensor=input_tensor, targets=None
        )  # Default target is the predicted class
        grayscale_cam = grayscale_cam[0, :]  # Extract the first image's heatmap

    # Convert image to numpy for visualization
    image_np = np.array(image) / 255.0  # Normalize to [0, 1]
    cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)

    # Save or display the GradCAM result
    if save_path:
        plt.imsave(save_path, cam_image)
        print(f"GradCAM explanation saved to {save_path}")

    plt.imshow(cam_image)
    plt.axis("off")
    plt.show()


def main(model_idx=-1):
    """Main function for prediction and explanation."""
    # Load model
    for i in range(
        0 if model_idx == -1 else model_idx, len(dl_models) if model_idx == -1 else model_idx + 1
    ):
        model_name = dl_models[i].name()
        print(f"Explaining Model {model_name} with input size {dl_models[i].inputSize()}")

        currentModel = ResizedModel(dl_models[i].inputSize(), dl_models[i]()).to(device)
        # Check if we have a saved model and load it
        best_model_path, checkpoint = load_best_model(dl_models[i].name())
        start_epoch = 0
        best_val_acc = 0.0

        if checkpoint is not None:
            # Load model weights
            currentModel.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise Exception(
                "You have to train the model\nPlease use make train to train all models or refer to the readme in the repository"
            )

        # Example: Load an image for prediction
        image_path = input("Enter path to test image: ")
        image = np.array(Image.open(image_path).convert("RGB"))

        image_path = Path(image_path)

        currentModel.eval()
        for param in currentModel.parameters():
            param.requires_grad = True
        # Make prediction
        _, class_name, probabilities = predict_image(currentModel, image)

        # Plot prediction
        plot_path = FIGURES_DIR / f"prediction_{image_path.stem}_{model_name}_result.png"
        plot_prediction(image, class_name, probabilities, save_path=plot_path)

        # Explain prediction with GradCam
        gradcam_path = FIGURES_DIR / f"gradcam_{image_path.stem}_{model_name}_explanation.png"
        explain_prediction_gcam(currentModel, image, gradcam_path)
        
        # Explain prediction with GradCam
        gradcam_path = FIGURES_DIR / f"gradcam++_{image_path.stem}_{model_name}_explanation.png"
        explain_prediction_gcamPP(currentModel, image, gradcam_path)

        # Explain prediction with LIME
        lime_path = FIGURES_DIR / f"lime_{image_path.stem}_{model_name}_explanation.png"
        explain_prediction_lime(currentModel, image, save_path=lime_path)

        # Explain prediction with SHAP
        shap_path = FIGURES_DIR / f"shap_{image_path.stem}_{model_name}_explanation.png"
        explain_prediction_shap(currentModel, image, save_path=shap_path)


if __name__ == "__main__":
    main()
