import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from XAI.features import extract_all_features
from XAI.modeling.models.ML_Base_model import BaseMLModel
from XAI.config import MODELS_DIR, CLASS_NAMES


def load_latest_model(model_name_prefix):
    """
    Load the latest model with the given prefix from the models directory
    
    Args:
        model_name_prefix: Start of the model filename (e.g., 'DecisionTree', 'RandomForest')
        
    Returns:
        Loaded model
    """
    # Get all model files with the given prefix
    model_files = list(MODELS_DIR.glob(f"{model_name_prefix}_*.pkl"))
    
    if not model_files:
        raise FileNotFoundError(f"No models found with prefix {model_name_prefix}")
    
    # Find the latest model based on timestamp in filename
    latest_model_path = max(model_files, key=os.path.getctime)
    print(f"Loading model from {latest_model_path}")
    
    # Load the model
    return BaseMLModel.load(latest_model_path)


def load_feature_scaler():
    """
    Load the feature scaler
    
    Returns:
        Loaded scaler
    """
    scaler_path = MODELS_DIR / "ml_scaler.joblib"
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    return joblib.load(scaler_path)


def predict_single_image(image_path, model, scaler):
    """
    Predict skin lesion class for a single image
    
    Args:
        image_path: Path to the image file
        model: Trained model
        scaler: Feature scaler
        
    Returns:
        Class prediction and probability
    """
    # Read and preprocess image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract features
    features = extract_all_features(image)
    features = features.reshape(1, -1)  # Reshape for single sample
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    class_idx = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get class name and probability
    class_name = class_idx
    prob = probabilities.max()
    
    # Map class index to class name if needed
    if isinstance(class_idx, (int, np.integer)):
        class_name = list(CLASS_NAMES.keys())[class_idx]
    
    return class_name, prob, probabilities


def visualize_prediction(image_path, class_name, probabilities, class_names):
    """
    Visualize the prediction with the image and probability bar chart
    
    Args:
        image_path: Path to the image file
        class_name: Predicted class name
        probabilities: Prediction probabilities for all classes
        class_names: List of all class names
    """
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title(f'Prediction: {class_name}')
    ax1.axis('off')
    
    # Display probabilities
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, probabilities, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    
    plt.tight_layout()
    plt.show()


def predict_batch_images(image_dir, model, scaler, limit=None):
    """
    Predict skin lesion classes for a batch of images
    
    Args:
        image_dir: Directory containing images
        model: Trained model
        scaler: Feature scaler
        limit: Maximum number of images to process (for testing)
        
    Returns:
        DataFrame with predictions
    """
    results = []
    image_paths = list(Path(image_dir).glob("*.jpg"))
    
    if limit:
        image_paths = image_paths[:limit]
    
    for image_path in image_paths:
        try:
            # Extract features
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features = extract_all_features(image)
            features = features.reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            # Make prediction
            class_idx = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Get class name and probability
            class_name = class_idx
            prob = probabilities.max()
            
            # Map class index to class name if needed
            if isinstance(class_idx, (int, np.integer)):
                class_name = list(CLASS_NAMES.keys())[class_idx]
            
            results.append({
                'image_path': image_path.name,
                'predicted_class': class_name,
                'confidence': prob
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return pd.DataFrame(results)


def main():
    """Demo prediction using the trained model"""
    try:
        # Load the model (use appropriate model name)
        model = load_latest_model("RandomForest")
        
        # Load the scaler
        scaler = load_feature_scaler()
        
        # Get sample image path (adjust as needed)
        sample_image_path = Path("notebooks/input.jpg")
        
        if not sample_image_path.exists():
            print(f"Sample image not found at {sample_image_path}")
            return
        
        # Make prediction
        class_name, probability, all_probs = predict_single_image(
            sample_image_path, model, scaler
        )
        
        print(f"Predicted class: {class_name}")
        print(f"Confidence: {probability:.4f}")
        
        # Visualize
        visualize_prediction(
            sample_image_path, 
            class_name, 
            all_probs, 
            list(CLASS_NAMES.keys())
        )
        
    except Exception as e:
        print(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
