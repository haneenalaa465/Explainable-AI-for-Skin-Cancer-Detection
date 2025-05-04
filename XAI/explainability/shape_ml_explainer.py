"""
SHAP (SHapley Additive exPlanations) implementation for skin lesion classifiers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import shap
from tqdm import tqdm

from XAI.features import extract_all_features
from XAI.modeling.predict_ml import load_latest_model, load_feature_scaler
from XAI.config import REPORTS_DIR, CLASS_NAMES
from XAI.explainability.feature_importance import get_feature_names


def setup_shap_explainer(model, background_data=None):
    """
    Set up a SHAP explainer for the model.
    
    Args:
        model: Trained ML model
        background_data: Background data for SHAP explainer (sample of training data)
        
    Returns:
        shap.Explainer: Configured SHAP explainer
    """
    # Check model type
    model_type = model.name()
    
    if model_type == "RandomForest":
        # For tree models, use TreeExplainer
        # Unwrap sklearn model from the BaseMLModel wrapper
        sklearn_model = model.model
        explainer = shap.TreeExplainer(sklearn_model, background_data)
    elif model_type == "DecisionTree":
        # For tree models, use TreeExplainer
        sklearn_model = model.model
        explainer = shap.TreeExplainer(sklearn_model, background_data)
    else:
        # For other models, use KernelExplainer with background data
        if background_data is None:
            raise ValueError("Background data is required for non-tree models")
        
        # Define prediction function
        def predict_fn(X):
            return model.predict_proba(X)
        
        explainer = shap.KernelExplainer(predict_fn, background_data)
    
    return explainer


def generate_shap_values(explainer, X):
    """
    Generate SHAP values for the given instances.
    
    Args:
        explainer: SHAP explainer
        X: Feature values to explain
        
    Returns:
        shap.Explanation: SHAP values
    """
    # Generate SHAP values
    shap_values = explainer(X)
    
    return shap_values


def explain_prediction_with_shap(image_path, model, scaler, explainer=None, 
                               background_data=None, feature_names=None):
    """
    Explain a prediction for a single image using SHAP.
    
    Args:
        image_path: Path to image file
        model: Trained model
        scaler: Feature scaler
        explainer: SHAP explainer (if None, a new one will be created)
        background_data: Background data for SHAP explainer
        feature_names: List of feature names
        
    Returns:
        Tuple of (shap_values, image, class_name, probabilities, features_scaled)
    """
    # Read and preprocess image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract features
    features = extract_all_features(image)
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    class_idx = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get class name
    class_name = class_idx
    if isinstance(class_idx, (int, np.integer)):
        class_name = list(CLASS_NAMES.keys())[class_idx]
    
    # Set up SHAP explainer if not provided
    if explainer is None:
        explainer = setup_shap_explainer(model, background_data)
    
    # Generate SHAP values
    shap_values = generate_shap_values(explainer, features_scaled)
    
    return shap_values, image, class_name, probabilities, features_scaled


def visualize_shap_summary(explainer, shap_values, features, feature_names=None, 
                          max_display=20, class_idx=0, save_path=None):
    """
    Visualize SHAP summary plot.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        features: Feature values
        feature_names: List of feature names
        max_display: Maximum number of features to display
        class_idx: Index of class to explain
        save_path: Path to save the visualization
    """
    if feature_names is None:
        feature_names = get_feature_names()
    
    plt.figure(figsize=(12, 10))
    
    # For multi-class models, SHAP values might have an extra dimension for classes
    # We need to select the class of interest
    if hasattr(shap_values, 'shape') and len(shap_values.shape) > 2:
        # For shap.Explanation objects
        if isinstance(shap_values, shap.Explanation):
            if 'classes' in shap_values.output_names:
                # If there's a 'classes' dimension, select the class of interest
                class_values = shap_values[:, :, class_idx]
                shap.summary_plot(class_values, features, feature_names=feature_names, 
                               max_display=max_display, show=False)
            else:
                # Otherwise, just plot the values as is
                shap.summary_plot(shap_values, features, feature_names=feature_names, 
                               max_display=max_display, show=False)
        else:
            # For old-style shap values (list of arrays)
            shap.summary_plot(shap_values[class_idx], features, feature_names=feature_names, 
                           max_display=max_display, show=False)
    else:
        # For binary classification or single output
        shap.summary_plot(shap_values, features, feature_names=feature_names, 
                       max_display=max_display, show=False)
    
    plt.title(f'SHAP Summary Plot for Class: {list(CLASS_NAMES.keys())[class_idx]}')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
    
    plt.show()


def visualize_shap_waterfall(shap_values, feature_names=None, sample_idx=0, 
                           class_idx=0, max_display=20, save_path=None):
    """
    Visualize SHAP waterfall plot for a single prediction.
    
    Args:
        shap_values: SHAP values
        feature_names: List of feature names
        sample_idx: Index of sample to explain
        class_idx: Index of class to explain
        max_display: Maximum number of features to display
        save_path: Path to save the visualization
    """
    if feature_names is None:
        feature_names = get_feature_names()
    
    plt.figure(figsize=(12, 10))
    
    # Extract shap values for the sample and class
    if isinstance(shap_values, shap.Explanation):
        # For newer SHAP versions
        if len(shap_values.shape) > 2:  # Multi-class
            sample_values = shap_values[sample_idx, :, class_idx]
        else:  # Binary or single output
            sample_values = shap_values[sample_idx]
        
        # Plot waterfall
        shap.plots.waterfall(sample_values, max_display=max_display, show=False)
    else:
        # For older SHAP versions
        if isinstance(shap_values, list):  # Multi-class
            sample_values = shap_values[class_idx][sample_idx]
        else:  # Binary
            sample_values = shap_values[sample_idx]
        
        # Create an Explanation object
        exp = shap.Explanation(values=sample_values, 
                              base_values=np.zeros(1),  # Placeholder
                              data=np.zeros((1, len(sample_values))),  # Placeholder
                              feature_names=feature_names[:len(sample_values)])
        
        # Plot waterfall
        shap.plots.waterfall(exp, max_display=max_display, show=False)
    
    plt.title(f'SHAP Waterfall Plot for Sample {sample_idx}, Class: {list(CLASS_NAMES.keys())[class_idx]}')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP waterfall plot saved to {save_path}")
    
    plt.show()


def visualize_shap_force(explainer, shap_values, features, feature_names=None, 
                        sample_idx=0, class_idx=0, save_path=None):
    """
    Visualize SHAP force plot for a single prediction.
    
    Args:
        explainer: SHAP explainer
        shap_values: SHAP values
        features: Feature values
        feature_names: List of feature names
        sample_idx: Index of sample to explain
        class_idx: Index of class to explain
        save_path: Path to save the visualization
    """
    if feature_names is None:
        feature_names = get_feature_names()
    
    plt.figure(figsize=(20, 3))
    
    # Extract shap values for the sample and class
    if isinstance(shap_values, shap.Explanation):
        # For newer SHAP versions
        if len(shap_values.shape) > 2:  # Multi-class
            sample_values = shap_values[sample_idx, :, class_idx]
        else:  # Binary or single output
            sample_values = shap_values[sample_idx]
        
        # Plot force
        shap.plots.force(sample_values, features[sample_idx], feature_names=feature_names, 
                         matplotlib=True, show=False)
    else:
        # For older SHAP versions
        if isinstance(shap_values, list):  # Multi-class
            sample_values = shap_values[class_idx][sample_idx]
        else:  # Binary
            sample_values = shap_values[sample_idx]
        
        # Plot force
        shap.force_plot(explainer.expected_value[class_idx], 
                      sample_values, 
                      features[sample_idx], 
                      feature_names=feature_names,
                      matplotlib=True, show=False)
    
    plt.title(f'SHAP Force Plot for Sample {sample_idx}, Class: {list(CLASS_NAMES.keys())[class_idx]}')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP force plot saved to {save_path}")
    
    plt.show()


def visualize_shap_decision_plot(shap_values, features, feature_names=None, 
                               sample_idx=0, class_idx=0, save_path=None):
    """
    Visualize SHAP decision plot for a single prediction.
    
    Args:
        shap_values: SHAP values
        features: Feature values
        feature_names: List of feature names
        sample_idx: Index of sample to explain
        class_idx: Index of class to explain
        save_path: Path to save the visualization
    """
    if feature_names is None:
        feature_names = get_feature_names()
    
    plt.figure(figsize=(12, 10))
    
    # Extract shap values for the sample and class
    if isinstance(shap_values, shap.Explanation):
        # For newer SHAP versions
        if len(shap_values.shape) > 2:  # Multi-class
            sample_values = shap_values[sample_idx, :, class_idx]
        else:  # Binary or single output
            sample_values = shap_values[sample_idx]
        
        # Plot decision
        shap.decision_plot(base_value=sample_values.base_values, 
                         shap_values=sample_values.values, 
                         features=features[sample_idx], 
                         feature_names=feature_names, 
                         show=False)
    else:
        # For older SHAP versions
        if isinstance(shap_values, list):  # Multi-class
            base_value = explainer.expected_value[class_idx]
            sample_values = shap_values[class_idx][sample_idx]
        else:  # Binary
            base_value = explainer.expected_value
            sample_values = shap_values[sample_idx]
        
        # Plot decision
        shap.decision_plot(base_value, sample_values, 
                         features=features[sample_idx],
                         feature_names=feature_names,
                         show=False)
    
    plt.title(f'SHAP Decision Plot for Sample {sample_idx}, Class: {list(CLASS_NAMES.keys())[class_idx]}')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP decision plot saved to {save_path}")
    
    plt.show()


def visualize_shap_explanations(shap_values, image, features, class_name, probabilities, 
                              feature_names=None, save_dir=None):
    """
    Create a comprehensive SHAP visualization for a single image prediction.
    
    Args:
        shap_values: SHAP values
        image: Original image
        features: Feature values
        class_name: Predicted class name
        probabilities: Prediction probabilities
        feature_names: List of feature names
        save_dir: Directory to save visualizations
    """
    if feature_names is None:
        feature_names = get_feature_names()
    
    # Find class index
    if isinstance(class_name, str):
        class_idx = list(CLASS_NAMES.keys()).index(class_name)
    else:
        class_idx = class_name
        class_name = list(CLASS_NAMES.keys())[class_idx]
    
    # Create save directory if provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Set up figure
    plt.figure(figsize=(16, 12))
    
    # 2. Display the image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title(f'Prediction: {CLASS_NAMES.get(class_name, class_name)}\n'
              f'Confidence: {probabilities.max():.2f}')
    plt.axis('off')
    
    # 3. Display SHAP summary plot (top 10 features)
    plt.subplot(2, 2, 2)
    
    # For multi-class models, SHAP values might have an extra dimension for classes
    if isinstance(shap_values, shap.Explanation):
        # For newer SHAP versions
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 2:
            # Extract values for the predicted class
            shap_class_values = shap_values[:, :, class_idx]
            
            # Create a temporary figure for the SHAP summary
            temp_fig = plt.figure()
            shap.summary_plot(shap_class_values, features, feature_names=feature_names, 
                           max_display=10, show=False)
            
            # Capture the plot and add it to our main figure
            temp_fig.canvas.draw()
            summary_plot = np.array(temp_fig.canvas.renderer.buffer_rgba())
            plt.close(temp_fig)
            
            plt.imshow(summary_plot)
            plt.axis('off')
            plt.title('Top 10 Features (SHAP Summary)')
        else:
            # For binary or single output
            temp_fig = plt.figure()
            shap.summary_plot(shap_values, features, feature_names=feature_names, 
                           max_display=10, show=False)
            temp_fig.canvas.draw()
            summary_plot = np.array(temp_fig.canvas.renderer.buffer_rgba())
            plt.close(temp_fig)
            
            plt.imshow(summary_plot)
            plt.axis('off')
            plt.title('Top 10 Features (SHAP Summary)')
    else:
        # For old-style shap values (list of arrays)
        temp_fig = plt.figure()
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[class_idx], features, feature_names=feature_names, 
                           max_display=10, show=False)
        else:
            shap.summary_plot(shap_values, features, feature_names=feature_names, 
                           max_display=10, show=False)
        temp_fig.canvas.draw()
        summary_plot = np.array(temp_fig.canvas.renderer.buffer_rgba())
        plt.close(temp_fig)
        
        plt.imshow(summary_plot)
        plt.axis('off')
        plt.title('Top 10 Features (SHAP Summary)')
    
    # 4. Display feature importance (bar chart)
    plt.subplot(2, 2, 3)
    
    # Extract absolute SHAP values and find top features
    if isinstance(shap_values, shap.Explanation):
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 2:
            # For multi-class, get values for the predicted class
            shap_abs = np.abs(shap_values.values[:, :, class_idx].mean(axis=0))
        else:
            # For binary or single output
            shap_abs = np.abs(shap_values.values.mean(axis=0))
    else:
        # For old-style values
        if isinstance(shap_values, list):
            shap_abs = np.abs(shap_values[class_idx]).mean(axis=0)
        else:
            shap_abs = np.abs(shap_values).mean(axis=0)
    
    # Get top 10 features and their importances
    top_indices = np.argsort(shap_abs)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = shap_abs[top_indices]
    
    # Create bar chart
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importances, align='center')
    plt.yticks(y_pos, top_features)
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 10 Features by Importance')
    
    # 5. Display class probabilities
    plt.subplot(2, 2, 4)
    if isinstance(probabilities, np.ndarray) and len(probabilities) > 1:
        class_names = list(CLASS_NAMES.values())
        y_pos = np.arange(len(class_names))
        plt.barh(y_pos, probabilities, align='center')
        plt.yticks(y_pos, class_names)
        plt.xlabel('Probability')
        plt.title('Class Probabilities')
    
    plt.tight_layout()
    
    # Save the full visualization if save_dir provided
    if save_dir:
        plt.savefig(save_dir / "shap_explanation.png", dpi=300, bbox_inches='tight')
        print(f"SHAP explanation saved to {save_dir / 'shap_explanation.png'}")
    
    plt.show()
    
    # Create additional SHAP visualizations if save_dir provided
    if save_dir:
        # Waterfall plot
        try:
            plt.figure(figsize=(12, 10))
            if isinstance(shap_values, shap.Explanation):
                if hasattr(shap_values, 'shape') and len(shap_values.shape) > 2:
                    sample_values = shap_values[0, :, class_idx]
                else:
                    sample_values = shap_values[0]
                shap.plots.waterfall(sample_values, max_display=10, show=False)
            plt.title(f'SHAP Waterfall Plot for Class: {CLASS_NAMES.get(class_name, class_name)}')
            plt.savefig(save_dir / "shap_waterfall.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating waterfall plot: {e}")
        
        # Decision plot
        try:
            plt.figure(figsize=(12, 10))
            if isinstance(shap_values, shap.Explanation):
                if hasattr(shap_values, 'shape') and len(shap_values.shape) > 2:
                    sample_values = shap_values[0, :, class_idx]
                else:
                    sample_values = shap_values[0]
                shap.decision_plot(base_value=sample_values.base_values, 
                                 shap_values=sample_values.values, 
                                 features=features[0], 
                                 feature_names=feature_names, 
                                 show=False)
            plt.title(f'SHAP Decision Plot for Class: {CLASS_NAMES.get(class_name, class_name)}')
            plt.savefig(save_dir / "shap_decision.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating decision plot: {e}")


def create_shap_explanation_report(image_paths, model, scaler, background_data=None, 
                                 output_dir=None, feature_names=None):
    """
    Create SHAP explanation reports for multiple images.
    
    Args:
        image_paths: List of paths to image files
        model: Trained model
        scaler: Feature scaler
        background_data: Background data for SHAP explainer
        output_dir: Directory to save reports
        feature_names: List of feature names
    """
    if output_dir is None:
        output_dir = REPORTS_DIR / "explainability" / "shap"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up SHAP explainer once to reuse
    explainer = setup_shap_explainer(model, background_data)
    
    # Process each image
    for i, image_path in enumerate(tqdm(image_paths, desc="Generating SHAP explanations")):
        try:
            # Create output directory for this image
            img_output_dir = Path(output_dir) / Path(image_path).stem
            img_output_dir.mkdir(exist_ok=True)
            
            # Generate explanation
            shap_values, image, class_name, probabilities, features_scaled = explain_prediction_with_shap(
                image_path, model, scaler, explainer, background_data, feature_names
            )
            
            # Visualize
            visualize_shap_explanations(
                shap_values, image, features_scaled, class_name, probabilities, 
                feature_names, img_output_dir
            )
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def main():
    """Demo SHAP explanations for the trained model"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='SHAP Explanations')
    parser.add_argument('--model', type=str, default='RandomForest', 
                      help='Model type (DecisionTree or RandomForest)')
    parser.add_argument('--image', type=str, default=None,
                      help='Path to an image file for explanation')
    parser.add_argument('--image_dir', type=str, default=None,
                      help='Directory with images to explain')
    parser.add_argument('--background_data', type=str, default=None,
                      help='Path to background data (pickle file) for SHAP explainer')
    args = parser.parse_args()
    
    try:
        # Load the model
        model = load_latest_model(args.model)
        
        # Load the scaler
        scaler = load_feature_scaler()
        
        # Create output directory
        output_dir = REPORTS_DIR / "explainability" / "shap"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load background data if provided
        background_data = None
        if args.background_data:
            print(f"Loading background data from {args.background_data}")
            background_df = pd.read_pickle(args.background_data)
            if 'features' in background_df.columns:
                # Extract features from DataFrame
                features_list = background_df['features'].tolist()
                background_data = np.stack(features_list)
                # Scale features
                background_data = scaler.transform(background_data)
            else:
                print("Warning: 'features' column not found in background data")
        
        # Get feature names
        feature_names = get_feature_names()
        
        # Explain a single image
        if args.image:
            image_path = Path(args.image)
            if not image_path.exists():
                print(f"Image not found at {image_path}")
                return
            
            # Create output directory for this image
            img_output_dir = output_dir / image_path.stem
            img_output_dir.mkdir(exist_ok=True)
            
            # Generate explanation
            shap_values, image, class_name, probabilities, features_scaled = explain_prediction_with_shap(
                image_path, model, scaler, background_data=background_data, feature_names=feature_names
            )
            
            # Visualize
            visualize_shap_explanations(
                shap_values, image, features_scaled, class_name, probabilities, 
                feature_names, img_output_dir
            )
        
        # Explain multiple images
        elif args.image_dir:
            image_dir = Path(args.image_dir)
            if not image_dir.exists():
                print(f"Directory not found: {image_dir}")
                return
            
            # Get all images in directory
            image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            
            if not image_paths:
                print(f"No images found in {image_dir}")
                return
            
            # Create explanations
            create_shap_explanation_report(
                image_paths, model, scaler, 
                background_data=background_data,
                output_dir=output_dir,
                feature_names=feature_names
            )
        
        else:
            print("Please provide either --image or --image_dir")
        
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        raise


if __name__ == "__main__":
    main()