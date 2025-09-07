"""
Script to run all explainability techniques on a skin lesion image.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from XAI.modeling.predict_ml import load_latest_model, load_feature_scaler
from XAI.features import extract_all_features
from XAI.config import REPORTS_DIR, CLASS_NAMES

# Import explainability modules
from XAI.explainability.feature_importance import (
    explain_image_with_feature_importance,
    get_feature_names
)
from XAI.explainability.lime_explainer import (
    explain_prediction_with_lime,
    visualize_lime_explanation
)
from XAI.explainability.shap_explainer import (
    explain_prediction_with_shap,
    visualize_shap_explanations
)
from XAI.explainability.pdp_explainer import (
    load_feature_dataset,
    explain_top_features_for_image
)


def load_background_data(feature_path=None, sample_size=100):
    """Load and prepare background data for explainers"""
    # Load feature dataset
    X, y = load_feature_dataset(feature_path, sample_size=sample_size)
    
    # Load scaler
    scaler = load_feature_scaler()
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y


def analyze_image(image_path, model_type='RandomForest', 
                feature_path=None, output_dir=None):
    """
    Analyze a skin lesion image using all explainability techniques.
    
    Args:
        image_path: Path to the image file
        model_type: Type of model to use (DecisionTree or RandomForest)
        feature_path: Path to features pickle file for background data
        output_dir: Directory to save results
    """
    image_path = Path(image_path)
    
    # Create output directory
    if output_dir is None:
        output_dir = REPORTS_DIR / "explainability" / image_path.stem
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each technique
    feature_imp_dir = output_dir / "feature_importance"
    lime_dir = output_dir / "lime"
    shap_dir = output_dir / "shap"
    pdp_dir = output_dir / "pdp"
    
    for subdir in [feature_imp_dir, lime_dir, shap_dir, pdp_dir]:
        subdir.mkdir(exist_ok=True)
    
    # Load model and scaler
    print(f"Loading {model_type} model...")
    model = load_latest_model(model_type)
    scaler = load_feature_scaler()
    
    # Load background data for LIME, SHAP and PDP
    print("Loading background data...")
    X_background, y_background = load_background_data(feature_path, sample_size=100)
    
    # Get feature names
    feature_names = get_feature_names()
    
    # Read image
    print(f"Analyzing image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Feature Importance Analysis
    print("Running Feature Importance analysis...")
    class_name, prob, feature_df = explain_image_with_feature_importance(
        image_path, model, scaler, top_n=15
    )
    
    # Save feature importance data
    feature_df.to_csv(feature_imp_dir / "feature_importance.csv")
    plt.savefig(feature_imp_dir / "feature_importance_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. LIME Explanation
    print("Running LIME analysis...")
    lime_explanation, _, _, lime_probs = explain_prediction_with_lime(
        image_path, model, scaler, num_features=15, training_data=X_background
    )
    
    # Visualize LIME
    visualize_lime_explanation(
        lime_explanation, image, class_name, lime_probs, 
        feature_names, save_path=lime_dir / "lime_explanation.png"
    )
    plt.close()
    
    # 3. SHAP Explanation
    print("Running SHAP analysis...")
    shap_values, _, _, shap_probs, features_scaled = explain_prediction_with_shap(
        image_path, model, scaler, background_data=X_background
    )
    
    # Visualize SHAP
    visualize_shap_explanations(
        shap_values, image, features_scaled, class_name, 
        shap_probs, feature_names, save_dir=shap_dir
    )
    plt.close()
    
    # 4. PDP Explanation
    print("Running PDP analysis...")
    _, _, _, _ = explain_top_features_for_image(
        image_path, model, scaler, X_background, 
        feature_names=feature_names,
        n_top_features=5,
        save_dir=pdp_dir
    )
    plt.close()
    
    # Generate combined report
    generate_combined_report(
        image_path, image, class_name, prob, 
        feature_imp_dir, lime_dir, shap_dir, pdp_dir,
        output_dir
    )
    
    print(f"Analysis complete! Results saved to {output_dir}")
    
    return output_dir


def generate_combined_report(
    image_path, image, class_name, prob, 
    feature_imp_dir, lime_dir, shap_dir, pdp_dir,
    output_dir):
    """
    Generate a combined HTML report with all explainability results.
    
    Args:
        image_path: Path to image file
        image: Image data
        class_name: Predicted class name
        prob: Prediction probability
        feature_imp_dir: Feature importance directory
        lime_dir: LIME directory
        shap_dir: SHAP directory
        pdp_dir: PDP directory
        output_dir: Output directory
    """
    # Get display name for the class
    class_display = CLASS_NAMES.get(class_name, class_name)
    
    # Create HTML report
    with open(output_dir / "combined_report.html", 'w') as f:
        f.write("<html><head>")
        f.write("<title>Skin Lesion Explainability Report</title>")
        f.write("<style>")
        f.write("body {font-family: Arial, sans-serif; margin: 40px; line-height: 1.6;}")
        f.write("h1, h2, h3 {color: #2c3e50;}")
        f.write(".container {display: flex; flex-wrap: wrap; justify-content: center;}")
        f.write(".section {margin: 20px; max-width: 100%;}")
        f.write(".prediction {background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px;}")
        f.write(".prediction-info {display: flex; align-items: center;}")
        f.write(".prediction-text {margin-left: 20px;}")
        f.write(".explainer {background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px;}")
        f.write(".explainer img {max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px;}")
        f.write("</style>")
        f.write("</head><body>")
        
        # Title
        f.write(f"<h1>Explainability Report for Skin Lesion Classification</h1>")
        
        # Prediction section
        f.write("<div class='prediction'>")
        f.write("<h2>Image & Prediction</h2>")
        f.write("<div class='prediction-info'>")
        
        # Save and embed image
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(output_dir / "input_image.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        f.write(f"<img src='input_image.png' width='300'>")
        f.write("<div class='prediction-text'>")
        f.write(f"<h3>Prediction: {class_display}</h3>")
        f.write(f"<p>Confidence: {prob:.2f}</p>")
        f.write(f"<p>File: {image_path.name}</p>")
        f.write("</div></div></div>")
        
        # Explainability sections
        f.write("<h2>Explainability Results</h2>")
        
        # Feature Importance
        f.write("<div class='explainer'>")
        f.write("<h3>1. Feature Importance Analysis</h3>")
        f.write("<p>This shows which features are most important for the model's prediction, based on the model's internal feature importance scores.</p>")
        f.write(f"<img src='feature_importance/feature_importance_plot.png' width='800'>")
        f.write("</div>")
        
        # LIME
        f.write("<div class='explainer'>")
        f.write("<h3>2. LIME (Local Interpretable Model-agnostic Explanations)</h3>")
        f.write("<p>LIME explains individual predictions by approximating the model locally with a simpler, interpretable model.</p>")
        f.write(f"<img src='lime/lime_explanation.png' width='800'>")
        f.write("</div>")
        
        # SHAP
        f.write("<div class='explainer'>")
        f.write("<h3>3. SHAP (SHapley Additive exPlanations)</h3>")
        f.write("<p>SHAP values show the contribution of each feature to the prediction using game theory principles.</p>")
        f.write(f"<img src='shap/shap_explanation.png' width='800'>")
        f.write("<div class='container'>")
        
        # Add additional SHAP visualizations if they exist
        for img_name in ["shap_waterfall.png", "shap_decision.png"]:
            if (shap_dir / img_name).exists():
                f.write(f"<div class='section'><img src='shap/{img_name}' width='400'></div>")
        
        f.write("</div></div>")
        
        # PDP
        f.write("<div class='explainer'>")
        f.write("<h3>4. Partial Dependence Plots (PDP)</h3>")
        f.write("<p>PDPs show how the prediction changes as feature values change, revealing the relationship between features and the model's output.</p>")
        f.write(f"<img src='pdp/pdp_explanation_{image_path.stem}.png' width='800'>")
        f.write("</div>")
        
        # Conclusion
        f.write("<div class='section'>")
        f.write("<h2>Conclusion</h2>")
        f.write("<p>The techniques above provide multiple perspectives on how the model made its prediction. ")
        f.write("Feature importance shows which features are globally important for the model, ")
        f.write("while LIME and SHAP provide local explanations for this specific image. ")
        f.write("Partial dependence plots reveal how features influence the model's predictions more generally.</p>")
        f.write("</div>")
        
        f.write("</body></html>")
    
    print(f"Combined report generated: {output_dir / 'combined_report.html'}")


def main():
    """Main function to run all explainers on a given image"""
    parser = argparse.ArgumentParser(description='Run all explainability techniques on a skin lesion image')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the image file')
    parser.add_argument('--model', type=str, default='RandomForest',
                      help='Model type (DecisionTree or RandomForest)')
    parser.add_argument('--features', type=str, default=None,
                      help='Path to features pickle file for background data')
    parser.add_argument('--output', type=str, default=None,
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    try:
        output_dir = analyze_image(
            args.image, args.model, args.features, args.output
        )
        print(f"Please check {output_dir / 'combined_report.html'} for complete results.")
    except Exception as e:
        print(f"Error analyzing image: {e}")
        raise


if __name__ == "__main__":
    main()
