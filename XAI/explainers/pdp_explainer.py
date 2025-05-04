"""
Partial Dependence Plots (PDP) implementation for skin lesion classifiers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.inspection import partial_dependence, plot_partial_dependence
from tqdm import tqdm

from XAI.features import extract_all_features
from XAI.modeling.predict_ml import load_latest_model, load_feature_scaler
from XAI.config import REPORTS_DIR, CLASS_NAMES
from XAI.explainability.feature_importance import get_feature_names


def load_feature_dataset(feature_path=None, sample_size=None):
    """
    Load a dataset of features for PDP analysis.
    
    Args:
        feature_path: Path to features pickle file
        sample_size: Number of samples to use (for large datasets)
        
    Returns:
        numpy.ndarray: Feature array
    """
    if feature_path is None:
        # Use default path
        from XAI.config import PROCESSED_DATA_DIR
        feature_path = PROCESSED_DATA_DIR / "ham10000_features.pkl"
    
    # Load features
    print(f"Loading features from {feature_path}")
    features_df = pd.read_pickle(feature_path)
    
    # Extract feature arrays
    features_list = features_df['features'].tolist()
    features_array = np.stack(features_list)
    
    # Extract labels
    labels = features_df['dx'].values
    
    # Sample if needed
    if sample_size is not None and sample_size < len(features_array):
        # Stratified sampling to maintain class distribution
        from sklearn.model_selection import train_test_split
        _, features_array, _, labels = train_test_split(
            features_array, labels, 
            test_size=sample_size/len(features_array),
            stratify=labels, 
            random_state=42
        )
    
    return features_array, labels


def compute_partial_dependence(model, X, features, feature_names=None, 
                             num_points=50, class_idx=None):
    """
    Compute partial dependence for specified features.
    
    Args:
        model: Trained model
        X: Feature values (background dataset)
        features: List of feature indices to analyze
        feature_names: List of feature names
        num_points: Number of grid points for each feature
        class_idx: Index of class to analyze (None for all classes)
        
    Returns:
        Tuple of (pd_results, pd_values, pd_axes)
    """
    # Unwrap sklearn model from the BaseMLModel wrapper
    sklearn_model = model.model
    
    # Compute partial dependence
    result = partial_dependence(
        sklearn_model, X, features=features, 
        kind='average', grid_resolution=num_points,
        method='brute'
    )
    
    return result


def plot_pdp_for_features(model, X, feature_indices, feature_names, 
                        class_idx=None, save_dir=None):
    """
    Create partial dependence plots for specified features.
    
    Args:
        model: Trained model
        X: Feature dataset
        feature_indices: List of feature indices to analyze
        feature_names: List of feature names
        class_idx: Index of class to analyze (None for all classes)
        save_dir: Directory to save plots
    """
    # Create save directory if provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up colors for multiple classes
    if class_idx is None and len(CLASS_NAMES) > 1:
        colors = plt.cm.tab10.colors[:len(CLASS_NAMES)]
    else:
        colors = None
    
    # Compute and plot partial dependence for each feature
    for feature_idx in tqdm(feature_indices, desc="Computing PDPs"):
        feature_name = feature_names[feature_idx]
        
        # Compute partial dependence
        pd_result = compute_partial_dependence(
            model, X, [feature_idx], feature_names, 
            class_idx=class_idx
        )
        
        # Extract values
        pd_values = pd_result['average']
        pd_axes = pd_result['values']
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot PD for each class or the specified class
        if class_idx is None and len(CLASS_NAMES) > 1:
            for i, class_name in enumerate(CLASS_NAMES.keys()):
                plt.plot(pd_axes[0], pd_values[i].T, label=CLASS_NAMES[class_name], 
                       color=colors[i], linewidth=2)
            plt.legend()
        else:
            # Plot for single class
            target_idx = class_idx if class_idx is not None else 0
            plt.plot(pd_axes[0], pd_values[target_idx].T, 
                   color='blue', linewidth=2)
            
            # Add class name to title if specified
            class_title = ""
            if class_idx is not None:
                class_name = list(CLASS_NAMES.keys())[class_idx]
                class_title = f" for {CLASS_NAMES[class_name]}"
        
        # Add rug plot at the bottom
        feature_values = X[:, feature_idx]
        plt.plot(feature_values, np.zeros_like(feature_values) - 0.05, '|', ms=5, 
               color='black', alpha=0.2)
        
        # Customize plot
        plt.xlabel(feature_name)
        plt.ylabel('Partial Dependence')
        plt.title(f'Partial Dependence Plot for {feature_name}{class_title if class_idx is not None else ""}')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot if directory provided
        if save_dir:
            safe_name = feature_name.replace(' ', '_').replace('/', '_')
            class_suffix = f"_class{class_idx}" if class_idx is not None else ""
            plt.savefig(save_dir / f"pdp_{safe_name}{class_suffix}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()


def plot_2d_pdp(model, X, feature_idx1, feature_idx2, feature_names, 
              class_idx=0, save_path=None):
    """
    Create 2D partial dependence plot for a pair of features.
    
    Args:
        model: Trained model
        X: Feature dataset
        feature_idx1: Index of first feature
        feature_idx2: Index of second feature
        feature_names: List of feature names
        class_idx: Index of class to analyze
        save_path: Path to save plot
    """
    # Unwrap sklearn model from the BaseMLModel wrapper
    sklearn_model = model.model
    
    # Compute 2D partial dependence
    pd_result = partial_dependence(
        sklearn_model, X, features=[(feature_idx1, feature_idx2)], 
        kind='average', grid_resolution=20,
        method='brute'
    )
    
    # Extract values
    pd_values = pd_result['average']
    pd_axes = pd_result['values']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get feature names
    feat_name1 = feature_names[feature_idx1]
    feat_name2 = feature_names[feature_idx2]
    
    # Create 2D contour plot
    XX, YY = np.meshgrid(pd_axes[0][0], pd_axes[0][1])
    Z = pd_values[class_idx].T
    
    # Plot contour
    CS = ax.contourf(XX, YY, Z, cmap='viridis', alpha=0.7)
    
    # Add contour lines
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('Partial Dependence')
    
    # Add scatter plot of actual feature values
    if X.shape[0] <= 1000:
        # For small datasets, plot all points
        ax.scatter(X[:, feature_idx1], X[:, feature_idx2], 
                  s=20, alpha=0.3, color='white', edgecolor='black')
    else:
        # For larger datasets, sample points to avoid overcrowding
        sample_idx = np.random.choice(X.shape[0], 1000, replace=False)
        ax.scatter(X[sample_idx, feature_idx1], X[sample_idx, feature_idx2], 
                  s=20, alpha=0.3, color='white', edgecolor='black')
    
    # Get class name
    class_name = list(CLASS_NAMES.keys())[class_idx]
    
    # Customize plot
    ax.set_xlabel(feat_name1)
    ax.set_ylabel(feat_name2)
    ax.set_title(f'2D Partial Dependence Plot for {CLASS_NAMES[class_name]}')
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D PDP saved to {save_path}")
    
    plt.show()


def identify_important_feature_pairs(model, X, feature_names, n_top_features=10):
    """
    Identify pairs of important features for 2D partial dependence plots.
    
    Args:
        model: Trained model
        X: Feature dataset
        feature_names: List of feature names
        n_top_features: Number of top features to consider
        
    Returns:
        List of tuples with feature index pairs
    """
    # Extract feature importances
    if not hasattr(model, 'feature_importances'):
        raise ValueError("Model does not have feature_importances method")
    
    importances = model.feature_importances()
    
    # Get top features
    top_indices = np.argsort(importances)[-n_top_features:]
    
    # Generate pairs of top features
    feature_pairs = []
    for i in range(len(top_indices)):
        for j in range(i+1, len(top_indices)):
            feature_pairs.append((top_indices[i], top_indices[j]))
    
    return feature_pairs


def create_pdp_report(model, X, feature_names, output_dir=None, 
                    n_top_features=10, class_idx=None):
    """
    Create comprehensive PDP report with top features and feature pairs.
    
    Args:
        model: Trained model
        X: Feature dataset
        feature_names: List of feature names
        output_dir: Directory to save report
        n_top_features: Number of top features to analyze
        class_idx: Index of class to analyze (None for all classes)
    """
    if output_dir is None:
        output_dir = REPORTS_DIR / "explainability" / "pdp"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature importances
    importances = model.feature_importances()
    
    # Get top features
    top_indices = np.argsort(importances)[-n_top_features:]
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # Create directory for individual PDP plots
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(exist_ok=True)
    
    # Plot individual PDPs for top features
    plot_pdp_for_features(
        model, X, top_indices, feature_names, 
        class_idx=class_idx, save_dir=individual_dir
    )
    
    # Create directory for 2D PDP plots
    interaction_dir = output_dir / "interactions"
    interaction_dir.mkdir(exist_ok=True)
    
    # Identify important feature pairs
    feature_pairs = identify_important_feature_pairs(
        model, X, feature_names, n_top_features=5
    )
    
    # Create 2D PDPs for feature pairs
    for class_i in range(len(CLASS_NAMES)):
        # Skip other classes if specific class requested
        if class_idx is not None and class_i != class_idx:
            continue
        
        class_name = list(CLASS_NAMES.keys())[class_i]
        
        for feat_idx1, feat_idx2 in feature_pairs:
            # Get feature names
            feat_name1 = feature_names[feat_idx1]
            feat_name2 = feature_names[feat_idx2]
            
            # Create safe filenames
            safe_name1 = feat_name1.replace(' ', '_').replace('/', '_')
            safe_name2 = feat_name2.replace(' ', '_').replace('/', '_')
            
            # Create save path
            save_path = interaction_dir / f"2d_pdp_{safe_name1}_{safe_name2}_class{class_i}.png"
            
            # Plot 2D PDP
            try:
                plot_2d_pdp(
                    model, X, feat_idx1, feat_idx2, feature_names, 
                    class_idx=class_i, save_path=save_path
                )
            except Exception as e:
                print(f"Error creating 2D PDP for features {feat_name1} and {feat_name2}: {e}")
    
    # Create summary report with feature importances and PDPs
    plt.figure(figsize=(12, 8))
    
    # Create bar chart of top feature importances
    y_pos = np.arange(len(top_indices))
    plt.barh(y_pos, importances[top_indices], align='center')
    plt.yticks(y_pos, top_feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances')
    
    # Save summary
    plt.savefig(output_dir / "feature_importance_summary.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create HTML report (optional)
    with open(output_dir / "pdp_report.html", 'w') as f:
        f.write("<html><head>")
        f.write("<title>Partial Dependence Plots Report</title>")
        f.write("<style>body {font-family: Arial; margin: 40px;} ")
        f.write("h1 {color: #333;} .gallery {display: flex; flex-wrap: wrap;} ")
        f.write(".plot {margin: 10px; text-align: center;}")
        f.write("</style></head><body>")
        f.write("<h1>Partial Dependence Plots Report</h1>")
        
        # Feature importance summary
        f.write("<h2>Feature Importance Summary</h2>")
        f.write(f"<img src='feature_importance_summary.png' width='800'>")
        
        # Individual PDPs
        f.write("<h2>Individual Feature PDPs</h2>")
        f.write("<div class='gallery'>")
        
        for idx in top_indices:
            feature_name = feature_names[idx]
            safe_name = feature_name.replace(' ', '_').replace('/', '_')
            
            class_suffix = f"_class{class_idx}" if class_idx is not None else ""
            img_path = f"individual/pdp_{safe_name}{class_suffix}.png"
            
            f.write(f"<div class='plot'>")
            f.write(f"<img src='{img_path}' width='400'>")
            f.write(f"<p>{feature_name}</p>")
            f.write("</div>")
        
        f.write("</div>")
        
        # Feature interaction PDPs
        f.write("<h2>Feature Interaction PDPs</h2>")
        f.write("<div class='gallery'>")
        
        for feat_idx1, feat_idx2 in feature_pairs:
            feat_name1 = feature_names[feat_idx1]
            feat_name2 = feature_names[feat_idx2]
            
            safe_name1 = feat_name1.replace(' ', '_').replace('/', '_')
            safe_name2 = feat_name2.replace(' ', '_').replace('/', '_')
            
            # For each class or the specified class
            for class_i in range(len(CLASS_NAMES)):
                if class_idx is not None and class_i != class_idx:
                    continue
                
                class_name = list(CLASS_NAMES.keys())[class_i]
                img_path = f"interactions/2d_pdp_{safe_name1}_{safe_name2}_class{class_i}.png"
                
                # Check if file exists
                if Path(output_dir / img_path).exists():
                    f.write(f"<div class='plot'>")
                    f.write(f"<img src='{img_path}' width='400'>")
                    f.write(f"<p>{feat_name1} × {feat_name2} ({CLASS_NAMES[class_name]})</p>")
                    f.write("</div>")
        
        f.write("</div>")
        f.write("</body></html>")
    
    print(f"PDP report created at {output_dir / 'pdp_report.html'}")


def explain_top_features_for_image(image_path, model, scaler, X_background, 
                                 feature_names=None, n_top_features=5, 
                                 save_dir=None):
    """
    Explain prediction for a single image using PDP for top features.
    
    Args:
        image_path: Path to image file
        model: Trained model
        scaler: Feature scaler
        X_background: Background dataset for PDP
        feature_names: List of feature names
        n_top_features: Number of top features to explain
        save_dir: Directory to save visualizations
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
    
    # Get feature importances
    importances = model.feature_importances()
    
    # Get actual feature values for the image
    feature_values = features_scaled[0]
    
    # Get top features
    top_indices = np.argsort(importances)[-n_top_features:]
    top_feature_names = [feature_names[i] for i in top_indices]
    top_feature_importances = importances[top_indices]
    top_feature_values = feature_values[top_indices]
    
    # Create save directory if provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Display the image
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax1.imshow(image)
    ax1.set_title(f'Prediction: {CLASS_NAMES.get(class_name, class_name)}\n'
                 f'Confidence: {probabilities.max():.2f}')
    ax1.axis('off')
    
    # 2. Display feature importance bar chart
    ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    y_pos = np.arange(len(top_indices))
    ax2.barh(y_pos, top_feature_importances, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_feature_names)
    ax2.set_xlabel('Feature Importance')
    ax2.set_title('Top Feature Importances')
    
    # 3. Display class probabilities
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    if isinstance(probabilities, np.ndarray) and len(probabilities) > 1:
        class_names = list(CLASS_NAMES.values())
        y_pos = np.arange(len(class_names))
        ax3.barh(y_pos, probabilities, align='center')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(class_names)
        ax3.set_xlabel('Probability')
        ax3.set_title('Class Probabilities')
    
    # 4. Display PDP for top feature with actual value marked
    # For top 2 features, show PD plots with actual value
    for i, ax_idx in enumerate([(1, 1), (1, 2)]):
        if i < len(top_indices):
            feature_idx = top_indices[i]
            feature_value = feature_values[feature_idx]
            
            # Compute PDP
            pd_result = compute_partial_dependence(
                model, X_background, [feature_idx], feature_names, 
                class_idx=class_idx if isinstance(class_idx, int) else None
            )
            
            # Extract values
            pd_values = pd_result['average']
            pd_axes = pd_result['values']
            
            # Create plot
            ax = plt.subplot2grid((2, 3), ax_idx)
            
            if isinstance(class_idx, int):
                # Plot for specific class
                ax.plot(pd_axes[0], pd_values[0].T, color='blue', linewidth=2)
            else:
                # Plot for all classes
                colors = plt.cm.tab10.colors[:len(CLASS_NAMES)]
                for j, c_name in enumerate(CLASS_NAMES.keys()):
                    ax.plot(pd_axes[0], pd_values[j].T, 
                           label=CLASS_NAMES[c_name], 
                           color=colors[j], linewidth=2)
                ax.legend()
            
            # Mark actual value
            val_idx = np.argmin(np.abs(pd_axes[0] - feature_value))
            if isinstance(class_idx, int):
                pd_y = pd_values[0][val_idx]
            else:
                # Use predicted class
                pd_y = pd_values[class_idx][val_idx]
            
            ax.plot(feature_value, pd_y, 'ro', ms=10, label='Actual value')
            
            # Customize plot
            ax.set_xlabel(feature_names[feature_idx])
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'PDP for {feature_names[feature_idx]}')
            ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save visualization if save_dir provided
    if save_dir:
        img_name = Path(image_path).stem
        plt.savefig(save_dir / f"pdp_explanation_{img_name}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"PDP explanation saved to {save_dir / f'pdp_explanation_{img_name}.png'}")
    
    plt.show()
    
    return class_name, probabilities, top_indices, top_feature_values


def main():
    """Demo PDP explanations for the trained model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDP Explanations')
    parser.add_argument('--model', type=str, default='RandomForest', 
                      help='Model type (DecisionTree or RandomForest)')
    parser.add_argument('--image', type=str, default=None,
                      help='Path to an image file for explanation')
    parser.add_argument('--features', type=str, default=None,
                      help='Path to features pickle file for background data')
    parser.add_argument('--n_top', type=int, default=10,
                      help='Number of top features to analyze')
    parser.add_argument('--class_idx', type=int, default=None,
                      help='Index of class to analyze (None for all classes)')
    parser.add_argument('--report', action='store_true',
                      help='Generate comprehensive PDP report')
    args = parser.parse_args()
    
    try:
        # Load the model
        model = load_latest_model(args.model)
        
        # Load the scaler
        scaler = load_feature_scaler()
        
        # Create output directory
        output_dir = REPORTS_DIR / "explainability" / "pdp"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get feature names
        feature_names = get_feature_names()
        
        # Load feature dataset
        X, y = load_feature_dataset(args.features, sample_size=500)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
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
            explain_top_features_for_image(
                image_path, model, scaler, X_scaled, 
                feature_names=feature_names,
                n_top_features=args.n_top,
                save_dir=img_output_dir
            )
        
        # Generate comprehensive PDP report
        elif args.report:
            create_pdp_report(
                model, X_scaled, feature_names, 
                output_dir=output_dir,
                n_top_features=args.n_top,
                class_idx=args.class_idx
            )
        
        # If no specific action provided, just show PDPs for top features
        else:
            # Get feature importances
            importances = model.feature_importances()
            
            # Get top features
            top_indices = np.argsort(importances)[-args.n_top:]
            
            # Plot PDPs for top features
            plot_pdp_for_features(
                model, X_scaled, top_indices, feature_names, 
                class_idx=args.class_idx, save_dir=output_dir
            )
        
    except Exception as e:
        print(f"Error generating PDP explanations: {e}")
        raise


if __name__ == "__main__":
    main()
