import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

from XAI.modeling.models.DecisionTreeModel import DTModel
from XAI.modeling.models.RandomForestModel import RFModel
from XAI.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR


def train_ml_model(model_class, X_train, y_train, X_val, y_val, hyperparams=None, grid_search=False):
    """
    Train a machine learning model with optional hyperparameter tuning
    
    Args:
        model_class: The ML model class to use
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        hyperparams: Dictionary of hyperparameters to try
        grid_search: Whether to use grid search for hyperparameter tuning
        
    Returns:
        Trained model and best parameters if grid search is used
    """
    if grid_search and hyperparams:
        # Perform grid search
        print(f"Performing grid search for {model_class.name()}")
        
        # Create a base model instance for grid search
        if model_class == DTModel:
            base_model = DTModel().model
        elif model_class == RFModel:
            base_model = RFModel().model
        else:
            raise ValueError(f"Unknown model class: {model_class}")
        
        grid = GridSearchCV(
            base_model, 
            hyperparams, 
            cv=5, 
            scoring='accuracy', 
            verbose=1, 
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid.best_params_
        print(f"Best parameters: {best_params}")
        
        # Create a new model with best parameters
        if model_class == DTModel:
            model = DTModel(**best_params)
        elif model_class == RFModel:
            model = RFModel(**best_params)
        
    else:
        # Create model with default or provided parameters
        if hyperparams:
            model = model_class(**hyperparams)
        else:
            model = model_class()
    
    # Train the model
    print(f"Training {model.name()} model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    accuracy = np.mean(y_pred == y_val)
    print(f"Validation accuracy: {accuracy:.4f}")
    
    if grid_search and hyperparams:
        return model, best_params
    else:
        return model


def save_results(model, X_val, y_val, class_names, model_name):
    """Save evaluation results for the model"""
    # Create directories if they don't exist
    os.makedirs(REPORTS_DIR / "figures", exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Generate classification report
    report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORTS_DIR / "figures" / f"classification_report_{model_name}.csv")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / f"confusion_matrix_{model_name}.png")
    
    # Save feature importances if available
    if hasattr(model, 'feature_importances'):
        feature_importances = model.feature_importances()
        plt.figure(figsize=(12, 8))
        features_df = pd.DataFrame({
            'Feature': [f'Feature {i}' for i in range(len(feature_importances))],
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False).head(20)  # Top 20 features
        
        sns.barplot(x='Importance', y='Feature', data=features_df)
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "figures" / f"feature_importance_{model_name}.png")


def main(model_idx=-1):
    """Train machine learning models on extracted features"""
    # Load features
    features_path = PROCESSED_DATA_DIR / "ham10000_features.pkl"
    
    if not os.path.exists(features_path):
        print(f"Features file not found at {features_path}. Please run features.py first.")
        return
    
    print(f"Loading features from {features_path}")
    features_df = pd.read_pickle(features_path)
    
    # Convert features list to numpy array
    X = np.stack(features_df['features'].values)
    y = features_df['dx'].values
    
    # Get class names
    class_names = sorted(features_df['dx'].unique())
    print(f"Classes: {class_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for inference
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "ml_scaler.joblib")
    
    # Define models to train
    models_to_train = [
        {
            'class': DTModel,
            'hyperparams': {
                'max_depth': None,  # Default value
                'random_state': 0
            },
            'grid_search_params': {
                'max_depth': [None, 5, 10, 15, 20, 25],
                'criterion': ['gini', 'entropy']
            },
            'use_grid_search': True
        },
        {
            'class': RFModel,
            'hyperparams': {
                'n_estimators': 200,
                'random_state': 0
            },
            'grid_search_params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'use_grid_search': True
        }
    ]
    
    # Train models based on model_idx
    if model_idx == -1:
        # Train all models
        model_range = range(len(models_to_train))
    else:
        # Train specific model
        if model_idx < 0 or model_idx >= len(models_to_train):
            print(f"Error: model_idx {model_idx} is out of range. Available models: 0-{len(models_to_train)-1}")
            return
        model_range = range(model_idx, model_idx + 1)
    
    for idx in model_range:
        model_config = models_to_train[idx]
        model_class = model_config['class']
        model_name = model_class.name()
        
        print(f"\n=== Training {model_name} (index: {idx}) ===")
        
        # Train model with or without grid search
        if model_config.get('use_grid_search', False):
            model, best_params = train_ml_model(
                model_class,
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                model_config['grid_search_params'],
                True
            )
            
            # Update hyperparams with best parameters
            model_config['hyperparams'].update(best_params)
        else:
            model = train_ml_model(
                model_class,
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                model_config['hyperparams'],
                False
            )
        
        # Save model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = MODELS_DIR / f"{model_name}_{timestamp}.pkl"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save performance metrics
        save_results(model, X_test_scaled, y_test, class_names, model_name)
        
        # Save hyperparameters
        with open(MODELS_DIR / f"{model_name}_{timestamp}_params.txt", 'w') as f:
            for param, value in model_config['hyperparams'].items():
                f.write(f"{param}: {value}\n")


if __name__ == "__main__":
    main()
