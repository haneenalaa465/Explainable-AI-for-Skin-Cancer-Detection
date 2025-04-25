"""
Visualization utilities for analyzing skin lesion images and model performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import itertools
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm

from XAI.config import (
    FIGURES_DIR, CLASS_NAMES, PROCESSED_DATA_DIR, RAW_DATA_DIR
)


def plot_class_distribution(metadata_path, save_path=None):
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        metadata_path: Path to metadata CSV file
        save_path: Path to save the plot
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Count classes
    class_counts = metadata['dx'].value_counts()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot class distribution
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution in HAM10000 Dataset')
    
    # Add value labels on top of bars
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + 50, str(count), ha='center')
    
    # Replace class codes with full names
    plt.xticks(range(len(CLASS_NAMES)), [CLASS_NAMES[cls] for cls in class_counts.index], rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()
    
    return class_counts

def plot_sample_images(metadata_path, image_dir, num_samples=5, save_path=None):
    """
    Plot sample images from each class.
    
    Args:
        metadata_path: Path to metadata CSV file
        image_dir: Directory containing images
        num_samples: Number of samples to plot per class
        save_path: Path to save the plot
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Get samples for each class
    samples_by_class = {}
    for class_name in CLASS_NAMES.keys():
        samples = metadata[metadata['dx'] == class_name].sample(min(num_samples, sum(metadata['dx'] == class_name)))
        samples_by_class[class_name] = samples
    
    # Create figure
    fig, axs = plt.subplots(len(CLASS_NAMES), num_samples, figsize=(num_samples * 3, len(CLASS_NAMES) * 3))
    
    # Plot samples
    for i, (class_name, samples) in enumerate(samples_by_class.items()):
        for j, (_, row) in enumerate(samples.iterrows()):
            img_id = row['image_id']
            
            # Find image file
            img_path = None
            for img_dir in [image_dir + '/HAM10000_images_part_1', image_dir + '/HAM10000_images_part_2']:
                temp_path = f"{img_dir}/{img_id}.jpg"
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            
            if img_path is None:
                print(f"Warning: Image {img_id} not found")
                continue
            
            # Load and display image
            img = Image.open(img_path)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            
            # Add class label to first image in row
            if j == 0:
                axs[i, j].set_title(f"{CLASS_NAMES[class_name]}", fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Sample images plot saved to {save_path}")
    
    plt.show()


def plot_image_characteristics(metadata_path, save_path=None):
    """
    Plot characteristics of images by class (age, sex, location).
    
    Args:
        metadata_path: Path to metadata CSV file
        save_path: Path to save the plot
    """
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot age distribution by class
    sns.boxplot(x='dx', y='age', data=metadata, ax=axs[0])
    axs[0].set_title('Age Distribution by Class')
    axs[0].set_xlabel('Class')
    axs[0].set_ylabel('Age')
    axs[0].set_xticklabels([CLASS_NAMES[cls] for cls in sorted(metadata['dx'].unique())], rotation=45, ha='right')
    
    # Plot sex distribution by class
    sex_counts = pd.crosstab(metadata['dx'], metadata['sex'])
    sex_counts_norm = sex_counts.div(sex_counts.sum(axis=1), axis=0)
    sex_counts_norm.plot(kind='bar', stacked=True, ax=axs[1])
    axs[1].set_title('Sex Distribution by Class')
    axs[1].set_xlabel('Class')
    axs[1].set_ylabel('Proportion')
    axs[1].set_xticklabels([CLASS_NAMES[cls] for cls in sex_counts.index], rotation=45, ha='right')
    axs[1].legend(title='Sex')
    
    # Plot localization distribution by class
    loc_counts = pd.crosstab(metadata['dx'], metadata['localization'])
    loc_counts_norm = loc_counts.div(loc_counts.sum(axis=1), axis=0)
    loc_counts_norm.plot(kind='bar', stacked=True, ax=axs[2])
    axs[2].set_title('Localization Distribution by Class')
    axs[2].set_xlabel('Class')
    axs[2].set_ylabel('Proportion')
    axs[2].set_xticklabels([CLASS_NAMES[cls] for cls in loc_counts.index], rotation=45, ha='right')
    axs[2].legend(title='Localization', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Image characteristics plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(conf_matrix, classes=None, normalize=False, title='Confusion Matrix', 
                          cmap=plt.cm.Blues, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Title of the plot
        cmap: Colormap
        save_path: Path to save the plot
    """
    if classes is None:
        classes = list(CLASS_NAMES.values())
    
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_roc_curve(true_labels, pred_probs, classes=None, save_path=None):
    """
    Plot ROC curve for each class.
    
    Args:
        true_labels: True labels (one-hot encoded)
        pred_probs: Predicted probabilities
        classes: List of class names
        save_path: Path to save the plot
    """
    if classes is None:
        classes = list(CLASS_NAMES.values())
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, class_name in enumerate(classes):
        # Convert true_labels to binary format for current class
        y_true_binary = (np.array(true_labels) == i).astype(int)
        y_score = np.array(pred_probs)[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve plot saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(true_labels, pred_probs, classes=None, save_path=None):
    """
    Plot precision-recall curve for each class.
    
    Args:
        true_labels: True labels (one-hot encoded)
        pred_probs: Predicted probabilities
        classes: List of class names
        save_path: Path to save the plot
    """
    if classes is None:
        classes = list(CLASS_NAMES.values())
    
    # Compute precision-recall curve for each class
    precision = {}
    recall = {}
    avg_precision = {}
    
    for i, class_name in enumerate(classes):
        # Convert true_labels to binary format for current class
        y_true_binary = (np.array(true_labels) == i).astype(int)
        y_score = np.array(pred_probs)[:, i]
        
        precision[i], recall[i], _ = precision_recall_curve(y_true_binary, y_score)
        avg_precision[i] = np.mean(precision[i])
    
    # Plot all precision-recall curves
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(classes):
        plt.plot(recall[i], precision[i], lw=2,
                 label=f'{class_name} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-recall curve plot saved to {save_path}")
    
    plt.show()


def plot_tsne_features(model, data_loader, device, perplexity=30, n_iter=1000, save_path=None):
    """
    Plot t-SNE visualization of features from the model's penultimate layer.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run the model on
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        save_path: Path to save the plot
    """
    # Set model to evaluation mode
    model.eval()
    
    # Extract features and labels
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Extracting features"):
            inputs = inputs.to(device)
            
            # Extract features from the penultimate layer
            batch_features = model.extract_features(inputs)
            
            # Add to lists
            features.append(batch_features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    
    # Concatenate batches
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    
    # Apply t-SNE
    print(f"Applying t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Plot t-SNE visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, 
                          cmap='tab10', alpha=0.7, s=10)
    
    # Add legend
    class_indices = np.unique(labels)
    class_names = [list(CLASS_NAMES.values())[i] for i in class_indices]
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=class_names,
                       loc="best", title="Classes")
    
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to generate plots for the HAM10000 dataset."""
    # Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Plot class distribution
    metadata_path = f"{RAW_DATA_DIR}/HAM10000_metadata.csv"
    
    if os.path.exists(metadata_path):
        # Plot class distribution
        plot_class_distribution(
            metadata_path, 
            save_path=f"{FIGURES_DIR}/class_distribution.png"
        )
        
        # Plot sample images
        plot_sample_images(
            metadata_path, 
            RAW_DATA_DIR,
            save_path=f"{FIGURES_DIR}/sample_images.png"
        )
        
        # Plot image characteristics
        plot_image_characteristics(
            metadata_path, 
            save_path=f"{FIGURES_DIR}/image_characteristics.png"
        )
    else:
        print(f"Metadata file not found at {metadata_path}")


if __name__ == "__main__":
    main()