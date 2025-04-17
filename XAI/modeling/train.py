"""
Training script for the skin lesion classification model.
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from XAI.config import (
    MODELS_DIR, FIGURES_DIR, CLASS_NAMES, RANDOM_SEED,
    NUM_EPOCHS, LEARNING_RATE, LR_MIN
)
from XAI.dataset import prepare_data
from XAI.modeling.model import SkinLesionCNN


def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
               num_epochs=NUM_EPOCHS, model_save_path=None):
    """
    Train the model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run the model on
        num_epochs: Number of epochs to train for
        model_save_path: Path to save the best model
        
    Returns:
        tuple: Trained model and dictionary with training history
    """
    # Initialize tensorboard writer
    writer = SummaryWriter()
    
    # Initialize variables to track training progress
    best_val_loss = float('inf')
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Start training
    start_time = time.time()
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })
        
        # Calculate average training loss and accuracy
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        # Adjust learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if model_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, model_save_path)
                print(f"Model saved to {model_save_path}")
    
    # Calculate training time
    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_val_acc:.4f}")
    
    # Close tensorboard writer
    writer.close()
    
    # Load best model
    if model_save_path and os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on the provided data loader.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to run the model on
        
    Returns:
        tuple: Average loss and accuracy
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            
            # Calculate metrics
            loss += batch_loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    # Calculate average loss and accuracy
    loss = loss / total
    accuracy = correct / total
    
    return loss, accuracy


def test_model(model, test_loader, device, save_results=True):
    """
    Test the model and generate performance metrics.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run the model on
        save_results: Whether to save results to files
        
    Returns:
        dict: Dictionary with test results
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    class_report = classification_report(
        all_labels, all_preds, 
        target_names=list(CLASS_NAMES.values()),
        output_dict=True
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(CLASS_NAMES.values())))
    
    # Save results if requested
    if save_results:
        # Save classification report
        report_df = pd.DataFrame(class_report).transpose()
        report_path = FIGURES_DIR / "classification_report.csv"
        report_df.to_csv(report_path)
        print(f"Classification report saved to {report_path}")
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(CLASS_NAMES.values()),
            yticklabels=list(CLASS_NAMES.values())
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        conf_matrix_path = FIGURES_DIR / "confusion_matrix.png"
        plt.savefig(conf_matrix_path)
        plt.close()
        print(f"Confusion matrix saved to {conf_matrix_path}")
    
    # Return results
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'true_labels': all_labels
    }


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to train the model."""
    # Set random seed for reproducibility
    set_seed()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data()
    
    # Initialize model
    model = SkinLesionCNN().to(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=LR_MIN
    )
    
    # Create directory for saving model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = MODELS_DIR / "skin_lesion_cnn_best.pth"
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        num_epochs=NUM_EPOCHS, model_save_path=model_save_path
    )
    
    # Plot training history
    history_plot_path = FIGURES_DIR / "training_history.png"
    plot_training_history(history, save_path=history_plot_path)
    
    # Test model
    test_results = test_model(model, test_loader, device, save_results=True)
    
    return model, history, test_results


if __name__ == "__main__":
    main()