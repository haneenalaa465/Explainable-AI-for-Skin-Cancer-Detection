"""
Feature extraction and engineering for skin lesion classification.
Based on the methods described in the paper: 
"Skin lesion classification of dermoscopic images using machine learning and convolutional neural network"
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.measure import moments_hu
import mahotas as mt

from XAI.config import (
    RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
    HAM10000_METADATA, CLASS_NAMES
)


def extract_color_histogram(image, bins=32):
    """
    Extract color histogram features for each channel.
    
    Args:
        image (numpy.ndarray): RGB image
        bins (int): Number of bins in histogram
        
    Returns:
        numpy.ndarray: Concatenated histogram features
    """
    histograms = []
    
    # Extract histogram for each channel (RGB)
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.extend(hist)
    
    return np.array(histograms)


def extract_shape_features(image):
    """
    Extract shape features (Hu Moments) from the image.
    
    Args:
        image (numpy.ndarray): RGB image
        
    Returns:
        numpy.ndarray: Shape features
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Threshold to get binary image (adjust threshold as needed)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate Hu Moments
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log transform to make feature values more manageable
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments


def extract_texture_features(image):
    """
    Extract Haralick texture features from the image.
    
    Args:
        image (numpy.ndarray): RGB image
        
    Returns:
        numpy.ndarray: Texture features
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate Haralick texture features
    textures = mt.features.haralick(gray)
    
    # Take the mean of features
    ht_mean = textures.mean(axis=0)
    
    return ht_mean


def extract_lbp_features(image, radius=8, n_points=24):
    """
    Extract Local Binary Pattern features for texture.
    
    Args:
        image (numpy.ndarray): RGB image
        radius (int): Radius of circle (spatial resolution)
        n_points (int): Number of points in a circularly symmetric neighbor set
        
    Returns:
        numpy.ndarray: LBP features
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Calculate histogram of LBP values
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist


def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extract Gray Level Co-occurrence Matrix (GLCM) features.
    
    Args:
        image (numpy.ndarray): RGB image
        distances (list): List of distances
        angles (list): List of angles
        
    Returns:
        numpy.ndarray: GLCM features
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Quantize the image to levels 0-7 (8 levels total)
    levels = 8
    max_value = 256
    gray = (gray / max_value * levels).astype(np.uint8)
    
    # Make sure all values are within range (0 to levels-1)
    gray = np.clip(gray, 0, levels-1)
    
    # Calculate GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, 
                       levels=levels, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    
    for prop in properties:
        feature = graycoprops(glcm, prop).flatten()
        features.extend(feature)
    
    return np.array(features)


def extract_all_features(image):
    """
    Extract all features (color, shape, texture) from an image.
    
    Args:
        image (numpy.ndarray): RGB image
        
    Returns:
        numpy.ndarray: Concatenated features
    """
    # Extract individual feature types
    color_features = extract_color_histogram(image)
    shape_features = extract_shape_features(image)
    texture_features = extract_texture_features(image)
    lbp_features = extract_lbp_features(image)
    glcm_features = extract_glcm_features(image)
    
    # Concatenate all features
    all_features = np.concatenate([
        color_features, 
        shape_features, 
        texture_features, 
        lbp_features, 
        glcm_features
    ])
    
    return all_features


def extract_features_from_dataset(dataset_dir, save_path=None):
    """
    Extract features from all images in the dataset.
    
    Args:
        dataset_dir (Path): Directory containing the organized dataset
        save_path (Path): Path to save the features DataFrame
        
    Returns:
        pandas.DataFrame: DataFrame with features for all images
    """
    features_data = []
    
    # Process each class directory
    for class_name in CLASS_NAMES.keys():
        class_dir = dataset_dir / class_name
        
        if not class_dir.exists():
            print(f"Warning: Directory for class {class_name} not found at {class_dir}")
            continue
        
        # Process each image in the class directory
        for img_path in tqdm(list(class_dir.glob("*.jpg")), desc=f"Processing {class_name}"):
            try:
                # Read image
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract features
                features = extract_all_features(image)
                
                # Add to data
                features_data.append({
                    'image_id': img_path.stem,
                    'dx': class_name,
                    'features': features
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_data)
    
    # Save features if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save features using pickle (more efficient for large arrays)
        features_df.to_pickle(save_path)
        print(f"Features saved to {save_path}")
    
    return features_df


def main():
    """Extract and save features from the HAM10000 dataset."""
    # Check if dataset is organized
    organized_dir = INTERIM_DATA_DIR / "organized_by_class"
    if not organized_dir.exists():
        print("Dataset not organized. Please run dataset.py first.")
        return
    
    # Extract features
    features_save_path = PROCESSED_DATA_DIR / "ham10000_features.pkl"
    extract_features_from_dataset(organized_dir, features_save_path)


if __name__ == "__main__":
    main()
