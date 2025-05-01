from abc import ABC, abstractmethod
import numpy as np
import pickle
import os
from pathlib import Path


class BaseMLModel(ABC):
    """Base class for all machine learning models"""
    
    @staticmethod
    @abstractmethod
    def name():
        """Return the name of the model"""
        pass
    
    @abstractmethod
    def fit(self, X, y):
        """Train the model on given features and labels"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predict class labels for X"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Predict class probabilities for X"""
        pass
    
    def save(self, path):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
    @classmethod
    def load(cls, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)
