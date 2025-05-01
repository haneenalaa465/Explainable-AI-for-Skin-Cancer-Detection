from sklearn.ensemble import RandomForestClassifier
from XAI.modeling.models.ML_Base_model import BaseMLModel

class RFModel(BaseMLModel):
    """Random Forest model for skin lesion classification"""
    
    def __init__(self, n_estimators=100, random_state=0):
        """
        Initialize the Random Forest model
        
        Args:
            n_estimators: Number of trees in the forest
            random_state: Random state for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
    
    @staticmethod
    def name():
        return "RandomForest"
    
    def fit(self, X, y):
        """Train the model on given features and labels"""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Predict class labels for X"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for X"""
        return self.model.predict_proba(X)
    
    def feature_importances(self):
        """Return feature importances"""
        return self.model.feature_importances_
