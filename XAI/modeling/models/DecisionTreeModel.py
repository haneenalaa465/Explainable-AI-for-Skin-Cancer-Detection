from sklearn.tree import DecisionTreeClassifier
from XAI.modeling.models.ML_Base_model import BaseMLModel


class DTModel(BaseMLModel):
    """Decision Tree model for skin lesion classification"""
    
    def __init__(self, max_depth=None, random_state=0):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        
    @staticmethod
    def name():
        return "DecisionTree"
    
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
