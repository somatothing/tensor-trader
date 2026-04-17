"""Decision Tree model for market prediction."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class MarketDecisionTree:
    """Decision Tree classifier for market direction prediction."""
    
    def __init__(self, 
                 max_depth: int = 10,
                 min_samples_split: int = 10,
                 min_samples_leaf: int = 5,
                 random_state: int = 42):
        """
        Initialize Decision Tree model.
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf node
            random_state: Random seed
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_names: List[str] = []
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}
        
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced'
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'MarketDecisionTree':
        """
        Train the Decision Tree model.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional list of feature names
        
        Returns:
            self
        """
        if feature_names is not None:
            self.feature_names = feature_names
        
        logger.info(f"Training Decision Tree on {X.shape[0]} samples with {X.shape[1]} features")
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Feature importance
        if len(self.feature_names) == len(self.model.feature_importances_):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            self.metrics['feature_importance'] = dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20])
        
        logger.info("Decision Tree training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        self.metrics.update({
            'accuracy': accuracy_score(y, predictions),
            'classification_report': classification_report(y, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y, predictions).tolist(),
        })
        
        return self.metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        if not self.feature_names:
            return {f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save(self, filepath: str):
        """Save model to disk."""
        data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'metrics': self.metrics,
            'params': {
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'random_state': self.random_state,
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'MarketDecisionTree':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']
        self.metrics = data.get('metrics', {})
        
        params = data.get('params', {})
        self.max_depth = params.get('max_depth', 10)
        self.min_samples_split = params.get('min_samples_split', 10)
        self.min_samples_leaf = params.get('min_samples_leaf', 5)
        self.random_state = params.get('random_state', 42)
        
        logger.info(f"Model loaded from {filepath}")
        return self


if __name__ == '__main__':
    # Test with dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.choice([-1, 0, 1], size=1000)
    
    model = MarketDecisionTree(max_depth=5)
    model.fit(X, y, feature_names=[f'feature_{i}' for i in range(20)])
    
    # Evaluate
    metrics = model.evaluate(X, y)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Test prediction
    pred = model.predict(X[:5])
    print(f"Sample predictions: {pred}")
