"""Decision Tree Classifier for market prediction."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging

logger = logging.getLogger(__name__)


class MarketDecisionTree:
    """Decision Tree classifier for market direction prediction."""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 20,
                 min_samples_leaf: int = 10, random_state: int = 42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced'
        )
        self.feature_names: List[str] = []
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'MarketDecisionTree':
        """Train the decision tree model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            feature_names: Optional list of feature names
        
        Returns:
            self
        """
        if feature_names is not None:
            self.feature_names = feature_names
        
        logger.info(f"Training Decision Tree on {X.shape[0]} samples with {X.shape[1]} features")
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate feature importance
        if len(self.feature_names) == len(self.model.feature_importances_):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            self.metrics['feature_importance'] = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20])  # Top 20 features
        
        logger.info("Decision Tree training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        self.metrics.update({
            'accuracy': accuracy_score(y, predictions),
            'classification_report': classification_report(y, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y, predictions).tolist(),
        })
        
        return self.metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of folds
        
        Returns:
            Cross-validation scores
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        if not self.feature_names:
            return {f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'metrics': self.metrics
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'MarketDecisionTree':
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.is_trained = data['is_trained']
        self.metrics = data.get('metrics', {})
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
    print(f"Top 5 important features: {list(model.metrics['feature_importance'].keys())[:5]}")
    
    # Test prediction
    pred = model.predict(X[:5])
    print(f"Sample predictions: {pred}")
