"""XGBoost Gradient Boosting model for market prediction."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import xgboost, provide fallback if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, using sklearn GradientBoostingClassifier as fallback")
    from sklearn.ensemble import GradientBoostingClassifier


class MarketXGBoost:
    """XGBoost classifier for market direction prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, 
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42,
                 use_gpu: bool = False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.feature_names: List[str] = []
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}
        self.label_map: Dict[int, int] = {}  # Maps original labels to 0,1,2,...
        self.inverse_label_map: Dict[int, int] = {}
        
        if XGBOOST_AVAILABLE:
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'random_state': random_state,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'verbosity': 0
            }
            if use_gpu:
                params['tree_method'] = 'gpu_hist'
            self.model = xgb.XGBClassifier(**params)
        else:
            # Fallback to sklearn
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            feature_names: Optional[List[str]] = None,
            early_stopping_rounds: int = 10) -> 'MarketXGBoost':
        """Train the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target labels
            eval_set: Optional validation set for early stopping
            feature_names: Optional list of feature names
            early_stopping_rounds: Number of rounds for early stopping
        
        Returns:
            self
        """
        if feature_names is not None:
            self.feature_names = feature_names
        
        logger.info(f"Training XGBoost on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Map labels to contiguous integers starting from 0
        unique_labels = np.unique(y)
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        self.inverse_label_map = {new: old for old, new in self.label_map.items()}
        y_mapped = np.array([self.label_map[label] for label in y])
        
        if XGBOOST_AVAILABLE:
            fit_params = {}
            if eval_set is not None:
                X_val, y_val = eval_set
                y_val_mapped = np.array([self.label_map.get(label, label) for label in y_val])
                fit_params['eval_set'] = [(X_val, y_val_mapped)]
                fit_params['early_stopping_rounds'] = early_stopping_rounds
                fit_params['verbose'] = False
            
            self.model.fit(X, y_mapped, **fit_params)
            
            # Store best iteration
            if hasattr(self.model, 'best_iteration'):
                self.metrics['best_iteration'] = self.model.best_iteration
        else:
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
        
        logger.info("XGBoost training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        predictions = self.model.predict(X)
        # Inverse map predictions back to original labels
        if self.inverse_label_map:
            predictions = np.array([self.inverse_label_map.get(p, p) for p in predictions])
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
        
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        self.metrics.update({
            'accuracy': accuracy_score(y, predictions),
            'log_loss': log_loss(y, probabilities),
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
        if XGBOOST_AVAILABLE:
            self.model.save_model(filepath)
        else:
            import joblib
            joblib.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'metrics': self.metrics
            }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'MarketXGBoost':
        """Load model from disk."""
        if XGBOOST_AVAILABLE:
            self.model.load_model(filepath)
        else:
            import joblib
            data = joblib.load(filepath)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.is_trained = data['is_trained']
            self.metrics = data.get('metrics', {})
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
        return self


if __name__ == '__main__':
    # Test with dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.choice([-1, 0, 1], size=1000)
    
    model = MarketXGBoost(n_estimators=50, max_depth=4)
    model.fit(X, y, feature_names=[f'feature_{i}' for i in range(20)])
    
    # Evaluate
    metrics = model.evaluate(X, y)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Using XGBoost: {XGBOOST_AVAILABLE}")
    
    # Test prediction
    pred = model.predict(X[:5])
    print(f"Sample predictions: {pred}")
