"""Training pipeline for Tensor Trader models."""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import os
from datetime import datetime

from ..features.pipeline import FeaturePipeline, create_target_labels
from ..models.tree.decision_tree import MarketDecisionTree
from ..models.boosting.xgboost_model import MarketXGBoost
from ..models.gnn.market_gnn import MarketGNN
from .hyperparameters import DynamicHyperparameterMacro, EntropyBasedTuner

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self, model_type: str = 'xgboost', 
                 test_size: float = 0.2, val_size: float = 0.1,
                 random_state: int = 42):
        self.model_type = model_type
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.model = None
        self.feature_pipeline = FeaturePipeline()
        self.metrics: Dict[str, Any] = {}
        self.training_history: List[Dict] = []
        
    def prepare_data(self, df: pd.DataFrame, target_lookahead: int = 5,
                    target_threshold: float = 0.005) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training.
        
        Args:
            df: Raw OHLCV DataFrame
            target_lookahead: Periods to look ahead for target
            target_threshold: Minimum price change for labeling
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Preparing data...")
        
        # Generate features
        features = self.feature_pipeline.transform(df)
        
        # Create target labels
        features_with_target = create_target_labels(
            features, 
            lookahead=target_lookahead,
            threshold=target_threshold
        )
        
        # Drop NaN values but keep track of how many we lose
        n_before = len(features_with_target)
        features_with_target = features_with_target.dropna()
        n_after = len(features_with_target)
        logger.info(f"Dropped {n_before - n_after} rows with NaN values ({n_after} remaining)")
        
        if n_after < 50:
            raise ValueError(f"Insufficient data after dropping NaN: {n_after} samples. Need at least 50.")
        
        # Extract feature columns (exclude OHLCV and target columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'future_return', 'target', 'target_return', 'target_binary']
        feature_cols = [c for c in features_with_target.columns if c not in exclude_cols]
        
        X = features_with_target[feature_cols].values
        y = features_with_target['target'].values
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              hyperparam_tuning: bool = False, n_trials: int = 20) -> Any:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            hyperparam_tuning: Whether to perform hyperparameter tuning
            n_trials: Number of tuning trials
        
        Returns:
            Trained model
        """
        logger.info(f"Training {self.model_type} model...")
        
        if hyperparam_tuning and X_val is not None and y_val is not None:
            # Perform entropy-based hyperparameter tuning
            logger.info("Starting hyperparameter tuning...")
            
            macro = DynamicHyperparameterMacro(self.model_type)
            
            if self.model_type == 'xgboost':
                tuner = EntropyBasedTuner(MarketXGBoost, macro, n_trials=n_trials)
            elif self.model_type == 'decision_tree':
                tuner = EntropyBasedTuner(MarketDecisionTree, macro, n_trials=n_trials)
            else:
                raise ValueError(f"Hyperparameter tuning not supported for {self.model_type}")
            
            best_params = tuner.tune(X_train, y_train, X_val, y_val)
            self.training_history.extend(tuner.get_tuning_history())
            
            # Train final model with best params
            if self.model_type == 'xgboost':
                self.model = MarketXGBoost(**best_params)
            elif self.model_type == 'decision_tree':
                self.model = MarketDecisionTree(**best_params)
        else:
            # Use default hyperparameters
            if self.model_type == 'xgboost':
                self.model = MarketXGBoost(n_estimators=100, max_depth=6)
            elif self.model_type == 'decision_tree':
                self.model = MarketDecisionTree(max_depth=10)
            elif self.model_type == 'gnn':
                self.model = MarketGNN(input_dim=X_train.shape[1])
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train the model
        if self.model_type == 'gnn':
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, feature_names=self.feature_pipeline.get_feature_names())
        
        logger.info("Training completed")
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set."""
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_macro': f1_score(y_test, predictions, average='macro', zero_division=0),
            'precision_macro': precision_score(y_test, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, predictions, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        for label in np.unique(y_test):
            mask = y_test == label
            if mask.sum() > 0:
                self.metrics[f'precision_class_{label}'] = precision_score(
                    y_test == label, predictions == label, zero_division=0
                )
                self.metrics[f'recall_class_{label}'] = recall_score(
                    y_test == label, predictions == label, zero_division=0
                )
        
        logger.info(f"Test accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"F1 score: {self.metrics['f1_macro']:.4f}")
        
        return self.metrics
    
    def save(self, model_path: str, metadata_path: Optional[str] = None):
        """Save model and metadata."""
        if self.model is None:
            raise RuntimeError("Model must be trained before saving")
        
        # Save model
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'metrics': self.metrics,
            'feature_names': self.feature_pipeline.get_feature_names(),
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat(),
        }
        
        if metadata_path is None:
            metadata_path = model_path.replace('.pkl', '_metadata.json').replace('.json', '_metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, model_path: str, model_type: Optional[str] = None):
        """Load a trained model."""
        if model_type is None:
            model_type = self.model_type
        
        if model_type == 'xgboost':
            self.model = MarketXGBoost()
        elif model_type == 'decision_tree':
            self.model = MarketDecisionTree()
        elif model_type == 'gnn':
            self.model = MarketGNN(input_dim=1)  # Will be updated on load
        
        self.model.load(model_path)
        self.model_type = model_type
        
        logger.info(f"Model loaded from {model_path}")


def train_model(df: pd.DataFrame, model_type: str = 'xgboost',
                output_dir: str = './models/checkpoints',
                hyperparam_tuning: bool = False) -> TrainingPipeline:
    """Convenience function to train a model.
    
    Args:
        df: Raw OHLCV DataFrame
        model_type: Type of model to train
        output_dir: Directory to save model
        hyperparam_tuning: Whether to perform hyperparameter tuning
    
    Returns:
        Trained TrainingPipeline instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = TrainingPipeline(model_type=model_type)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_data(df)
    
    # Train
    pipeline.train(X_train, y_train, X_val, y_val, hyperparam_tuning=hyperparam_tuning)
    
    # Evaluate
    pipeline.evaluate(X_test, y_test)
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(output_dir, f'{model_type}_{timestamp}.pkl')
    pipeline.save(model_path)
    
    return pipeline


if __name__ == '__main__':
    # Test with dummy data
    print("Testing training pipeline with dummy data...")
    
    # Create dummy OHLCV data
    np.random.seed(42)
    n_samples = 500
    
    df = pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure high >= low
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1) + 1)
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1) - 1)
    
    print(f"Input data shape: {df.shape}")
    
    # Train
    pipeline = train_model(df, model_type='decision_tree', hyperparam_tuning=False)
    
    print(f"\nFinal metrics:")
    for key, value in pipeline.metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTraining pipeline: SUCCESS")
