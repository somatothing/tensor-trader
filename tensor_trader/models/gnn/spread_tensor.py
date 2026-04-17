"""Spread-Tensor model for market state representation."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle

logger = logging.getLogger(__name__)


class SpreadTensorModel:
    """
    Spread-Tensor model for market state representation.
    
    This model creates a high-dimensional tensor representation of market state
    using spreads between different timeframes and features, enabling
    cross-timeframe analysis without requiring PyTorch.
    """
    
    def __init__(self, 
                 input_dim: int = 50,
                 tensor_rank: int = 3,
                 n_components: int = 16,
                 random_state: int = 42):
        """
        Initialize Spread Tensor model.
        
        Args:
            input_dim: Number of input features
            tensor_rank: Rank of the tensor decomposition
            n_components: Number of tensor components
            random_state: Random seed
        """
        self.input_dim = input_dim
        self.tensor_rank = tensor_rank
        self.n_components = n_components
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Tensor decomposition factors
        self.factors: List[np.ndarray] = []
        self.weights: np.ndarray = np.ones(n_components) / n_components
        
        # Classification head
        self.classifier_weights: Optional[np.ndarray] = None
        self.classifier_bias: Optional[np.ndarray] = None
        
        self.is_trained = False
        self.classes_: Optional[np.ndarray] = None
        
    def _create_tensor_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Create tensor representation from input features.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Tensor representation (n_samples, n_components)
        """
        n_samples = X.shape[0]
        actual_input_dim = X.shape[1]
        
        # Initialize factors if not done or if dimensions changed
        if not self.factors or self.factors[0].shape[0] != actual_input_dim:
            self.factors = []
            for r in range(self.tensor_rank):
                factor = np.random.randn(actual_input_dim, self.n_components) * 0.01
                self.factors.append(factor)
        
        # Compute tensor decomposition
        # Simplified: use factor projections
        tensor_features = np.zeros((n_samples, self.n_components))
        
        for i in range(self.n_components):
            component = np.ones(n_samples)
            for factor in self.factors:
                # Project features onto factor
                projection = X @ factor[:, i]
                component *= np.tanh(projection)  # Non-linearity
            tensor_features[:, i] = component * self.weights[i]
        
        return tensor_features
    
    def _compute_spreads(self, X: np.ndarray) -> np.ndarray:
        """
        Compute spread features from input.
        
        Args:
            X: Input features
        
        Returns:
            Spread features
        """
        # Compute pairwise spreads (differences)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Select subset of features for spread computation
        n_spread_features = min(n_features, 20)
        X_subset = X[:, :n_spread_features]
        
        # Compute spreads
        spreads = []
        for i in range(n_spread_features):
            for j in range(i+1, min(i+5, n_spread_features)):
                spread = X_subset[:, i] - X_subset[:, j]
                spreads.append(spread)
        
        if spreads:
            spread_matrix = np.column_stack(spreads)
            # Combine original features with spreads
            return np.hstack([X, spread_matrix])
        
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SpreadTensorModel':
        """
        Train the Spread Tensor model.
        
        Args:
            X: Feature matrix
            y: Target labels
        
        Returns:
            self
        """
        logger.info(f"Training Spread Tensor on {X.shape[0]} samples")
        
        # Compute spread features
        X_spread = self._compute_spreads(X)
        
        # Create tensor representation
        tensor_repr = self._create_tensor_representation(X_spread)
        
        # Simple linear classifier on tensor representation
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize classifier
        self.classifier_weights = np.random.randn(self.n_components, n_classes) * 0.01
        self.classifier_bias = np.zeros(n_classes)
        
        # Simple gradient descent training
        learning_rate = 0.01
        n_epochs = 100
        
        # Convert labels to indices
        y_indices = np.array([np.where(self.classes_ == label)[0][0] for label in y])
        
        for epoch in range(n_epochs):
            # Forward pass
            logits = tensor_repr @ self.classifier_weights + self.classifier_bias
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Cross-entropy loss gradient
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(y_indices)), y_indices] = 1
            
            grad_logits = (probs - one_hot) / len(y_indices)
            grad_weights = tensor_repr.T @ grad_logits
            grad_bias = np.sum(grad_logits, axis=0)
            
            # Update weights
            self.classifier_weights -= learning_rate * grad_weights
            self.classifier_bias -= learning_rate * grad_bias
        
        self.is_trained = True
        logger.info("Spread Tensor training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Compute spread features
        X_spread = self._compute_spreads(X)
        
        # Create tensor representation
        tensor_repr = self._create_tensor_representation(X_spread)
        
        # Forward pass
        logits = tensor_repr @ self.classifier_weights + self.classifier_bias
        predictions_idx = np.argmax(logits, axis=1)
        
        # Map back to original labels
        return self.classes_[predictions_idx]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Probabilities for each class
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Compute spread features
        X_spread = self._compute_spreads(X)
        
        # Create tensor representation
        tensor_repr = self._create_tensor_representation(X_spread)
        
        # Forward pass
        logits = tensor_repr @ self.classifier_weights + self.classifier_bias
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'factors': self.factors,
                'weights': self.weights,
                'classifier_weights': self.classifier_weights,
                'classifier_bias': self.classifier_bias,
                'is_trained': self.is_trained,
                'classes_': self.classes_,
                'input_dim': self.input_dim,
                'tensor_rank': self.tensor_rank,
                'n_components': self.n_components
            }, f)
    
    def load(self, filepath: str) -> 'SpreadTensorModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.factors = data['factors']
            self.weights = data['weights']
            self.classifier_weights = data['classifier_weights']
            self.classifier_bias = data['classifier_bias']
            self.is_trained = data['is_trained']
            self.classes_ = data['classes_']
            self.input_dim = data['input_dim']
            self.tensor_rank = data['tensor_rank']
            self.n_components = data['n_components']
        return self
