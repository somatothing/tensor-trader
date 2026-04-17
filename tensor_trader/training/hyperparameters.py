"""Dynamic hyperparameter management and entropy-based tuning."""
import numpy as np
import logging
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Defines a hyperparameter search space."""
    name: str
    type: str  # 'int', 'float', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    
    def sample(self, entropy_factor: float = 1.0) -> Any:
        """Sample a value from the space with entropy-based exploration."""
        if self.type == 'int':
            if self.log_scale:
                log_low = np.log(self.low)
                log_high = np.log(self.high)
                sample = np.exp(np.random.uniform(log_low, log_high))
                return int(np.clip(sample, self.low, self.high))
            else:
                # Adjust range based on entropy factor
                range_size = (self.high - self.low) * entropy_factor
                center = (self.low + self.high) / 2
                adjusted_low = max(self.low, center - range_size / 2)
                adjusted_high = min(self.high, center + range_size / 2)
                return int(np.random.uniform(adjusted_low, adjusted_high))
        
        elif self.type == 'float':
            if self.log_scale:
                log_low = np.log(self.low)
                log_high = np.log(self.high)
                sample = np.exp(np.random.uniform(log_low, log_high))
                return float(np.clip(sample, self.low, self.high))
            else:
                range_size = (self.high - self.low) * entropy_factor
                center = (self.low + self.high) / 2
                adjusted_low = max(self.low, center - range_size / 2)
                adjusted_high = min(self.high, center + range_size / 2)
                return float(np.random.uniform(adjusted_low, adjusted_high))
        
        elif self.type == 'categorical':
            return np.random.choice(self.choices)
        
        else:
            raise ValueError(f"Unknown hyperparameter type: {self.type}")


class DynamicHyperparameterMacro:
    """Manages dynamic hyperparameters with entropy-based adaptation."""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.spaces: Dict[str, HyperparameterSpace] = {}
        self.history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -np.inf
        self.entropy: float = 1.0
        self.entropy_decay: float = 0.95
        self.min_entropy: float = 0.1
        
        self._init_spaces()
    
    def _init_spaces(self):
        """Initialize hyperparameter spaces based on model type."""
        if self.model_type == 'xgboost':
            self.spaces = {
                'n_estimators': HyperparameterSpace('n_estimators', 'int', 50, 500),
                'max_depth': HyperparameterSpace('max_depth', 'int', 3, 12),
                'learning_rate': HyperparameterSpace('learning_rate', 'float', 0.01, 0.3, log_scale=True),
                'subsample': HyperparameterSpace('subsample', 'float', 0.6, 1.0),
                'colsample_bytree': HyperparameterSpace('colsample_bytree', 'float', 0.6, 1.0),
                'min_child_weight': HyperparameterSpace('min_child_weight', 'int', 1, 10),
                'gamma': HyperparameterSpace('gamma', 'float', 0, 5),
                'reg_alpha': HyperparameterSpace('reg_alpha', 'float', 0, 1, log_scale=True),
                'reg_lambda': HyperparameterSpace('reg_lambda', 'float', 0, 1, log_scale=True),
            }
        elif self.model_type == 'decision_tree':
            self.spaces = {
                'max_depth': HyperparameterSpace('max_depth', 'int', 3, 20),
                'min_samples_split': HyperparameterSpace('min_samples_split', 'int', 2, 50),
                'min_samples_leaf': HyperparameterSpace('min_samples_leaf', 'int', 1, 20),
                'max_features': HyperparameterSpace('max_features', 'categorical', 
                                                    choices=['sqrt', 'log2', None]),
            }
        elif self.model_type == 'gnn':
            self.spaces = {
                'hidden_dim': HyperparameterSpace('hidden_dim', 'int', 32, 256),
                'num_layers': HyperparameterSpace('num_layers', 'int', 2, 5),
                'dropout': HyperparameterSpace('dropout', 'float', 0.1, 0.5),
                'learning_rate': HyperparameterSpace('learning_rate', 'float', 0.0001, 0.01, log_scale=True),
            }
    
    def sample_params(self) -> Dict[str, Any]:
        """Sample a set of hyperparameters."""
        params = {}
        for name, space in self.spaces.items():
            params[name] = space.sample(self.entropy)
        return params
    
    def update_entropy(self, score: float, params: Dict[str, Any]):
        """Update entropy based on performance."""
        self.history.append({'score': score, 'params': params})
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            # Reduce entropy as we find better solutions
            self.entropy = max(self.min_entropy, self.entropy * self.entropy_decay)
            logger.info(f"New best score: {score:.4f}, entropy reduced to {self.entropy:.4f}")
        else:
            # Slightly increase entropy to explore more
            self.entropy = min(1.0, self.entropy * 1.05)
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best hyperparameters found."""
        if self.best_params is None:
            return self.sample_params()
        return self.best_params
    
    def calculate_entropy(self, scores: List[float]) -> float:
        """Calculate entropy of score distribution."""
        if len(scores) < 2:
            return 1.0
        
        # Normalize scores to probabilities
        scores = np.array(scores)
        scores = scores - scores.min() + 1e-8  # Ensure positive
        probs = scores / scores.sum()
        
        # Calculate Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return entropy / np.log(len(scores))  # Normalize to [0, 1]


class EntropyBasedTuner:
    """Entropy-based hyperparameter tuner."""
    
    def __init__(self, model_class: Callable, hyperparam_macro: DynamicHyperparameterMacro,
                 n_trials: int = 50, n_jobs: int = -1):
        self.model_class = model_class
        self.hyperparam_macro = hyperparam_macro
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.results: List[Dict[str, Any]] = []
        
    def tune(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             metric_func: Optional[Callable] = None) -> Dict[str, Any]:
        """Run hyperparameter tuning with entropy-based exploration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            metric_func: Custom metric function (defaults to accuracy)
        
        Returns:
            Best hyperparameters found
        """
        from sklearn.metrics import accuracy_score
        
        if metric_func is None:
            metric_func = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
        
        logger.info(f"Starting entropy-based tuning with {self.n_trials} trials")
        
        for trial in range(self.n_trials):
            # Sample hyperparameters
            params = self.hyperparam_macro.sample_params()
            
            try:
                # Train model
                model = self.model_class(**params)
                model.fit(X_train, y_train)
                
                # Evaluate
                predictions = model.predict(X_val)
                score = metric_func(y_val, predictions)
                
                # Update entropy
                self.hyperparam_macro.update_entropy(score, params)
                
                # Store result
                result = {
                    'trial': trial,
                    'params': params,
                    'score': score,
                    'entropy': self.hyperparam_macro.entropy
                }
                self.results.append(result)
                
                if trial % 10 == 0:
                    logger.info(f"Trial {trial}/{self.n_trials}: Score={score:.4f}, "
                              f"Entropy={self.hyperparam_macro.entropy:.4f}")
                
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                continue
        
        best_result = max(self.results, key=lambda x: x['score'])
        logger.info(f"Best score: {best_result['score']:.4f}")
        logger.info(f"Best params: {best_result['params']}")
        
        return best_result['params']
    
    def get_tuning_history(self) -> List[Dict[str, Any]]:
        """Get full tuning history."""
        return self.results
    
    def plot_tuning_progress(self, save_path: Optional[str] = None):
        """Plot tuning progress."""
        try:
            import matplotlib.pyplot as plt
            
            trials = [r['trial'] for r in self.results]
            scores = [r['score'] for r in self.results]
            entropies = [r['entropy'] for r in self.results]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot scores
            ax1.plot(trials, scores, 'b-', alpha=0.7)
            ax1.set_ylabel('Score')
            ax1.set_title('Hyperparameter Tuning Progress')
            ax1.grid(True)
            
            # Plot entropy
            ax2.plot(trials, entropies, 'r-', alpha=0.7)
            ax2.set_xlabel('Trial')
            ax2.set_ylabel('Entropy')
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available for plotting")


if __name__ == '__main__':
    # Test hyperparameter macro
    macro = DynamicHyperparameterMacro('xgboost')
    
    print("Testing hyperparameter sampling:")
    for i in range(5):
        params = macro.sample_params()
        print(f"Sample {i+1}: {params}")
    
    # Simulate tuning
    print("\nSimulating entropy-based tuning:")
    for i in range(10):
        score = np.random.random()
        params = macro.sample_params()
        macro.update_entropy(score, params)
        print(f"Iteration {i+1}: Score={score:.4f}, Entropy={macro.entropy:.4f}")
    
    print(f"\nBest params: {macro.get_best_params()}")
