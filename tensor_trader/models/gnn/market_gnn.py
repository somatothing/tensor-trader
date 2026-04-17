"""Graph Neural Network for spatial-temporal market analysis."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import PyTorch and PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using sklearn MLP fallback")

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class MarketGNN:
    """Graph Neural Network for market structure learning.
    
    Models market data as a graph where:
    - Nodes represent different timeframes or assets
    - Edges represent temporal or correlation relationships
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 3,
                 num_layers: int = 3, dropout: float = 0.3, use_attention: bool = True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.is_trained = False
        
        if TORCH_AVAILABLE:
            self._init_torch_model()
        else:
            self._init_sklearn_fallback()
    
    def _init_torch_model(self):
        """Initialize PyTorch model."""
        if PYG_AVAILABLE:
            self.model = _MarketGNNPyG(
                self.input_dim, self.hidden_dim, self.output_dim,
                self.num_layers, self.dropout, self.use_attention
            )
        else:
            # Simple MLP fallback
            layers = []
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            
            for _ in range(self.num_layers - 1):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            
            layers.append(nn.Linear(self.hidden_dim, self.output_dim))
            self.model = nn.Sequential(*layers)
        
        self.model.eval()
    
    def _init_sklearn_fallback(self):
        """Initialize sklearn MLP fallback."""
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim,) * self.num_layers,
            activation='relu',
            alpha=self.dropout,
            max_iter=500,
            random_state=42
        )
    
    def predict(self, X: np.ndarray, edge_index: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions on numpy arrays."""
        if TORCH_AVAILABLE:
            self.model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(X)
                if PYG_AVAILABLE and edge_index is not None:
                    edge_tensor = torch.LongTensor(edge_index)
                    logits = self.model(x_tensor, edge_tensor)
                else:
                    logits = self.model(x_tensor)
                probs = F.softmax(logits, dim=-1)
                return probs.numpy()
        else:
            # Sklearn fallback
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")
            return self.model.predict_proba(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray, edge_index: Optional[np.ndarray] = None):
        """Train the model."""
        if TORCH_AVAILABLE:
            # Simple training loop for PyTorch
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            self.model.train()
            x_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y + 1)  # Shift labels to 0,1,2
            
            for epoch in range(100):
                optimizer.zero_grad()
                
                if PYG_AVAILABLE and edge_index is not None:
                    edge_tensor = torch.LongTensor(edge_index)
                    out = self.model(x_tensor, edge_tensor)
                else:
                    out = self.model(x_tensor)
                
                loss = criterion(out, y_tensor)
                loss.backward()
                optimizer.step()
            
            self.is_trained = True
        else:
            # Sklearn training
            self.model.fit(X, y)
            self.is_trained = True
    
    def save(self, filepath: str):
        """Save model."""
        if TORCH_AVAILABLE:
            torch.save(self.model.state_dict(), filepath)
        else:
            import joblib
            joblib.dump(self.model, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        if TORCH_AVAILABLE:
            self.model.load_state_dict(torch.load(filepath))
        else:
            import joblib
            self.model = joblib.load(filepath)
        self.is_trained = True


if TORCH_AVAILABLE and PYG_AVAILABLE:
    class _MarketGNNPyG(nn.Module):
        """PyTorch Geometric implementation."""
        
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                     num_layers: int, dropout: float, use_attention: bool):
            super().__init__()
            self.convs = nn.ModuleList()
            
            if use_attention:
                self.convs.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout))
                conv_hidden = hidden_dim * 4
            else:
                self.convs.append(GCNConv(input_dim, hidden_dim))
                conv_hidden = hidden_dim
            
            for i in range(num_layers - 1):
                if use_attention and i < num_layers - 2:
                    self.convs.append(GATConv(conv_hidden, hidden_dim, heads=4, dropout=dropout))
                    conv_hidden = hidden_dim * 4
                else:
                    self.convs.append(GCNConv(conv_hidden, hidden_dim))
                    conv_hidden = hidden_dim
            
            self.fc1 = nn.Linear(conv_hidden, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x, edge_index, batch=None):
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
            
            if batch is not None:
                x = global_mean_pool(x, batch)
            
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


if __name__ == '__main__':
    # Test with dummy data
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"PyG available: {PYG_AVAILABLE}")
    
    # Create dummy data
    batch_size = 32
    num_features = 20
    num_classes = 3
    
    model = MarketGNN(input_dim=num_features, hidden_dim=32, output_dim=num_classes)
    
    x = np.random.randn(batch_size, num_features).astype(np.float32)
    output = model.predict(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0]}")
    print(f"Sum of probabilities: {output[0].sum():.4f}")
    print("GNN: SUCCESS")
