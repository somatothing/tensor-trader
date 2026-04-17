"""Spread-Tensor state representation for multi-timeframe market analysis.

This module provides a lightweight, numpy-based alternative to Graph Neural Networks
for representing market state as high-dimensional tensors of spreads and features.
No PyTorch dependency - pure numpy implementation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe in the multiplex."""
    name: str  # e.g., '1m', '5m', '15m', '1h', '1d'
    period_seconds: int
    features: List[str]
    weight: float = 1.0


class SpreadTensor:
    """
    Multi-timeframe market state representation as a 3D tensor:
    [timeframes x assets x features]
    
    This captures spatial-temporal relationships without requiring
    PyTorch Geometric or any deep learning framework.
    """
    
    def __init__(self, 
                 timeframes: List[TimeframeConfig],
                 assets: List[str],
                 feature_dim: int,
                 lookback_windows: Dict[str, int] = None):
        """
        Initialize SpreadTensor.
        
        Args:
            timeframes: List of timeframe configurations
            assets: List of asset symbols
            feature_dim: Dimension of feature vectors
            lookback_windows: Dict mapping timeframe name to lookback window size
        """
        self.timeframes = {tf.name: tf for tf in timeframes}
        self.timeframe_order = [tf.name for tf in timeframes]
        self.assets = assets
        self.n_assets = len(assets)
        self.feature_dim = feature_dim
        self.lookback_windows = lookback_windows or {tf: 20 for tf in self.timeframe_order}
        
        # Tensor shape: [timeframes, assets, features]
        self.tensor = np.zeros((len(timeframes), len(assets), feature_dim))
        self.asset_index = {asset: i for i, asset in enumerate(assets)}
        self.tf_index = {name: i for i, name in enumerate(self.timeframe_order)}
        
        # Historical buffers for temporal aggregation
        self.history: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Cross-timeframe spread matrix
        self.spread_matrix = np.zeros((len(timeframes), len(timeframes)))
        
        logger.info(f"SpreadTensor initialized: {len(timeframes)} timeframes, "
                   f"{len(assets)} assets, {feature_dim} features")
    
    def update_timeframe(self, 
                        timeframe: str, 
                        asset: str, 
                        features: np.ndarray,
                        price: float,
                        timestamp: Optional[pd.Timestamp] = None):
        """
        Update tensor slice for a specific timeframe and asset.
        
        Args:
            timeframe: Timeframe name (e.g., '1m')
            asset: Asset symbol
            features: Feature vector (will be truncated/padded to feature_dim)
            price: Current price for spread calculation
            timestamp: Optional timestamp for logging
        """
        if timeframe not in self.tf_index:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        if asset not in self.asset_index:
            raise ValueError(f"Unknown asset: {asset}")
        
        tf_idx = self.tf_index[timeframe]
        asset_idx = self.asset_index[asset]
        
        # Ensure feature vector is correct size
        if len(features) != self.feature_dim:
            if len(features) < self.feature_dim:
                # Pad with zeros
                features = np.pad(features, (0, self.feature_dim - len(features)))
            else:
                # Truncate
                features = features[:self.feature_dim]
        
        # Update tensor
        self.tensor[tf_idx, asset_idx, :] = features
        
        # Update history
        self.history[timeframe][asset].append(features)
        window = self.lookback_windows.get(timeframe, 20)
        if len(self.history[timeframe][asset]) > window:
            self.history[timeframe][asset].pop(0)
        
        # Update spread matrix (cross-timeframe price relationships)
        self._update_spread_matrix(timeframe, asset, price)
    
    def _update_spread_matrix(self, timeframe: str, asset: str, price: float):
        """Update cross-timeframe spread relationships."""
        # Store price for spread calculation
        if not hasattr(self, '_price_cache'):
            self._price_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        self._price_cache[timeframe][asset] = price
        
        # Calculate spreads between timeframes for this asset
        tf_idx = self.tf_index[timeframe]
        for other_tf, other_idx in self.tf_index.items():
            if other_tf in self._price_cache and asset in self._price_cache[other_tf]:
                other_price = self._price_cache[other_tf][asset]
                if other_price != 0:
                    spread = abs(price - other_price) / other_price
                    self.spread_matrix[tf_idx, other_idx] = spread
                    self.spread_matrix[other_idx, tf_idx] = spread
    
    def compute_temporal_features(self, timeframe: str, asset: str) -> np.ndarray:
        """
        Compute temporal aggregation features from history.
        
        Returns: [mean, std, min, max, trend] for each feature dimension
        """
        if timeframe not in self.history or asset not in self.history[timeframe]:
            return np.zeros(self.feature_dim * 5)
        
        history = np.array(self.history[timeframe][asset])
        if len(history) < 2:
            return np.zeros(self.feature_dim * 5)
        
        # Temporal statistics
        mean_feat = np.mean(history, axis=0)
        std_feat = np.std(history, axis=0)
        min_feat = np.min(history, axis=0)
        max_feat = np.max(history, axis=0)
        
        # Trend (linear slope approximation)
        if len(history) >= 3:
            x = np.arange(len(history))
            trend = np.array([
                np.polyfit(x, history[:, i], 1)[0] 
                for i in range(self.feature_dim)
            ])
        else:
            trend = np.zeros(self.feature_dim)
        
        return np.concatenate([mean_feat, std_feat, min_feat, max_feat, trend])
    
    def compute_cross_asset_spread(self, feature_idx: int = 0) -> np.ndarray:
        """
        Compute spread relationships between assets for a specific feature.
        
        Returns: [assets x assets] spread matrix
        """
        n = self.n_assets
        spread = np.zeros((n, n))
        
        for i, asset_i in enumerate(self.assets):
            for j, asset_j in enumerate(self.assets):
                if i != j:
                    # Calculate spread across all timeframes
                    spreads = []
                    for tf in self.timeframe_order:
                        tf_idx = self.tf_index[tf]
                        val_i = self.tensor[tf_idx, i, feature_idx]
                        val_j = self.tensor[tf_idx, j, feature_idx]
                        if val_j != 0:
                            spreads.append(abs(val_i - val_j) / abs(val_j))
                    
                    if spreads:
                        spread[i, j] = np.mean(spreads)
        
        return spread
    
    def flatten(self, include_spreads: bool = True, 
                include_temporal: bool = True) -> np.ndarray:
        """
        Flatten tensor to feature vector for ML models.
        
        Args:
            include_spreads: Include cross-timeframe spread features
            include_temporal: Include temporal aggregation features
        
        Returns:
            Flattened feature vector
        """
        # Base tensor flatten
        features = self.tensor.flatten()
        
        if include_spreads:
            # Add spread matrix (upper triangle only to avoid redundancy)
            spread_flat = self.spread_matrix[np.triu_indices_from(self.spread_matrix, k=1)]
            features = np.concatenate([features, spread_flat])
        
        if include_temporal:
            # Add temporal features for each timeframe-asset pair
            temporal = []
            for tf in self.timeframe_order:
                for asset in self.assets:
                    temporal.extend(self.compute_temporal_features(tf, asset))
            features = np.concatenate([features, np.array(temporal)])
        
        return features
    
    def get_attention_weights(self, query_timeframe: str) -> np.ndarray:
        """
        Compute attention-like weights between timeframes based on spread similarity.
        
        This mimics GNN attention without requiring neural networks.
        
        Args:
            query_timeframe: Timeframe to compute attention from
        
        Returns:
            Attention weights for each timeframe
        """
        if query_timeframe not in self.tf_index:
            raise ValueError(f"Unknown timeframe: {query_timeframe}")
        
        query_idx = self.tf_index[query_timeframe]
        
        # Compute similarity based on spread matrix
        similarities = []
        for tf in self.timeframe_order:
            tf_idx = self.tf_index[tf]
            # Negative spread = higher similarity
            sim = 1.0 / (1.0 + self.spread_matrix[query_idx, tf_idx])
            similarities.append(sim)
        
        # Softmax-like normalization
        similarities = np.array(similarities)
        exp_sim = np.exp(similarities - np.max(similarities))
        weights = exp_sim / np.sum(exp_sim)
        
        return weights
    
    def aggregate_cross_timeframe(self, 
                                  timeframe: str,
                                  aggregation: str = 'weighted') -> np.ndarray:
        """
        Aggregate features across timeframes using attention weights.
        
        Args:
            timeframe: Target timeframe
            aggregation: 'weighted', 'mean', or 'max'
        
        Returns:
            Aggregated feature tensor [assets x features]
        """
        tf_idx = self.tf_index[timeframe]
        
        if aggregation == 'weighted':
            weights = self.get_attention_weights(timeframe)
            result = np.zeros((self.n_assets, self.feature_dim))
            for i, w in enumerate(weights):
                result += w * self.tensor[i, :, :]
            return result
        
        elif aggregation == 'mean':
            return np.mean(self.tensor, axis=0)
        
        elif aggregation == 'max':
            return np.max(self.tensor, axis=0)
        
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'tensor': self.tensor.tolist(),
            'spread_matrix': self.spread_matrix.tolist(),
            'timeframes': self.timeframe_order,
            'assets': self.assets,
            'feature_dim': self.feature_dim,
            'lookback_windows': self.lookback_windows
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpreadTensor':
        """Deserialize from dictionary."""
        timeframes = [
            TimeframeConfig(name=tf, period_seconds=0, features=[])
            for tf in data['timeframes']
        ]
        
        st = cls(
            timeframes=timeframes,
            assets=data['assets'],
            feature_dim=data['feature_dim'],
            lookback_windows=data.get('lookback_windows', {})
        )
        
        st.tensor = np.array(data['tensor'])
        st.spread_matrix = np.array(data['spread_matrix'])
        
        return st


class SpreadTensorFeatureExtractor:
    """
    Feature extractor that converts market data into SpreadTensor format.
    """
    
    def __init__(self, 
                 timeframes: List[str] = None,
                 assets: List[str] = None,
                 base_features: List[str] = None):
        """
        Initialize feature extractor.
        
        Args:
            timeframes: List of timeframe strings (e.g., ['1m', '5m', '15m', '1h', '1d'])
            assets: List of asset symbols
            base_features: List of base feature names to extract
        """
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '1d']
        self.assets = assets or ['BTCUSDT', 'ETHUSDT']
        self.base_features = base_features or [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'sma_20', 'sma_50', 'volume', 'returns', 'volatility'
        ]
        
        # Create configs
        tf_configs = []
        period_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1d': 86400}
        for tf in self.timeframes:
            tf_configs.append(TimeframeConfig(
                name=tf,
                period_seconds=period_map.get(tf, 60),
                features=self.base_features
            ))
        
        self.spread_tensor = SpreadTensor(
            timeframes=tf_configs,
            assets=self.assets,
            feature_dim=len(self.base_features)
        )
    
    def extract_from_dataframe(self, 
                               df: pd.DataFrame,
                               timeframe: str,
                               asset: str) -> np.ndarray:
        """
        Extract features from OHLCV DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe name
            asset: Asset symbol
        
        Returns:
            Feature vector
        """
        if len(df) < 50:
            return np.zeros(len(self.base_features))
        
        features = []
        
        # RSI
        if 'rsi' in self.base_features:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50)
        
        # MACD
        if 'macd' in self.base_features or 'macd_signal' in self.base_features:
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            if 'macd' in self.base_features:
                features.append(macd.iloc[-1])
            if 'macd_signal' in self.base_features:
                features.append(macd_signal.iloc[-1])
        
        # Bollinger Bands
        if 'bb_upper' in self.base_features or 'bb_lower' in self.base_features:
            sma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            if 'bb_upper' in self.base_features:
                features.append((sma20 + 2 * std20).iloc[-1])
            if 'bb_lower' in self.base_features:
                features.append((sma20 - 2 * std20).iloc[-1])
        
        # SMAs
        if 'sma_20' in self.base_features:
            features.append(df['close'].rolling(window=20).mean().iloc[-1])
        if 'sma_50' in self.base_features:
            features.append(df['close'].rolling(window=50).mean().iloc[-1])
        
        # Volume
        if 'volume' in self.base_features:
            features.append(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1] 
                          if len(df) >= 20 else 1.0)
        
        # Returns
        if 'returns' in self.base_features:
            features.append(df['close'].pct_change().iloc[-1] * 100)
        
        # Volatility
        if 'volatility' in self.base_features:
            features.append(df['close'].pct_change().rolling(20).std().iloc[-1] * 100 
                          if len(df) >= 20 else 0)
        
        return np.array(features)
    
    def update(self, 
              df: pd.DataFrame,
              timeframe: str,
              asset: str,
              price: float):
        """Update SpreadTensor with new data."""
        features = self.extract_from_dataframe(df, timeframe, asset)
        self.spread_tensor.update_timeframe(timeframe, asset, features, price)
    
    def get_features(self, **kwargs) -> np.ndarray:
        """Get flattened feature vector."""
        return self.spread_tensor.flatten(**kwargs)


def create_multiplex_state(timeframe_data: Dict[str, pd.DataFrame],
                          assets: List[str],
                          current_prices: Dict[str, float]) -> SpreadTensor:
    """
    Create a SpreadTensor from multiplexed timeframe data.
    
    Args:
        timeframe_data: Dict mapping timeframe to OHLCV DataFrame
        assets: List of asset symbols
        current_prices: Dict mapping asset to current price
    
    Returns:
        Populated SpreadTensor
    """
    # Determine feature dimension from first dataframe
    sample_df = list(timeframe_data.values())[0]
    extractor = SpreadTensorFeatureExtractor(
        timeframes=list(timeframe_data.keys()),
        assets=assets
    )
    
    # Update tensor for each timeframe
    for tf, df in timeframe_data.items():
        for asset in assets:
            price = current_prices.get(asset, df['close'].iloc[-1] if len(df) > 0 else 0)
            extractor.update(df, tf, asset, price)
    
    return extractor.spread_tensor


if __name__ == '__main__':
    # Test SpreadTensor
    print("Testing SpreadTensor...")
    
    # Create sample configuration
    timeframes = [
        TimeframeConfig('1m', 60, ['rsi', 'macd', 'volume']),
        TimeframeConfig('5m', 300, ['rsi', 'macd', 'volume']),
        TimeframeConfig('15m', 900, ['rsi', 'macd', 'volume']),
    ]
    assets = ['BTCUSDT', 'ETHUSDT']
    
    st = SpreadTensor(timeframes, assets, feature_dim=3)
    
    # Simulate updates
    np.random.seed(42)
    for tf in ['1m', '5m', '15m']:
        for asset in assets:
            for i in range(25):
                features = np.random.randn(3)
                price = 50000 + np.random.randn() * 100
                st.update_timeframe(tf, asset, features, price)
    
    print(f"Tensor shape: {st.tensor.shape}")
    print(f"Spread matrix shape: {st.spread_matrix.shape}")
    
    # Test flattening
    flat = st.flatten()
    print(f"Flattened features: {len(flat)} dimensions")
    
    # Test attention
    weights = st.get_attention_weights('1m')
    print(f"Attention weights for 1m: {weights}")
    
    # Test aggregation
    agg = st.aggregate_cross_timeframe('1m', 'weighted')
    print(f"Aggregated shape: {agg.shape}")
    
    print("\nSpreadTensor test PASSED!")
