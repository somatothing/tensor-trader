"""Feature engineering pipeline combining all feature modules."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .indicators.technical import calculate_all_indicators
from .smc.smart_money import calculate_smc_features
from .price_action.patterns import calculate_price_action_features


class FeaturePipeline:
    """Pipeline for generating all features from raw OHLCV data."""
    
    def __init__(self, drop_na: bool = True):
        self.drop_na = drop_na
        self.feature_columns = []
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw OHLCV data into feature-rich DataFrame.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
        
        Returns:
            DataFrame with all features added
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Start with technical indicators
        result = calculate_all_indicators(df)
        
        # Add SMC features
        result = calculate_smc_features(result)
        
        # Add price action features
        result = calculate_price_action_features(result)
        
        # Drop rows with NaN values if requested
        if self.drop_na:
            # Use less aggressive dropping - handle NaN values intelligently
            feature_cols = [col for col in result.columns if col not in required_cols]
            if feature_cols:
                # Fill NaN values: forward fill then backward fill
                result[feature_cols] = result[feature_cols].ffill().bfill()
                # Fill any remaining NaN with 0 (for columns like fvg_top that may be all NaN)
                result[feature_cols] = result[feature_cols].fillna(0)
                # Only drop rows that still have NaN (should be minimal now)
                result = result.dropna()
        
        # Store feature column names (excluding OHLCV)
        self.feature_columns = [col for col in result.columns if col not in required_cols]
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_columns
    
    def get_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about features."""
        stats = {
            'total_features': len(self.feature_columns),
            'numeric_features': len([c for c in self.feature_columns if df[c].dtype in ['float64', 'int64']]),
            'features_with_na': len([c for c in self.feature_columns if df[c].isna().any()]),
            'feature_categories': {}
        }
        
        # Categorize features
        categories = {
            'rsi': [c for c in self.feature_columns if 'rsi' in c],
            'macd': [c for c in self.feature_columns if 'macd' in c],
            'bollinger': [c for c in self.feature_columns if 'bb_' in c],
            'moving_avg': [c for c in self.feature_columns if 'sma_' in c or 'ema_' in c],
            'supertrend': [c for c in self.feature_columns if 'supertrend' in c],
            'ichimoku': [c for c in self.feature_columns if 'ichimoku' in c or 'senkou' in c or 'tenkan' in c or 'kijun' in c],
            'fibonacci': [c for c in self.feature_columns if 'fib_' in c],
            'smc': [c for c in self.feature_columns if any(x in c for x in ['fvg', 'bos', 'choch', 'swing', 'supply', 'demand'])],
            'price_action': [c for c in self.feature_columns if any(x in c for x in ['engulfing', 'hammer', 'star', 'marubozu', 'box', 'wave'])],
        }
        
        for cat, cols in categories.items():
            stats['feature_categories'][cat] = len(cols)
        
        return stats


def create_target_labels(df: pd.DataFrame, lookahead: int = 5, 
                        threshold: float = 0.005) -> pd.DataFrame:
    """Create target labels for classification.
    
    Args:
        df: DataFrame with close prices
        lookahead: Number of periods to look ahead
        threshold: Minimum price change threshold for labeling
    
    Returns:
        DataFrame with target labels added
    """
    result = df.copy()
    
    # Future returns
    result['future_return'] = result['close'].shift(-lookahead) / result['close'] - 1
    
    # Classification labels
    result['target'] = 0  # 0 = hold/neutral
    result.loc[result['future_return'] > threshold, 'target'] = 1  # 1 = buy
    result.loc[result['future_return'] < -threshold, 'target'] = -1  # -1 = sell
    
    # Regression target
    result['target_return'] = result['future_return']
    
    # Binary classification
    result['target_binary'] = (result['future_return'] > 0).astype(int)
    
    return result


if __name__ == '__main__':
    # Test with sample data
    import asyncio
    import sys
    sys.path.insert(0, '/Users/somatothing/Desktop/devs/repo_name/boards')
    from tensor_trader.data.fetchers.bitget_fetcher import BitgetFetcher
    
    async def test_pipeline():
        fetcher = BitgetFetcher()
        await fetcher.connect()
        
        try:
            print("Fetching sample data...")
            df = await fetcher.fetch_ohlcv('BTCUSDT', '1m', limit=100)
            print(f"Fetched {len(df)} candles")
            
            print("\nRunning feature pipeline...")
            pipeline = FeaturePipeline(drop_na=True)
            features = pipeline.transform(df)
            
            print(f"\nOriginal shape: {df.shape}")
            print(f"Feature shape: {features.shape}")
            print(f"Features generated: {len(pipeline.get_feature_names())}")
            
            stats = pipeline.get_feature_stats(features)
            print(f"\nFeature statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
            
            print("\nSample features:")
            print(features[pipeline.get_feature_names()[:10]].head())
            
            # Check for NaN in Ichimoku specifically
            ichimoku_cols = [c for c in features.columns if 'ichimoku' in c or 'senkou' in c or 'tenkan' in c or 'kijun' in c]
            if ichimoku_cols:
                print(f"\nIchimoku NaN check:")
                for col in ichimoku_cols[:5]:
                    nan_count = features[col].isna().sum()
                    print(f"  {col}: {nan_count} NaN values")
            
        finally:
            await fetcher.close()
    
    asyncio.run(test_pipeline())
