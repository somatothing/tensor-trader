"""End-to-end integration tests for Tensor Trader."""
import pytest
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensor_trader.features.pipeline import FeaturePipeline, create_target_labels
from tensor_trader.models.boosting.xgboost_model import MarketXGBoost
from tensor_trader.models.tree.decision_tree import MarketDecisionTree
from tensor_trader.models.gnn.spread_tensor import SpreadTensorModel
from tensor_trader.inference.engine import InferenceEngine, SignalDirection
from tensor_trader.connectors.bitget.bitget_connector import BitgetMockConnector
from tensor_trader.connectors.hyperliquid.hyperliquid_connector import HyperliquidMockConnector
from tensor_trader.inference.executor import TradeExecutor, LiveTradingLoop


class TestFeaturePipeline:
    """Test feature engineering pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        pipeline = FeaturePipeline()
        assert pipeline is not None
        assert pipeline.drop_na == True
    
    def test_feature_generation(self):
        """Test features are generated from OHLCV data."""
        # Create sample OHLCV data
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 102,
            'low': np.random.randn(n).cumsum() + 98,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        # Ensure proper OHLCV structure
        df['high'] = np.maximum(df[['open', 'close']].max(axis=1) + 1, df['high'])
        df['low'] = np.minimum(df[['open', 'close']].min(axis=1) - 1, df['low'])
        
        pipeline = FeaturePipeline(drop_na=True)
        features = pipeline.transform(df)
        
        assert len(features) > 0
        assert len(pipeline.get_feature_names()) > 0
        print(f"Generated {len(pipeline.get_feature_names())} features")
    
    def test_target_labels(self):
        """Test target label creation."""
        df = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100
        })
        
        result = create_target_labels(df, lookahead=5, threshold=0.005)
        
        assert 'target' in result.columns
        assert 'future_return' in result.columns
        assert result['target'].isin([-1, 0, 1]).all()


class TestModels:
    """Test ML models."""
    
    def test_xgboost_training(self):
        """Test XGBoost model training."""
        np.random.seed(42)
        X = np.random.randn(200, 20)
        y = np.random.choice([-1, 0, 1], 200)
        
        model = MarketXGBoost(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        assert model.is_trained
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
    
    def test_decision_tree_training(self):
        """Test Decision Tree model training."""
        np.random.seed(42)
        X = np.random.randn(200, 20)
        y = np.random.choice([-1, 0, 1], 200)
        
        model = MarketDecisionTree(max_depth=5)
        model.fit(X, y)
        
        assert model.is_trained
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
    
    def test_spread_tensor_model(self):
        """Test Spread Tensor model."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.choice([-1, 0, 1], 100)
        
        model = SpreadTensorModel(input_dim=20)
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert len(predictions) == 10


class TestInferenceEngine:
    """Test inference engine."""
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.choice([-1, 0, 1], 100)
        
        # Train models
        xgb_model = MarketXGBoost(n_estimators=10, max_depth=3)
        xgb_model.fit(X, y)
        
        tree_model = MarketDecisionTree(max_depth=5)
        tree_model.fit(X, y)
        
        # Create inference engine
        models = {'xgboost': xgb_model, 'tree': tree_model}
        pipeline = FeaturePipeline()
        engine = InferenceEngine(models, pipeline)
        
        # Test prediction
        X_test = np.random.randn(1, 20)
        direction, confidence, details = engine.ensemble_predict(X_test)
        
        assert isinstance(direction, SignalDirection)
        assert 0 <= confidence <= 1
        assert 'predictions' in details
    
    def test_trading_decision(self):
        """Test trading decision generation."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 50000,
            'high': np.random.randn(n).cumsum() + 50100,
            'low': np.random.randn(n).cumsum() + 49900,
            'close': np.random.randn(n).cumsum() + 50000,
            'volume': np.random.randint(1000000, 10000000, n)
        })
        
        df['high'] = np.maximum(df[['open', 'close']].max(axis=1) + 10, df['high'])
        df['low'] = np.minimum(df[['open', 'close']].min(axis=1) - 10, df['low'])
        
        # Create and train models
        X = np.random.randn(50, 20)
        y = np.random.choice([-1, 0, 1], 50)
        
        xgb_model = MarketXGBoost(n_estimators=10, max_depth=3)
        xgb_model.fit(X, y)
        
        models = {'xgboost': xgb_model}
        pipeline = FeaturePipeline()
        engine = InferenceEngine(models, pipeline, confidence_threshold=0.5)
        
        # Make decision
        decision = engine.make_decision(df, 'BTCUSDT', '1m', 10000.0)
        
        # Decision may be None if confidence is too low
        if decision:
            assert decision.symbol == 'BTCUSDT'
            assert decision.timeframe == '1m'
            assert decision.confidence >= 0


class TestConnectors:
    """Test exchange connectors."""
    
    @pytest.mark.asyncio
    async def test_bitget_mock_connector(self):
        """Test Bitget mock connector."""
        connector = BitgetMockConnector()
        
        # Test ticker
        ticker = await connector.get_ticker('BTCUSDT')
        assert 'symbol' in ticker
        assert 'last' in ticker
        
        # Test OHLCV
        ohlcv = await connector.get_ohlcv('BTCUSDT', '1m', 10)
        assert len(ohlcv) == 10
        assert 'open' in ohlcv[0]
        
        # Test order
        order = await connector.place_order(
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            size=0.01
        )
        assert 'orderId' in order
    
    @pytest.mark.asyncio
    async def test_hyperliquid_mock_connector(self):
        """Test Hyperliquid mock connector."""
        connector = HyperliquidMockConnector()
        
        # Test ticker
        ticker = await connector.get_ticker('BTC')
        assert 'symbol' in ticker
        
        # Test OHLCV
        ohlcv = await connector.get_ohlcv('BTC', '1m', 10)
        assert len(ohlcv) == 10


class TestTradeExecution:
    """Test trade execution."""
    
    @pytest.mark.asyncio
    async def test_trade_executor(self):
        """Test trade executor."""
        connector = BitgetMockConnector()
        executor = TradeExecutor(connector, test_mode=True)
        
        from tensor_trader.inference.engine import TradingDecision, SignalDirection
        from datetime import datetime
        
        decision = TradingDecision(
            direction=SignalDirection.BUY,
            confidence=0.8,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            position_size=0.1,
            timestamp=datetime.now(),
            symbol='BTCUSDT',
            timeframe='1m',
            model_predictions={},
            risk_reward_ratio=2.0
        )
        
        result = await executor.execute_decision(decision)
        assert result.success
        assert result.symbol == 'BTCUSDT'


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""
    
    def test_full_pipeline(self):
        """Test complete pipeline from data to decision."""
        # 1. Generate sample data with realistic crypto volatility
        np.random.seed(42)
        n = 200
        
        # Bitcoin-like price with realistic volatility (~2-3% daily)
        returns = np.random.randn(n) * 0.015  # 1.5% std dev
        price = 50000 * np.exp(np.cumsum(returns))
        
        # Create OHLC from close
        open_p = price * (1 + np.random.randn(n) * 0.005)
        close = price
        high = np.maximum(open_p, close) * (1 + np.abs(np.random.randn(n)) * 0.01 + 0.005)
        low = np.minimum(open_p, close) * (1 - np.abs(np.random.randn(n)) * 0.01 - 0.005)
        
        df = pd.DataFrame({
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000000, 10000000, n)
        })
        
        # 2. Generate features
        pipeline = FeaturePipeline()
        features = pipeline.transform(df)
        
        # 3. Create labels
        features_with_target = create_target_labels(features, lookahead=5, threshold=0.005)
        
        # 4. Train model
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'future_return', 'target', 'target_return', 'target_binary']
        feature_cols = [c for c in features_with_target.columns if c not in exclude_cols]
        
        df_clean = features_with_target.dropna()
        if len(df_clean) < 50:
            pytest.skip("Insufficient data after cleaning")
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        model = MarketXGBoost(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        # 5. Make prediction
        predictions = model.predict(X[-10:])
        assert len(predictions) == 10
        
        print(f"\n✅ Full pipeline test PASSED")
        print(f"   - Data: {len(df)} candles")
        print(f"   - Features: {len(feature_cols)} features")
        print(f"   - Training samples: {len(X)}")
        print(f"   - Predictions: {predictions[:5]}")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
