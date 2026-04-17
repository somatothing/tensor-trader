"""Live trading loop for 1-minute timeframe decision making."""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import numpy as np
import pandas as pd

from ..features.pipeline import FeaturePipeline
from ..inference.engine import InferenceEngine
from ..models.boosting.xgboost_model import MarketXGBoost
from ..models.tree.decision_tree import MarketDecisionTree
from ..models.gnn.spread_tensor import SpreadTensorModel
from ..connectors.bitget.bitget_connector import BitgetConnector, BitgetMockConnector
from ..connectors.hyperliquid.hyperliquid_connector import HyperliquidConnector, HyperliquidMockConnector
from ..serving.api import ModelManager

logger = logging.getLogger(__name__)


class LiveTradingLoop:
    """
    Live trading loop for 1-minute timeframe decision making.
    
    Fetches real-time data, computes features, runs ensemble inference,
    and executes trades when confidence threshold is met.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "bitget",
        confidence_threshold: float = 0.7,
        risk_per_trade: float = 0.02,
        paper_trading: bool = True,
        use_mock: bool = False
    ):
        """
        Initialize live trading loop.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange to use (bitget, hyperliquid)
            confidence_threshold: Minimum confidence to execute trade
            risk_per_trade: Risk per trade as fraction of capital
            paper_trading: If True, simulate trades without real execution
            use_mock: If True, use mock connectors for testing
        """
        self.symbol = symbol
        self.exchange = exchange
        self.confidence_threshold = confidence_threshold
        self.risk_per_trade = risk_per_trade
        self.paper_trading = paper_trading
        self.use_mock = use_mock
        
        # Initialize components
        self.feature_pipeline = FeaturePipeline()
        self.model_manager = ModelManager()
        self.inference_engine: Optional[InferenceEngine] = None
        self.connector: Optional[Any] = None
        
        # Trading state
        self.is_running = False
        self.trades: List[Dict[str, Any]] = []
        self.positions: Dict[str, Any] = {}
        self.balance = 10000.0  # Starting balance for paper trading
        
        # Performance tracking
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
    async def initialize(self):
        """Initialize connectors and load models."""
        logger.info("Initializing live trading loop...")
        
        # Initialize exchange connector
        if self.use_mock:
            if self.exchange == "bitget":
                self.connector = BitgetMockConnector()
            else:
                self.connector = HyperliquidMockConnector()
        else:
            if self.exchange == "bitget":
                self.connector = BitgetConnector()
            else:
                self.connector = HyperliquidConnector()
        
        # Connect to exchange
        await self.connector.connect()
        logger.info(f"Connected to {self.exchange}")
        
        # Load models
        await self._load_models()
        
        logger.info("Live trading loop initialized")
    
    async def _load_models(self):
        """Load ensemble models."""
        models = {}
        
        # Try to load from ONNX files first
        import os
        onnx_dir = "models/onnx"
        
        if os.path.exists(f"{onnx_dir}/xgboost.onnx"):
            models['xgboost'] = self.model_manager.load_onnx_model(f"{onnx_dir}/xgboost.onnx")
            logger.info("Loaded XGBoost ONNX model")
        
        if os.path.exists(f"{onnx_dir}/decision_tree.onnx"):
            models['tree'] = self.model_manager.load_onnx_model(f"{onnx_dir}/decision_tree.onnx")
            logger.info("Loaded Decision Tree ONNX model")
        
        # Load SpreadTensor from pickle
        if os.path.exists(f"{onnx_dir}/spread_tensor.pkl"):
            import pickle
            with open(f"{onnx_dir}/spread_tensor.pkl", 'rb') as f:
                models['spread_tensor'] = pickle.load(f)
            logger.info("Loaded SpreadTensor pickle model")
        
        if models:
            self.inference_engine = InferenceEngine(
                models=models,
                feature_pipeline=self.feature_pipeline,
                confidence_threshold=self.confidence_threshold
            )
            logger.info(f"Inference engine initialized with {len(models)} models")
        else:
            logger.warning("No models loaded - using fallback")
            # Create dummy models for testing
            await self._create_fallback_models()
    
    async def _create_fallback_models(self):
        """Create fallback models if no saved models exist."""
        logger.info("Creating fallback models...")
        
        # Generate dummy training data
        np.random.seed(42)
        X = np.random.randn(100, 20)
        y = np.random.choice([-1, 0, 1], 100)
        
        models = {}
        
        # XGBoost
        xgb = MarketXGBoost(n_estimators=5, max_depth=3)
        xgb.fit(X, y)
        models['xgboost'] = xgb
        
        # Decision Tree
        tree = MarketDecisionTree(max_depth=5)
        tree.fit(X, y)
        models['tree'] = tree
        
        # SpreadTensor
        tensor = SpreadTensorModel(input_dim=20, n_components=8)
        tensor.fit(X, y)
        models['spread_tensor'] = tensor
        
        self.inference_engine = InferenceEngine(
            models=models,
            feature_pipeline=self.feature_pipeline,
            confidence_threshold=self.confidence_threshold
        )
        logger.info("Fallback models created")
    
    async def fetch_data(self, timeframe: str = "1m", limit: int = 200) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            timeframe: Timeframe string (1m, 5m, 15m, 1h, 1d)
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            df = await self.connector.get_ohlcv(self.symbol, timeframe, limit)
            logger.info(f"Fetched {len(df)} candles from {self.exchange}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise
    
    async def compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute features from OHLCV data.
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Feature matrix
        """
        features = self.feature_pipeline.transform(df)
        
        # Get feature columns (exclude OHLCV and target columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in features.columns if c not in exclude_cols]
        
        # Drop NaN rows
        features_clean = features.dropna()
        
        if len(features_clean) == 0:
            raise ValueError("No valid features after cleaning")
        
        return features_clean[feature_cols].values
    
    async def make_decision(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make trading decision using ensemble inference.
        
        Args:
            features: Feature matrix
        
        Returns:
            Decision dict with direction, confidence, and details
        """
        if self.inference_engine is None:
            raise RuntimeError("Inference engine not initialized")
        
        # Use last row for prediction
        X_latest = features[-1:]
        
        direction, confidence, details = self.inference_engine.ensemble_predict(X_latest)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
    
    async def execute_trade(self, decision: Dict[str, Any], current_price: float):
        """
        Execute trade based on decision.
        
        Args:
            decision: Decision dict from make_decision
            current_price: Current market price
        """
        direction = decision['direction']
        confidence = decision['confidence']
        
        # Skip if confidence below threshold or direction is HOLD
        if confidence < self.confidence_threshold or direction.value == 0:
            logger.info(f"Skipping trade: confidence={confidence:.3f}, direction={direction}")
            return
        
        # Calculate position size
        position_size = self.balance * self.risk_per_trade / current_price
        
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'direction': direction.name,
            'confidence': confidence,
            'price': current_price,
            'size': position_size,
            'paper_trading': self.paper_trading
        }
        
        if self.paper_trading:
            # Simulate trade
            logger.info(f"PAPER TRADE: {direction.name} {position_size:.6f} {self.symbol} @ {current_price}")
            self.positions[self.symbol] = {
                'direction': direction.name,
                'entry_price': current_price,
                'size': position_size
            }
        else:
            # Execute real trade
            try:
                side = "buy" if direction.value == 1 else "sell"
                order = await self.connector.place_order(
                    symbol=self.symbol,
                    side=side,
                    amount=position_size,
                    order_type="market"
                )
                trade['order_id'] = order.get('id', 'unknown')
                logger.info(f"LIVE TRADE EXECUTED: {order}")
            except Exception as e:
                logger.error(f"Trade execution failed: {e}")
                trade['error'] = str(e)
        
        self.trades.append(trade)
    
    async def run_single_iteration(self):
        """Run single iteration of trading loop."""
        try:
            # Fetch data
            df = await self.fetch_data("1m", 200)
            
            if len(df) < 50:
                logger.warning("Insufficient data, skipping iteration")
                return
            
            # Compute features
            features = await self.compute_features(df)
            
            # Make decision
            decision = await self.make_decision(features)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Log decision
            logger.info(f"Decision: {decision['direction'].name}, "
                       f"Confidence: {decision['confidence']:.3f}, "
                       f"Price: {current_price:.2f}")
            
            # Execute trade if conditions met
            await self.execute_trade(decision, current_price)
            
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}", exc_info=True)
    
    async def run(self, duration_minutes: Optional[int] = None):
        """
        Run trading loop.
        
        Args:
            duration_minutes: If set, run for specified duration, else run indefinitely
        """
        logger.info("Starting live trading loop...")
        self.is_running = True
        
        start_time = datetime.now()
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                logger.info(f"=== Iteration {iteration} ===")
                
                await self.run_single_iteration()
                
                # Check duration
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        logger.info(f"Duration limit ({duration_minutes}m) reached")
                        break
                
                # Wait for next minute
                next_minute = datetime.now() + timedelta(minutes=1)
                sleep_seconds = (next_minute - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    logger.info(f"Sleeping {sleep_seconds:.1f}s until next minute...")
                    await asyncio.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted by user")
        finally:
            self.is_running = False
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown trading loop."""
        logger.info("Shutting down trading loop...")
        self.is_running = False
        
        if self.connector:
            await self.connector.disconnect()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print trading summary."""
        logger.info("=" * 50)
        logger.info("TRADING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Paper Trading: {self.paper_trading}")
        logger.info(f"Exchange: {self.exchange}")
        logger.info(f"Symbol: {self.symbol}")
        
        if self.trades:
            avg_confidence = sum(t['confidence'] for t in self.trades) / len(self.trades)
            logger.info(f"Avg Confidence: {avg_confidence:.3f}")
        
        logger.info("=" * 50)
    
    def stop(self):
        """Signal to stop trading loop."""
        self.is_running = False


async def main():
    """Main entry point for live trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tensor Trader Live Trading Loop')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--exchange', default='bitget', choices=['bitget', 'hyperliquid'])
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--duration', type=int, help='Duration in minutes (optional)')
    parser.add_argument('--paper', action='store_true', default=True, help='Paper trading mode')
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    parser.add_argument('--mock', action='store_true', help='Use mock connectors')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trading loop
    loop = LiveTradingLoop(
        symbol=args.symbol,
        exchange=args.exchange,
        confidence_threshold=args.confidence,
        paper_trading=not args.live,
        use_mock=args.mock
    )
    
    # Initialize
    await loop.initialize()
    
    # Run
    try:
        await loop.run(duration_minutes=args.duration)
    except Exception as e:
        logger.error(f"Trading loop error: {e}", exc_info=True)
    finally:
        await loop.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
