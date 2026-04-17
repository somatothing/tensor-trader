"""Enhanced live trading loop with dashboard and comprehensive reporting."""
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
from ..dashboard.console_ui import ConsoleDashboard, create_progress_animator
from ..reporting.metrics import TradeMetricsReporter, TradeRecord, PositionRecord, EquityTracker

logger = logging.getLogger(__name__)


class EnhancedLiveTradingLoop:
    """
    Enhanced live trading loop with real-time dashboard and comprehensive reporting.
    
    Features:
    - Real-time multiplex timeframe display (1m, 5m, 15m, 1h, 1d)
    - Console progress animations
    - Comprehensive PnL/equity/trade-metrics reporting
    - 250+ feature indicators
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        exchange: str = "bitget",
        confidence_threshold: float = 0.7,
        risk_per_trade: float = 0.02,
        paper_trading: bool = True,
        use_mock: bool = False,
        enable_dashboard: bool = True,
        initial_equity: float = 10000.0
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.confidence_threshold = confidence_threshold
        self.risk_per_trade = risk_per_trade
        self.paper_trading = paper_trading
        self.use_mock = use_mock
        self.enable_dashboard = enable_dashboard
        self.initial_equity = initial_equity
        
        # Initialize components
        self.feature_pipeline = FeaturePipeline()
        self.model_manager = ModelManager()
        self.inference_engine: Optional[InferenceEngine] = None
        self.connector: Optional[Any] = None
        
        # Dashboard and reporting
        self.dashboard: Optional[ConsoleDashboard] = None
        self.progress_animator = None
        self.metrics_reporter = TradeMetricsReporter(initial_equity=initial_equity)
        self.equity_tracker = EquityTracker(initial_equity=initial_equity)
        
        # Trading state
        self.is_running = False
        self.trades: List[Dict[str, Any]] = []
        self.positions: Dict[str, Any] = {}
        self.balance = initial_equity
        self.current_equity = initial_equity
        
        # Performance tracking
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.iteration = 0
        
        # Multiplex timeframe data
        self.timeframe_data: Dict[str, pd.DataFrame] = {}
        self.last_update_times: Dict[str, datetime] = {}
        
    async def initialize(self):
        """Initialize connectors, dashboard, and load models."""
        logger.info("Initializing enhanced live trading loop...")
        
        # Initialize dashboard if enabled
        if self.enable_dashboard:
            self.dashboard = ConsoleDashboard(refresh_rate=1.0)
            self.progress_animator = create_progress_animator()
            self.dashboard.update_status(
                exchange=self.exchange,
                symbol=self.symbol,
                mode="PAPER" if self.paper_trading else "LIVE",
                is_connected=False,
                is_trading=False
            )
            self.dashboard.log("Initializing trading system...", "INFO")
        
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
        if self.dashboard:
            self.dashboard.update_status(is_connected=True)
            self.dashboard.log(f"Connected to {self.exchange}", "SUCCESS")
        
        # Load models
        await self._load_models()
        
        logger.info("Enhanced live trading loop initialized")
        if self.dashboard:
            self.dashboard.log("System ready for trading", "SUCCESS")
    
    async def _load_models(self):
        """Load ensemble models with progress animation."""
        if self.progress_animator:
            self.progress_animator.console.print("[cyan]Loading models...[/cyan]")
        
        models = {}
        
        # Try to load from ONNX files first
        import os
        onnx_dir = "models/onnx"
        
        if os.path.exists(f"{onnx_dir}/xgboost.onnx"):
            models['xgboost'] = self.model_manager.load_onnx_model(f"{onnx_dir}/xgboost.onnx")
            if self.dashboard:
                self.dashboard.log("Loaded XGBoost ONNX model", "INFO")
        
        if os.path.exists(f"{onnx_dir}/decision_tree.onnx"):
            models['tree'] = self.model_manager.load_onnx_model(f"{onnx_dir}/decision_tree.onnx")
            if self.dashboard:
                self.dashboard.log("Loaded Decision Tree ONNX model", "INFO")
        
        # Load SpreadTensor from pickle
        if os.path.exists(f"{onnx_dir}/spread_tensor.pkl"):
            import pickle
            with open(f"{onnx_dir}/spread_tensor.pkl", 'rb') as f:
                models['spread_tensor'] = pickle.load(f)
            if self.dashboard:
                self.dashboard.log("Loaded SpreadTensor model", "INFO")
        
        if models:
            self.inference_engine = InferenceEngine(
                models=models,
                feature_pipeline=self.feature_pipeline,
                confidence_threshold=self.confidence_threshold
            )
            if self.dashboard:
                self.dashboard.log(f"Inference engine initialized with {len(models)} models", "SUCCESS")
        else:
            if self.dashboard:
                self.dashboard.log("No models found - creating fallback models", "WARNING")
            await self._create_fallback_models()
    
    async def _create_fallback_models(self):
        """Create fallback models if no saved models exist."""
        if self.progress_animator:
            self.progress_animator.console.print("[yellow]Creating fallback models...[/yellow]")
        
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
        
        if self.dashboard:
            self.dashboard.log("Fallback models created", "INFO")
    
    async def fetch_multiplex_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all timeframes (1m, 5m, 15m, 1h, 1d).
        
        Returns:
            Dictionary of timeframe -> DataFrame
        """
        timeframes = ["1m", "5m", "15m", "1h", "1d"]
        data = {}
        
        if self.progress_animator:
            self.progress_animator.console.print("[cyan]Fetching multiplex timeframe data...[/cyan]")
        
        for tf in timeframes:
            try:
                limit = 200 if tf in ["1m", "5m"] else 100
                df = await self.connector.get_ohlcv(self.symbol, tf, limit)
                data[tf] = df
                self.timeframe_data[tf] = df
                self.last_update_times[tf] = datetime.now()
                
                if self.dashboard:
                    # Update timeframe data on dashboard
                    latest = df.iloc[-1] if not df.empty else {}
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    
                    change_pct = ((latest['close'] - prev['close']) / prev['close'] * 100) if prev.get('close') else 0
                    
                    self.dashboard.update_timeframe(tf, {
                        'price': latest.get('close', 0),
                        'open': latest.get('open', 0),
                        'high': latest.get('high', 0),
                        'low': latest.get('low', 0),
                        'close': latest.get('close', 0),
                        'volume': latest.get('volume', 0),
                        'change_pct': change_pct,
                        'rsi': 0,  # Will be calculated
                        'signal': 'NEUTRAL'
                    })
                    
                    self.progress_animator.data_fetch_progress(self.symbol, tf, len(df), limit)
                    
            except Exception as e:
                logger.error(f"Failed to fetch {tf} data: {e}")
                if self.dashboard:
                    self.dashboard.log(f"Failed to fetch {tf} data: {e}", "ERROR")
        
        return data
    
    async def compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute features from OHLCV data with progress animation.
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            Feature matrix
        """
        if self.progress_animator:
            self.progress_animator.console.print("[magenta]Computing 250+ features...[/magenta]")
        
        features = self.feature_pipeline.transform(df)
        
        # Get feature columns (exclude OHLCV and target columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in features.columns if c not in exclude_cols]
        
        if self.dashboard:
            self.dashboard.log(f"Generated {len(feature_cols)} features", "INFO")
        
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
        
        if self.progress_animator:
            signal_name = direction.name if hasattr(direction, 'name') else str(direction)
            self.progress_animator.inference_progress(self.symbol, confidence, signal_name)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
    
    async def execute_trade(self, decision: Dict[str, Any], current_price: float):
        """
        Execute trade based on decision with reporting.
        
        Args:
            decision: Decision dict from make_decision
            current_price: Current market price
        """
        direction = decision['direction']
        confidence = decision['confidence']
        
        # Skip if confidence below threshold or direction is HOLD
        if confidence < self.confidence_threshold:
            if self.dashboard:
                self.dashboard.log(f"Signal below threshold: {confidence:.2%}", "INFO")
            return
        
        # Get direction value
        direction_value = direction.value if hasattr(direction, 'value') else 0
        if direction_value == 0:
            return
        
        # Calculate position size
        position_size = self.balance * self.risk_per_trade / current_price
        
        trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.iteration}"
        
        trade = {
            'trade_id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'direction': direction.name if hasattr(direction, 'name') else str(direction),
            'confidence': confidence,
            'price': current_price,
            'size': position_size,
            'paper_trading': self.paper_trading
        }
        
        if self.paper_trading:
            # Simulate trade
            if self.dashboard:
                self.dashboard.log(
                    f"PAPER TRADE: {direction.name if hasattr(direction, 'name') else direction} "
                    f"{position_size:.6f} {self.symbol} @ {current_price}",
                    "SUCCESS"
                )
            
            # Record position
            position = PositionRecord(
                position_id=trade_id,
                symbol=self.symbol,
                side='long' if direction_value == 1 else 'short',
                entry_price=current_price,
                current_price=current_price,
                size=position_size,
                confidence=confidence
            )
            self.metrics_reporter.update_position(position)
            
            self.positions[trade_id] = {
                'direction': direction.name if hasattr(direction, 'name') else str(direction),
                'entry_price': current_price,
                'size': position_size,
                'entry_time': datetime.now()
            }
        else:
            # Execute real trade
            try:
                side = "buy" if direction_value == 1 else "sell"
                order = await self.connector.place_order(
                    symbol=self.symbol,
                    side=side,
                    amount=position_size,
                    order_type="market"
                )
                trade['order_id'] = order.get('id', 'unknown')
                if self.dashboard:
                    self.dashboard.log(f"LIVE TRADE EXECUTED: {order}", "SUCCESS")
            except Exception as e:
                logger.error(f"Trade execution failed: {e}")
                if self.dashboard:
                    self.dashboard.log(f"Trade execution failed: {e}", "ERROR")
                trade['error'] = str(e)
        
        self.trades.append(trade)
    
    async def run_single_iteration(self):
        """Run single iteration of trading loop with full dashboard updates."""
        self.iteration += 1
        
        try:
            # Update dashboard status
            if self.dashboard:
                self.dashboard.update_status(is_trading=True)
                self.dashboard.log(f"=== Iteration {self.iteration} ===", "INFO")
            
            # Fetch multiplex data
            data = await self.fetch_multiplex_data()
            
            if "1m" not in data or len(data["1m"]) < 50:
                if self.dashboard:
                    self.dashboard.log("Insufficient 1m data, skipping iteration", "WARNING")
                return
            
            # Compute features on 1m data
            features = await self.compute_features(data["1m"])
            
            # Make decision
            decision = await self.make_decision(features)
            
            # Get current price
            current_price = data["1m"]['close'].iloc[-1]
            
            # Update dashboard with decision
            if self.dashboard:
                signal_name = decision['direction'].name if hasattr(decision['direction'], 'name') else str(decision['direction'])
                self.dashboard.update_timeframe("1m", {
                    'signal': signal_name,
                    'price': current_price
                })
            
            # Execute trade if conditions met
            await self.execute_trade(decision, current_price)
            
            # Update equity and metrics
            self.current_equity = self.balance + self.total_pnl
            self.equity_tracker.add_point(self.current_equity)
            
            if self.dashboard:
                self.dashboard.update_equity(self.current_equity)
                self.dashboard.update_metrics(
                    total_trades=len(self.trades),
                    total_pnl=self.total_pnl,
                    current_drawdown=self.equity_tracker.current_drawdown
                )
                
                # Update positions
                positions_list = []
                for pos_id, pos in self.positions.items():
                    unrealized_pnl = (current_price - pos['entry_price']) * pos['size'] if pos['direction'] == 'BUY' else (pos['entry_price'] - current_price) * pos['size']
                    positions_list.append({
                        'symbol': self.symbol,
                        'side': pos['direction'],
                        'size': pos['size'],
                        'entry_price': pos['entry_price'],
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_pct': (unrealized_pnl / (pos['entry_price'] * pos['size'])) * 100 if pos['entry_price'] > 0 else 0
                    })
                self.dashboard.update_positions(positions_list)
            
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}", exc_info=True)
            if self.dashboard:
                self.dashboard.log(f"Error: {e}", "ERROR")
                self.dashboard.update_status(last_error=str(e))
    
    async def run(self, duration_minutes: Optional[int] = None):
        """
        Run trading loop with dashboard.
        
        Args:
            duration_minutes: If set, run for specified duration, else run indefinitely
        """
        logger.info("Starting enhanced live trading loop...")
        self.is_running = True
        
        start_time = datetime.now()
        
        # Start dashboard if enabled
        if self.enable_dashboard and self.dashboard:
            dashboard_task = asyncio.create_task(self.dashboard.run())
        
        try:
            while self.is_running:
                await self.run_single_iteration()
                
                # Check duration
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        if self.dashboard:
                            self.dashboard.log(f"Duration limit ({duration_minutes}m) reached", "INFO")
                        break
                
                # Wait for next minute
                next_minute = datetime.now() + timedelta(minutes=1)
                sleep_seconds = (next_minute - datetime.now()).total_seconds()
                if sleep_seconds > 0:
                    if self.dashboard:
                        self.dashboard.log(f"Sleeping {sleep_seconds:.1f}s until next minute...", "INFO")
                    await asyncio.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            if self.dashboard:
                self.dashboard.log("Trading loop interrupted by user", "WARNING")
        finally:
            self.is_running = False
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown trading loop and generate final report."""
        logger.info("Shutting down trading loop...")
        
        if self.dashboard:
            self.dashboard.log("Shutting down...", "INFO")
            self.dashboard.update_status(is_trading=False)
        
        if self.connector:
            await self.connector.disconnect()
        
        # Generate final report
        report = self.metrics_reporter.generate_full_report()
        
        # Save report
        report_path = f"reports/trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import os
        os.makedirs("reports", exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.dashboard:
            self.dashboard.log(f"Report saved to {report_path}", "SUCCESS")
            self.dashboard.print_summary()
        
        logger.info(f"Trading report saved to {report_path}")
    
    def stop(self):
        """Signal to stop trading loop."""
        self.is_running = False


async def main():
    """Main entry point for enhanced live trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tensor Trader Enhanced Live Trading')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--exchange', default='bitget', choices=['bitget', 'hyperliquid'])
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--duration', type=int, help='Duration in minutes')
    parser.add_argument('--paper', action='store_true', default=True, help='Paper trading')
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    parser.add_argument('--mock', action='store_true', help='Use mock connectors')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable dashboard')
    parser.add_argument('--equity', type=float, default=10000.0, help='Initial equity')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trading loop
    loop = EnhancedLiveTradingLoop(
        symbol=args.symbol,
        exchange=args.exchange,
        confidence_threshold=args.confidence,
        paper_trading=not args.live,
        use_mock=args.mock,
        enable_dashboard=not args.no_dashboard,
        initial_equity=args.equity
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
