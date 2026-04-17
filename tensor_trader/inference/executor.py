"""Trade executor for live trading."""
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from ..connectors.base_connector import BaseExchangeConnector, Order, OrderSide, OrderType
from .engine import TradingDecision, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Trade execution result."""
    success: bool
    order_id: Optional[str]
    symbol: str
    side: str
    size: float
    price: float
    timestamp: datetime
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


class TradeExecutor:
    """Executor for trading decisions."""
    
    def __init__(self, connector: BaseExchangeConnector, test_mode: bool = True):
        """
        Initialize trade executor.
        
        Args:
            connector: Exchange connector instance
            test_mode: If True, only simulates trades without execution
        """
        self.connector = connector
        self.test_mode = test_mode
        self.trade_history: List[TradeResult] = []
        self.open_positions: Dict[str, Dict] = {}
        
    async def execute_decision(self, decision: TradingDecision) -> TradeResult:
        """
        Execute a trading decision.
        
        Args:
            decision: TradingDecision to execute
        
        Returns:
            TradeResult with execution details
        """
        if decision.direction == SignalDirection.HOLD:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=decision.symbol,
                side="hold",
                size=0,
                price=0,
                timestamp=datetime.now(),
                error="HOLD signal - no trade executed"
            )
        
        # Map direction to order side
        side = OrderSide.BUY if decision.direction == SignalDirection.BUY else OrderSide.SELL
        
        # Create order
        order = Order(
            symbol=decision.symbol,
            side=side,
            order_type=OrderType.MARKET,  # Use market for immediate execution
            size=decision.position_size,
            price=None,  # Market order
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            client_order_id=f"tt_{int(datetime.now().timestamp() * 1000)}"
        )
        
        if self.test_mode:
            logger.info(f"[TEST MODE] Would execute: {side.value} {decision.position_size} {decision.symbol} "
                       f"@ ~{decision.entry_price}")
            return TradeResult(
                success=True,
                order_id=f"test_{int(datetime.now().timestamp() * 1000)}",
                symbol=decision.symbol,
                side=side.value,
                size=decision.position_size,
                price=decision.entry_price,
                timestamp=datetime.now(),
                error=None,
                raw_response={'test_mode': True, 'decision': decision.to_dict()}
            )
        
        # Execute real trade
        try:
            result = await self.connector.place_order(order)
            
            if result.get('error'):
                trade_result = TradeResult(
                    success=False,
                    order_id=None,
                    symbol=decision.symbol,
                    side=side.value,
                    size=decision.position_size,
                    price=decision.entry_price,
                    timestamp=datetime.now(),
                    error=result.get('error'),
                    raw_response=result
                )
            else:
                trade_result = TradeResult(
                    success=True,
                    order_id=result.get('order_id') or result.get('orderId'),
                    symbol=decision.symbol,
                    side=side.value,
                    size=result.get('size', decision.position_size),
                    price=result.get('price', decision.entry_price),
                    timestamp=datetime.now(),
                    error=None,
                    raw_response=result
                )
                
                # Track open position
                self.open_positions[decision.symbol] = {
                    'order_id': trade_result.order_id,
                    'side': side.value,
                    'entry_price': trade_result.price,
                    'size': trade_result.size,
                    'stop_loss': decision.stop_loss,
                    'take_profit': decision.take_profit,
                    'timestamp': datetime.now()
                }
            
            self.trade_history.append(trade_result)
            return trade_result
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            trade_result = TradeResult(
                success=False,
                order_id=None,
                symbol=decision.symbol,
                side=side.value,
                size=decision.position_size,
                price=decision.entry_price,
                timestamp=datetime.now(),
                error=str(e),
                raw_response=None
            )
            self.trade_history.append(trade_result)
            return trade_result
    
    async def close_position(self, symbol: str) -> TradeResult:
        """Close an open position."""
        if symbol not in self.open_positions:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side="close",
                size=0,
                price=0,
                timestamp=datetime.now(),
                error=f"No open position for {symbol}"
            )
        
        if self.test_mode:
            logger.info(f"[TEST MODE] Would close position: {symbol}")
            del self.open_positions[symbol]
            return TradeResult(
                success=True,
                order_id=f"test_close_{int(datetime.now().timestamp() * 1000)}",
                symbol=symbol,
                side="close",
                size=self.open_positions.get(symbol, {}).get('size', 0),
                price=0,
                timestamp=datetime.now(),
                error=None,
                raw_response={'test_mode': True, 'action': 'close'}
            )
        
        try:
            result = await self.connector.close_position(symbol)
            
            if symbol in self.open_positions:
                del self.open_positions[symbol]
            
            return TradeResult(
                success=True,
                order_id=result.get('order_id'),
                symbol=symbol,
                side="close",
                size=result.get('size', 0),
                price=result.get('price', 0),
                timestamp=datetime.now(),
                error=None,
                raw_response=result
            )
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return TradeResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side="close",
                size=0,
                price=0,
                timestamp=datetime.now(),
                error=str(e),
                raw_response=None
            )
    
    def get_trade_stats(self) -> Dict[str, Any]:
        """Get trade execution statistics."""
        if not self.trade_history:
            return {'total_trades': 0}
        
        successful = [t for t in self.trade_history if t.success]
        failed = [t for t in self.trade_history if not t.success]
        
        return {
            'total_trades': len(self.trade_history),
            'successful_trades': len(successful),
            'failed_trades': len(failed),
            'success_rate': len(successful) / len(self.trade_history) if self.trade_history else 0,
            'open_positions': len(self.open_positions),
            'test_mode': self.test_mode
        }


class LiveTradingLoop:
    """Main loop for live trading."""
    
    def __init__(self,
                 connector: BaseExchangeConnector,
                 inference_engine: Any,
                 symbols: List[str],
                 timeframe: str = '1m',
                 check_interval: int = 60,
                 test_mode: bool = True):
        """
        Initialize live trading loop.
        
        Args:
            connector: Exchange connector
            inference_engine: InferenceEngine instance
            symbols: List of symbols to trade
            timeframe: Primary timeframe for decisions
            check_interval: Seconds between checks
            test_mode: If True, simulates trades
        """
        self.connector = connector
        self.inference_engine = inference_engine
        self.symbols = symbols
        self.timeframe = timeframe
        self.check_interval = check_interval
        self.test_mode = test_mode
        self.executor = TradeExecutor(connector, test_mode=test_mode)
        self.running = False
        
    async def fetch_data(self, symbol: str, limit: int = 100) -> Optional[Any]:
        """Fetch OHLCV data from exchange."""
        try:
            candles = await self.connector.get_ohlcv(symbol, self.timeframe, limit)
            if not candles:
                return None
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame([{
                'timestamp': c.timestamp if hasattr(c, 'timestamp') else c['timestamp'],
                'open': c.open if hasattr(c, 'open') else c['open'],
                'high': c.high if hasattr(c, 'high') else c['high'],
                'low': c.low if hasattr(c, 'low') else c['low'],
                'close': c.close if hasattr(c, 'close') else c['close'],
                'volume': c.volume if hasattr(c, 'volume') else c['volume'],
            } for c in candles])
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def get_account_balance(self) -> float:
        """Get total account balance."""
        try:
            balances = await self.connector.get_account_balance()
            if balances:
                # Sum USDT or base currency
                total = sum(b.total for b in balances if hasattr(b, 'total'))
                return total if total > 0 else 10000.0  # Default for testing
            return 10000.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 10000.0  # Default for testing
    
    async def run_once(self):
        """Run one iteration of the trading loop."""
        for symbol in self.symbols:
            try:
                # Fetch data
                df = await self.fetch_data(symbol)
                if df is None or len(df) < 50:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Get account balance
                balance = await self.get_account_balance()
                
                # Make decision
                decision = self.inference_engine.make_decision(
                    df, symbol, self.timeframe, balance
                )
                
                if decision:
                    # Execute trade
                    result = await self.executor.execute_decision(decision)
                    if result.success:
                        logger.info(f"Trade executed: {result.order_id}")
                    else:
                        logger.error(f"Trade failed: {result.error}")
                        
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    async def run(self):
        """Run the trading loop continuously."""
        self.running = True
        logger.info(f"Starting live trading loop (test_mode={self.test_mode})")
        
        while self.running:
            await self.run_once()
            await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop the trading loop."""
        self.running = False
        logger.info("Trading loop stopped")
