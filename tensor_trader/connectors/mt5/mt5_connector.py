"""MetaTrader 5 connector for live trading.

Note: MT5 requires the MetaTrader 5 terminal to be running.
This connector uses the MetaTrader5 Python package to communicate
with the terminal via the MT5 API.
"""
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)

# Try to import MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not available. Install with: pip install MetaTrader5")


@dataclass
class MT5Config:
    """Configuration for MetaTrader 5 connection."""
    account: int
    password: str
    server: str
    path: Optional[str] = None  # Path to terminal64.exe
    timeout: int = 60000
    
    
class MT5Connector:
    """Connector for MetaTrader 5."""
    
    TIMEFRAME_MAP = {
        '1m': mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
        '5m': mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
        '15m': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
        '30m': mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
        '1h': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
        '4h': mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
        '1d': mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
    }
    
    def __init__(self, config: MT5Config):
        self.config = config
        self.connected = False
        
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not available")
    
    def connect(self) -> bool:
        """Connect to MT5 terminal."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 not available")
            return False
        
        # Initialize MT5
        if self.config.path:
            initialized = mt5.initialize(
                path=self.config.path,
                login=self.config.account,
                password=self.config.password,
                server=self.config.server,
                timeout=self.config.timeout
            )
        else:
            initialized = mt5.initialize()
        
        if not initialized:
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        # Login
        authorized = mt5.login(
            self.config.account,
            password=self.config.password,
            server=self.config.server
        )
        
        if not authorized:
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        self.connected = True
        logger.info(f"MT5 connected: Account {self.config.account} on {self.config.server}")
        return True
    
    def disconnect(self):
        """Disconnect from MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        
        info = mt5.account_info()
        if info is None:
            return {}
        
        return {
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'free_margin': info.margin_free,
            'leverage': info.leverage,
            'currency': info.currency,
        }
    
    def get_ohlcv(self, 
                  symbol: str, 
                  timeframe: str = '1m',
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get OHLCV data from MT5.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe string
            limit: Number of candles
        
        Returns:
            List of OHLCV candles
        """
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        
        tf = self.TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_M1)
        
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, limit)
        
        if rates is None:
            logger.error(f"Failed to get rates: {mt5.last_error()}")
            return []
        
        candles = []
        for rate in rates:
            candles.append({
                'timestamp': rate['time'],
                'open': rate['open'],
                'high': rate['high'],
                'low': rate['low'],
                'close': rate['close'],
                'volume': rate['tick_volume'],
            })
        
        return candles
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data."""
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
        
        return {
            'symbol': symbol,
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': tick.time,
        }
    
    def place_order(self,
                   symbol: str,
                   side: str,  # 'buy' or 'sell'
                   order_type: str,  # 'market' or 'limit'
                   volume: float,
                   price: Optional[float] = None,
                   sl: Optional[float] = None,
                   tp: Optional[float] = None,
                   comment: str = "TensorTrader") -> Dict[str, Any]:
        """
        Place an order in MT5.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            volume: Order volume in lots
            price: Limit price (for limit orders)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
        
        Returns:
            Order result
        """
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        
        # Map order type
        if order_type.lower() == 'market':
            order_type_mt5 = mt5.ORDER_TYPE_BUY if side.lower() == 'buy' else mt5.ORDER_TYPE_SELL
        elif order_type.lower() == 'limit':
            order_type_mt5 = mt5.ORDER_TYPE_BUY_LIMIT if side.lower() == 'buy' else mt5.ORDER_TYPE_SELL_LIMIT
        else:
            raise ValueError(f"Unknown order type: {order_type}")
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL if order_type.lower() == 'market' else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type_mt5,
            "price": price if price else mt5.symbol_info_tick(symbol).ask if side.lower() == 'buy' else mt5.symbol_info_tick(symbol).bid,
            "deviation": 10,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if sl:
            request["sl"] = sl
        if tp:
            request["tp"] = tp
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode}")
            return {'error': result.retcode, 'comment': result.comment}
        
        return {
            'order_id': result.order,
            'deal_id': result.deal,
            'volume': result.volume,
            'price': result.price,
        }
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions."""
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        return [{
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'type': 'buy' if pos.type == 0 else 'sell',
            'volume': pos.volume,
            'open_price': pos.price_open,
            'current_price': pos.price_current,
            'profit': pos.profit,
            'swap': pos.swap,
        } for pos in positions]
    
    def close_position(self, ticket: int) -> Dict[str, Any]:
        """Close a position by ticket."""
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {'error': 'Position not found'}
        
        pos = position[0]
        
        # Close with opposite order
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "TensorTrader Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {'error': result.retcode}
        
        return {
            'order_id': result.order,
            'deal_id': result.deal,
            'volume': result.volume,
            'price': result.price,
        }


class MT5MockConnector:
    """Mock MT5 connector for testing without MT5 terminal."""
    
    def __init__(self):
        self.connected = False
        self.positions = []
        self.orders = []
        
    def connect(self) -> bool:
        """Mock connect."""
        self.connected = True
        logger.info("MT5 Mock connected")
        return True
    
    def disconnect(self):
        """Mock disconnect."""
        self.connected = False
    
    def get_account_info(self) -> Dict[str, Any]:
        """Return mock account info."""
        return {
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'free_margin': 10000.0,
            'leverage': 100,
            'currency': 'USD',
        }
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[Dict[str, Any]]:
        """Return mock OHLCV data."""
        import random
        candles = []
        base_price = 1.1000 if 'EUR' in symbol else 50000.0
        
        for i in range(limit):
            timestamp = int(time.time()) - (i * 60)
            open_price = base_price + random.uniform(-0.001, 0.001)
            close_price = open_price + random.uniform(-0.0005, 0.0005)
            high_price = max(open_price, close_price) + random.uniform(0, 0.0003)
            low_price = min(open_price, close_price) - random.uniform(0, 0.0003)
            
            candles.append({
                'timestamp': timestamp,
                'open': round(open_price, 5),
                'high': round(high_price, 5),
                'low': round(low_price, 5),
                'close': round(close_price, 5),
                'volume': random.randint(100, 1000),
            })
        
        return candles
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return mock ticker."""
        return {
            'symbol': symbol,
            'bid': 1.1000,
            'ask': 1.1002,
            'last': 1.1001,
            'volume': 1000,
            'time': int(time.time()),
        }
    
    def place_order(self, **kwargs) -> Dict[str, Any]:
        """Return mock order."""
        order_id = int(time.time() * 1000)
        self.orders.append({
            'order_id': order_id,
            'status': 'filled',
            **kwargs
        })
        return {
            'order_id': order_id,
            'deal_id': order_id,
            'volume': kwargs.get('volume', 0.1),
            'price': kwargs.get('price', 1.1000),
        }
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return mock positions."""
        return self.positions
    
    def close_position(self, ticket: int) -> Dict[str, Any]:
        """Mock close position."""
        return {'order_id': ticket, 'status': 'closed'}


if __name__ == '__main__':
    # Test with mock connector
    print("Testing MT5 Mock Connector...")
    
    connector = MT5MockConnector()
    connector.connect()
    
    # Test account info
    info = connector.get_account_info()
    print(f"Account info: {info}")
    
    # Test OHLCV
    ohlcv = connector.get_ohlcv('EURUSD', '1m', 5)
    print(f"OHLCV candles: {len(ohlcv)}")
    print(f"First candle: {ohlcv[0]}")
    
    # Test order
    order = connector.place_order(
        symbol='EURUSD',
        side='buy',
        order_type='market',
        volume=0.1
    )
    print(f"Order: {order}")
    
    connector.disconnect()
    print("\nMT5 Mock Connector test PASSED!")
