"""Base connector interface for all exchanges."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Order data class."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    client_order_id: Optional[str] = None
    reduce_only: bool = False


@dataclass
class Position:
    """Position data class."""
    symbol: str
    side: OrderSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    leverage: float = 1.0


@dataclass
class AccountBalance:
    """Account balance data class."""
    asset: str
    free: float
    locked: float
    total: float


@dataclass
class Ticker:
    """Ticker data class."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: int


@dataclass
class OHLCV:
    """OHLCV candle data class."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BaseExchangeConnector(ABC):
    """Abstract base class for all exchange connectors."""
    
    TIMEFRAME_MAP = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self._rate_limit_remaining = 100
        self._rate_limit_reset = 0
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the exchange."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the exchange."""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker data."""
        pass
    
    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str = '1m', 
                        limit: int = 100) -> List[OHLCV]:
        """Get OHLCV candlestick data."""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> List[AccountBalance]:
        """Get account balance."""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place an order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get current positions."""
        pass
    
    @abstractmethod
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close a position."""
        pass
    
    def check_rate_limit(self) -> bool:
        """Check if rate limit allows request."""
        import time
        if time.time() > self._rate_limit_reset:
            self._rate_limit_remaining = 100
        return self._rate_limit_remaining > 0
    
    def update_rate_limit(self, remaining: int, reset_time: int):
        """Update rate limit info."""
        self._rate_limit_remaining = remaining
        self._rate_limit_reset = reset_time


class ConnectorFactory:
    """Factory for creating exchange connectors."""
    
    _connectors = {}
    
    @classmethod
    def register(cls, name: str, connector_class: type):
        """Register a connector class."""
        cls._connectors[name.lower()] = connector_class
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseExchangeConnector:
        """Create a connector instance."""
        name = name.lower()
        if name not in cls._connectors:
            raise ValueError(f"Unknown connector: {name}. Available: {list(cls._connectors.keys())}")
        return cls._connectors[name](config)
    
    @classmethod
    def list_connectors(cls) -> List[str]:
        """List available connectors."""
        return list(cls._connectors.keys())
