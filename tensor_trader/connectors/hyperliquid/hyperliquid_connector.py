"""Hyperliquid exchange connector for live trading and data fetching."""
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class HyperliquidConfig:
    """Configuration for Hyperliquid API."""
    wallet_address: str
    private_key: Optional[str] = None  # For signing transactions
    base_url: str = "https://api.hyperliquid.xyz"
    ws_url: str = "wss://api.hyperliquid.xyz/ws"
    testnet: bool = False
    
    def __post_init__(self):
        if self.testnet:
            self.base_url = "https://api.hyperliquid-testnet.xyz"
            self.ws_url = "wss://api.hyperliquid-testnet.xyz/ws"


class HyperliquidConnector:
    """Connector for Hyperliquid exchange API."""
    
    TIMEFRAME_MAP = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
    }
    
    def __init__(self, config: HyperliquidConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection = None
        self._price_callbacks: List[Callable] = []
        self._running = False
        
    async def _init_session(self):
        """Initialize aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol."""
        await self._init_session()
        
        # Hyperliquid uses allMids for all mid prices
        url = f"{self.config.base_url}/info"
        payload = {"type": "allMids"}
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                # Find the symbol in the response
                for item in data:
                    if item.get('coin') == symbol:
                        return {
                            'symbol': symbol,
                            'last': item.get('mid', '0'),
                            'markPrice': item.get('mark', '0'),
                            'indexPrice': item.get('index', '0'),
                        }
                return {}
            else:
                error_text = await response.text()
                logger.error(f"Failed to get ticker: {response.status} - {error_text}")
                return {}
    
    async def get_ohlcv(self, 
                       symbol: str, 
                       timeframe: str = '1m',
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get OHLCV (candlestick) data from Hyperliquid.
        
        Args:
            symbol: Trading pair (e.g., 'BTC')
            timeframe: Timeframe string ('1m', '5m', '15m', '1h', '1d')
            limit: Number of candles to fetch
        
        Returns:
            List of OHLCV candles
        """
        await self._init_session()
        
        # Calculate start time based on limit
        # Hyperliquid uses milliseconds
        end_time = int(time.time() * 1000)
        
        # Map timeframe to milliseconds
        tf_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        
        granularity = tf_ms.get(timeframe, 60 * 1000)
        start_time = end_time - (limit * granularity)
        
        url = f"{self.config.base_url}/info"
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "startTime": start_time,
                "endTime": end_time,
                "interval": timeframe
            }
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                candles = []
                
                for candle in data:
                    candles.append({
                        'timestamp': candle.get('t', 0),
                        'open': float(candle.get('o', 0)),
                        'high': float(candle.get('h', 0)),
                        'low': float(candle.get('l', 0)),
                        'close': float(candle.get('c', 0)),
                        'volume': float(candle.get('v', 0)),
                    })
                
                return candles
            else:
                error_text = await response.text()
                logger.error(f"Failed to get OHLCV: {response.status} - {error_text}")
                return []
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        await self._init_session()
        
        url = f"{self.config.base_url}/info"
        payload = {
            "type": "clearinghouseState",
            "user": self.config.wallet_address
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    'marginSummary': data.get('marginSummary', {}),
                    'assetPositions': data.get('assetPositions', []),
                    'withdrawable': data.get('withdrawable', '0'),
                }
            else:
                error_text = await response.text()
                logger.error(f"Failed to get balance: {response.status} - {error_text}")
                return {}
    
    async def place_order(self,
                         symbol: str,
                         side: str,  # 'B' for buy, 'A' for sell
                         size: float,
                         price: Optional[float] = None,
                         order_type: str = 'Limit',
                         reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place an order on Hyperliquid.
        
        Note: This is a simplified version. Real implementation requires
        EIP-712 signing which is complex.
        
        Args:
            symbol: Trading coin (e.g., 'BTC')
            side: 'B' for buy/long, 'A' for sell/short
            size: Order size
            price: Limit price (None for market orders)
            order_type: 'Limit' or 'Market'
            reduce_only: Whether order reduces position only
        
        Returns:
            Order response
        """
        await self._init_session()
        
        # This is a placeholder - real implementation requires signing
        logger.warning("Hyperliquid order placement requires EIP-712 signing")
        
        url = f"{self.config.base_url}/exchange"
        
        order = {
            "coin": symbol,
            "isBuy": side.upper() == 'B',
            "sz": str(size),
            "limitPx": str(price) if price else "0",
            "orderType": {"limit": {"tif": "Gtc"}} if order_type == 'Limit' else {"market": {}},
            "reduceOnly": reduce_only,
        }
        
        # Note: Real implementation would sign this with private key
        payload = {
            "action": {"type": "order", "orders": [order]},
            "nonce": int(time.time() * 1000),
            "signature": "PLACEHOLDER_SIGNATURE"  # Requires actual signing
        }
        
        # For now, return mock response
        return {
            'status': 'mock_order',
            'order': order,
            'warning': 'Real orders require EIP-712 signing implementation'
        }
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions."""
        await self._init_session()
        
        url = f"{self.config.base_url}/info"
        payload = {
            "type": "clearinghouseState",
            "user": self.config.wallet_address
        }
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                positions = data.get('assetPositions', [])
                
                if symbol:
                    positions = [p for p in positions if p.get('coin') == symbol]
                
                return positions
            else:
                error_text = await response.text()
                logger.error(f"Failed to get positions: {response.status} - {error_text}")
                return []
    
    async def start_websocket(self, symbols: List[str], callback: Callable):
        """Start WebSocket connection for real-time price updates."""
        self._running = True
        self._price_callbacks.append(callback)
        
        subscribe_msg = {
            "method": "subscribe",
            "subscription": {"type": "allMids"}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.config.ws_url) as ws:
                    self.ws_connection = ws
                    
                    # Subscribe to channels
                    await ws.send_json(subscribe_msg)
                    
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            # Call all registered callbacks
                            for cb in self._price_callbacks:
                                try:
                                    cb(data)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                        
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                        
                        if not self._running:
                            break
                            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
    
    async def stop_websocket(self):
        """Stop WebSocket connection."""
        self._running = False
        if self.ws_connection:
            await self.ws_connection.close()
    
    async def close(self):
        """Close all connections."""
        await self.stop_websocket()
        if self.session:
            await self.session.close()


class HyperliquidMockConnector:
    """Mock Hyperliquid connector for testing without real credentials."""
    
    def __init__(self):
        self.prices = {}
        self.positions = []
        self.orders = []
        
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return mock ticker data."""
        return {
            'symbol': symbol,
            'last': '50000.00',
            'markPrice': '50000.00',
            'indexPrice': '50000.00',
        }
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[Dict[str, Any]]:
        """Return mock OHLCV data."""
        import random
        candles = []
        base_price = 50000.0
        
        for i in range(limit):
            timestamp = int(time.time() * 1000) - (i * 60000)
            open_price = base_price + random.uniform(-100, 100)
            close_price = open_price + random.uniform(-50, 50)
            high_price = max(open_price, close_price) + random.uniform(0, 30)
            low_price = min(open_price, close_price) - random.uniform(0, 30)
            
            candles.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(random.uniform(0.1, 10.0), 4),
            })
        
        return candles
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """Return mock balance."""
        return {
            'marginSummary': {
                'accountValue': '10000.00',
                'totalMarginUsed': '1000.00',
                'totalPositionNotional': '5000.00'
            },
            'assetPositions': [],
            'withdrawable': '9000.00',
        }
    
    async def place_order(self, **kwargs) -> Dict[str, Any]:
        """Return mock order response."""
        order_id = f"mock_{int(time.time() * 1000)}"
        self.orders.append({
            'orderId': order_id,
            'status': 'mock_order',
            **kwargs
        })
        return {
            'status': 'mock_order',
            'orderId': order_id,
            'warning': 'Mock order - no real execution'
        }
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return mock positions."""
        return self.positions
    
    async def close(self):
        """No-op for mock."""
        pass


if __name__ == '__main__':
    # Test with mock connector
    print("Testing Hyperliquid Mock Connector...")
    
    async def test_mock():
        connector = HyperliquidMockConnector()
        
        # Test ticker
        ticker = await connector.get_ticker('BTC')
        print(f"Ticker: {ticker}")
        
        # Test OHLCV
        ohlcv = await connector.get_ohlcv('BTC', '1m', 5)
        print(f"OHLCV candles: {len(ohlcv)}")
        print(f"First candle: {ohlcv[0]}")
        
        # Test order
        order = await connector.place_order(
            symbol='BTC',
            side='B',
            size=0.01,
            price=50000
        )
        print(f"Order: {order}")
        
        print("\nHyperliquid Mock Connector test PASSED!")
    
    asyncio.run(test_mock())
