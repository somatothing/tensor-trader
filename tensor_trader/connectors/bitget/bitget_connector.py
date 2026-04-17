"""Bitget exchange connector for live trading and data fetching."""
import asyncio
import aiohttp
import json
import hmac
import hashlib
import base64
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BitgetConfig:
    """Configuration for Bitget API."""
    api_key: str
    api_secret: str
    passphrase: str
    base_url: str = "https://api.bitget.com"
    ws_url: str = "wss://ws.bitget.com/mix/v1/stream"
    testnet: bool = False
    
    def __post_init__(self):
        if self.testnet:
            self.base_url = "https://api.bitget.com"  # Bitget uses same URL with testnet flag


class BitgetConnector:
    """Connector for Bitget exchange API."""
    
    TIMEFRAME_MAP = {
        '1m': '1m',
        '5m': '5m', 
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
    }
    
    def __init__(self, config: BitgetConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection = None
        self._price_callbacks: List[Callable] = []
        self._running = False
        
    async def _init_session(self):
        """Initialize aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate Bitget API signature."""
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            self.config.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')
    
    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate request headers with authentication."""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, path, body)
        
        return {
            'ACCESS-KEY': self.config.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.config.passphrase,
            'Content-Type': 'application/json',
            'locale': 'en-US'
        }
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol."""
        await self._init_session()
        
        path = f"/api/mix/v1/market/ticker?symbol={symbol}"
        url = self.config.base_url + path
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('data', {})
            else:
                error_text = await response.text()
                logger.error(f"Failed to get ticker: {response.status} - {error_text}")
                return {}
    
    async def get_ohlcv(self, 
                       symbol: str, 
                       timeframe: str = '1m',
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get OHLCV (candlestick) data.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe string ('1m', '5m', '15m', '1h', '1d')
            limit: Number of candles to fetch
        
        Returns:
            List of OHLCV candles
        """
        await self._init_session()
        
        granularity = self.TIMEFRAME_MAP.get(timeframe, '1m')
        path = f"/api/mix/v1/market/candles?symbol={symbol}&granularity={granularity}&limit={limit}"
        url = self.config.base_url + path
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                candles = data.get('data', [])
                
                # Parse candles into standardized format
                parsed = []
                for candle in candles:
                    parsed.append({
                        'timestamp': int(candle[0]),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
                    })
                return parsed
            else:
                error_text = await response.text()
                logger.error(f"Failed to get OHLCV: {response.status} - {error_text}")
                return []
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance."""
        await self._init_session()
        
        path = "/api/mix/v1/account/accounts"
        url = self.config.base_url + path
        headers = self._get_headers('GET', path)
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('data', [])
            else:
                error_text = await response.text()
                logger.error(f"Failed to get balance: {response.status} - {error_text}")
                return {}
    
    async def place_order(self,
                         symbol: str,
                         side: str,  # 'buy' or 'sell'
                         order_type: str,  # 'limit' or 'market'
                         size: float,
                         price: Optional[float] = None,
                         client_oid: Optional[str] = None) -> Dict[str, Any]:
        """
        Place an order.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            order_type: 'limit' or 'market'
            size: Order size
            price: Limit price (required for limit orders)
            client_oid: Client order ID
        
        Returns:
            Order response
        """
        await self._init_session()
        
        path = "/api/mix/v1/order/placeOrder"
        url = self.config.base_url + path
        
        body = {
            'symbol': symbol,
            'marginCoin': 'USDT',
            'side': side.upper(),
            'orderType': order_type.upper(),
            'size': str(size),
            'clientOid': client_oid or str(int(time.time() * 1000)),
        }
        
        if order_type.lower() == 'limit' and price:
            body['price'] = str(price)
        
        body_json = json.dumps(body)
        headers = self._get_headers('POST', path, body_json)
        
        async with self.session.post(url, headers=headers, data=body_json) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('data', {})
            else:
                error_text = await response.text()
                logger.error(f"Failed to place order: {response.status} - {error_text}")
                return {'error': error_text}
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions."""
        await self._init_session()
        
        path = f"/api/mix/v1/position/singlePosition?symbol={symbol}" if symbol else "/api/mix/v1/position/allPosition"
        url = self.config.base_url + path
        headers = self._get_headers('GET', path)
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('data', [])
            else:
                error_text = await response.text()
                logger.error(f"Failed to get positions: {response.status} - {error_text}")
                return []
    
    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close a position."""
        await self._init_session()
        
        # First get position size
        positions = await self.get_positions(symbol)
        if not positions:
            return {'error': 'No position found'}
        
        position = positions[0]
        size = abs(float(position.get('total', 0)))
        pos_side = position.get('holdSide', '')
        
        if size == 0:
            return {'error': 'No position to close'}
        
        # Place opposite order to close
        close_side = 'sell' if pos_side == 'long' else 'buy'
        
        return await self.place_order(
            symbol=symbol,
            side=close_side,
            order_type='market',
            size=size
        )
    
    async def start_websocket(self, symbols: List[str], callback: Callable):
        """Start WebSocket connection for real-time price updates."""
        self._running = True
        self._price_callbacks.append(callback)
        
        subscribe_msg = {
            "op": "subscribe",
            "args": [
                {"instType": "mc", "channel": "ticker", "instId": symbol}
                for symbol in symbols
            ]
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


class BitgetMockConnector:
    """Mock Bitget connector for testing without real credentials."""
    
    def __init__(self):
        self.prices = {}
        self.positions = []
        self.orders = []
        
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return mock ticker data."""
        return {
            'symbol': symbol,
            'last': '50000.00',
            'high24h': '51000.00',
            'low24h': '49000.00',
            'baseVolume': '1000000',
            'quoteVolume': '50000000000'
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
            'USDT': {'available': '10000.00', 'frozen': '0.00'},
            'BTC': {'available': '0.5', 'frozen': '0.0'}
        }
    
    async def place_order(self, **kwargs) -> Dict[str, Any]:
        """Return mock order response."""
        order_id = f"mock_{int(time.time() * 1000)}"
        self.orders.append({
            'orderId': order_id,
            'status': 'filled',
            **kwargs
        })
        return {
            'orderId': order_id,
            'clientOid': kwargs.get('client_oid'),
            'status': 'filled'
        }
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return mock positions."""
        return self.positions
    
    async def close(self):
        """No-op for mock."""
        pass


if __name__ == '__main__':
    # Test with mock connector
    print("Testing Bitget Mock Connector...")
    
    async def test_mock():
        connector = BitgetMockConnector()
        
        # Test ticker
        ticker = await connector.get_ticker('BTCUSDT')
        print(f"Ticker: {ticker}")
        
        # Test OHLCV
        ohlcv = await connector.get_ohlcv('BTCUSDT', '1m', 5)
        print(f"OHLCV candles: {len(ohlcv)}")
        print(f"First candle: {ohlcv[0]}")
        
        # Test order
        order = await connector.place_order(
            symbol='BTCUSDT',
            side='buy',
            order_type='market',
            size=0.01
        )
        print(f"Order: {order}")
        
        print("\nBitget Mock Connector test PASSED!")
    
    asyncio.run(test_mock())
