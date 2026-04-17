"""Bitget exchange data fetcher."""
import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import pandas as pd

from .base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)


class BitgetFetcher(BaseFetcher):
    """Bitget API data fetcher."""
    
    BASE_URL = "https://api.bitget.com"
    BASE_URL_TESTNET = "https://api.bitget.com"
    
    TIMEFRAME_MAP = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '1h': '1h',
        '4h': '4h',
        '1d': '1day',
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 passphrase: Optional[str] = None, testnet: bool = False):
        super().__init__(api_key, api_secret, passphrase, testnet)
        self.base_url = self.BASE_URL_TESTNET if testnet else self.BASE_URL
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def connect(self) -> bool:
        """Initialize aiohttp session."""
        if self.session is None or self.session.closed:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(
                headers={'Content-Type': 'application/json'},
                connector=connector
            )
        return True
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate Bitget API signature."""
        if not self.api_secret:
            return ""
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return mac.digest().hex()
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                           auth: bool = False) -> Dict:
        """Make authenticated request to Bitget API."""
        await self._wait_for_rate_limit()
        
        if self.session is None:
            await self.connect()
        
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        if auth and self.api_key:
            timestamp = str(int(time.time() * 1000))
            body = json.dumps(params) if params else ""
            signature = self._generate_signature(timestamp, method, endpoint, body)
            headers.update({
                'ACCESS-KEY': self.api_key,
                'ACCESS-SIGN': signature,
                'ACCESS-TIMESTAMP': timestamp,
                'ACCESS-PASSPHRASE': self.passphrase or '',
            })
        
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                self._handle_rate_limit(dict(response.headers))
                
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Bitget API error {response.status}: {text}")
                
                data = await response.json()
                if data.get('code') != '00000':
                    raise Exception(f"Bitget API error: {data.get('msg')}")
                
                return data.get('data', {})
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m',
                         since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candles from Bitget.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candle timeframe
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch
        """
        tf = self.TIMEFRAME_MAP.get(timeframe, timeframe)
        
        params = {
            'symbol': symbol,
            'granularity': tf,
            'limit': str(min(limit, 1000))
        }
        
        if since:
            params['startTime'] = str(since)
            params['endTime'] = str(since + limit * self._timeframe_to_ms(timeframe))
        
        try:
            data = await self._make_request('GET', '/api/v2/spot/market/candles', params)
            
            if not data or not isinstance(data, list):
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            candles = []
            for candle in data:
                if isinstance(candle, list) and len(candle) >= 6:
                    candles.append({
                        'timestamp': int(candle[0]),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
                    })
                elif isinstance(candle, dict):
                    candles.append({
                        'timestamp': int(candle.get('ts', candle.get('timestamp', 0))),
                        'open': float(candle.get('open', 0)),
                        'high': float(candle.get('high', 0)),
                        'low': float(candle.get('low', 0)),
                        'close': float(candle.get('close', 0)),
                        'volume': float(candle.get('vol', candle.get('volume', 0))),
                    })
            
            df = pd.DataFrame(candles)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker data."""
        params = {'symbol': symbol}
        try:
            data = await self._make_request('GET', '/api/v2/spot/market/ticker', params)
            return data[0] if isinstance(data, list) else data
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}
    
    async def fetch_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Fetch order book."""
        params = {'symbol': symbol, 'limit': str(limit)}
        try:
            return await self._make_request('GET', '/api/v2/spot/market/orderbook', params)
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {}
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe to milliseconds."""
        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
        }
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 60 * 1000)
    
    async def close(self):
        """Close session."""
        if self.session and not self.session.closed:
            await self.session.close()


async def main():
    """Test the fetcher."""
    fetcher = BitgetFetcher()
    await fetcher.connect()
    
    try:
        # Test single timeframe
        print("Fetching BTCUSDT 1m data...")
        df_1m = await fetcher.fetch_ohlcv('BTCUSDT', '1m', limit=10)
        print(f"1m data shape: {df_1m.shape}")
        print(df_1m.head())
        
        # Test multiplexed fetch
        print("\nFetching multiplexed timeframes...")
        data = await fetcher.fetch_multiplexed('BTCUSDT', ['1m', '5m', '15m'], limit=5)
        for tf, df in data.items():
            print(f"{tf}: {df.shape[0]} candles")
        
        # Test ticker
        print("\nFetching ticker...")
        ticker = await fetcher.fetch_ticker('BTCUSDT')
        print(ticker)
        
    finally:
        await fetcher.close()


if __name__ == '__main__':
    asyncio.run(main())
