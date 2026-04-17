"""Hyperliquid exchange data fetcher."""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import pandas as pd

from .base_fetcher import BaseFetcher

logger = logging.getLogger(__name__)


class HyperliquidFetcher(BaseFetcher):
    """Hyperliquid API data fetcher."""
    
    BASE_URL = "https://api.hyperliquid.xyz"
    BASE_URL_TESTNET = "https://api.hyperliquid-testnet.xyz"
    
    TIMEFRAME_MAP = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 wallet_address: Optional[str] = None, testnet: bool = False):
        super().__init__(api_key, api_secret, None, testnet)
        self.wallet_address = wallet_address
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
    
    async def _make_request(self, method: str, endpoint: str, payload: Dict = None) -> Dict:
        """Make request to Hyperliquid API."""
        await self._wait_for_rate_limit()
        
        if self.session is None:
            await self.connect()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, json=payload) as response:
                self._handle_rate_limit(dict(response.headers))
                
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Hyperliquid API error {response.status}: {text}")
                
                return await response.json()
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m',
                         since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candles from Hyperliquid.
        
        Args:
            symbol: Trading coin (e.g., 'BTC')
            timeframe: Candle timeframe
            since: Start timestamp in milliseconds
            limit: Number of candles to fetch
        """
        tf = self.TIMEFRAME_MAP.get(timeframe, timeframe)
        
        # Hyperliquid uses a custom format for candles
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": tf,
                "startTime": since if since else int((datetime.now().timestamp() - limit * self._timeframe_to_seconds(tf)) * 1000),
                "endTime": int(datetime.now().timestamp() * 1000)
            }
        }
        
        try:
            data = await self._make_request('POST', '/info', payload)
            
            if not data or not isinstance(data, list):
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            candles = []
            for candle in data:
                if isinstance(candle, dict):
                    candles.append({
                        'timestamp': int(candle.get('t', candle.get('time', 0))),
                        'open': float(candle.get('o', candle.get('open', 0))),
                        'high': float(candle.get('h', candle.get('high', 0))),
                        'low': float(candle.get('l', candle.get('low', 0))),
                        'close': float(candle.get('c', candle.get('close', 0))),
                        'volume': float(candle.get('v', candle.get('volume', 0))),
                    })
                elif isinstance(candle, list) and len(candle) >= 6:
                    candles.append({
                        'timestamp': int(candle[0]),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
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
        payload = {"type": "allMids"}
        try:
            data = await self._make_request('POST', '/info', payload)
            if isinstance(data, dict) and symbol in data:
                return {
                    'symbol': symbol,
                    'markPrice': float(data[symbol]),
                }
            return {}
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}
    
    async def fetch_orderbook(self, symbol: str) -> Dict:
        """Fetch order book (L2)."""
        payload = {
            "type": "l2Book",
            "coin": symbol
        }
        try:
            return await self._make_request('POST', '/info', payload)
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {}
    
    async def fetch_meta(self) -> Dict:
        """Fetch exchange metadata including available coins."""
        payload = {"type": "meta"}
        try:
            return await self._make_request('POST', '/info', payload)
        except Exception as e:
            logger.error(f"Error fetching meta: {e}")
            return {}
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe to seconds."""
        multipliers = {
            'm': 60,
            'h': 60 * 60,
            'd': 24 * 60 * 60,
        }
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 60)
    
    async def close(self):
        """Close session."""
        if self.session and not self.session.closed:
            await self.session.close()


async def main():
    """Test the fetcher."""
    fetcher = HyperliquidFetcher()
    await fetcher.connect()
    
    try:
        # Test single timeframe
        print("Fetching BTC 1m data...")
        df_1m = await fetcher.fetch_ohlcv('BTC', '1m', limit=10)
        print(f"1m data shape: {df_1m.shape}")
        print(df_1m.head())
        
        # Test multiplexed fetch
        print("\nFetching multiplexed timeframes...")
        data = await fetcher.fetch_multiplexed('BTC', ['1m', '5m', '15m'], limit=5)
        for tf, df in data.items():
            print(f"{tf}: {df.shape[0]} candles")
        
        # Test ticker
        print("\nFetching ticker...")
        ticker = await fetcher.fetch_ticker('BTC')
        print(ticker)
        
        # Test meta
        print("\nFetching meta...")
        meta = await fetcher.fetch_meta()
        print(f"Available coins: {len(meta.get('universe', []))}")
        
    finally:
        await fetcher.close()


if __name__ == '__main__':
    asyncio.run(main())
