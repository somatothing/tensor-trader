"""Base fetcher class for exchange data."""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """Abstract base class for exchange data fetchers."""
    
    TIMEFRAME_MAP = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '1d': '1d',
    }
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 passphrase: Optional[str] = None, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        self.client = None
        self._rate_limit_remaining = 100
        self._rate_limit_reset = 0
        
    @abstractmethod
    async def connect(self) -> bool:
        """Initialize connection to exchange."""
        pass
    
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, 
                        since: Optional[int] = None, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from exchange."""
        pass
    
    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker data."""
        pass
    
    async def fetch_multiplexed(self, symbol: str, timeframes: List[str] = None,
                                since: Optional[int] = None, limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes simultaneously."""
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '1d']
        
        tasks = []
        for tf in timeframes:
            tasks.append(self.fetch_ohlcv(symbol, tf, since, limit))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {tf}: {result}")
                data[tf] = pd.DataFrame()
            else:
                data[tf] = result
        
        return data
    
    def _handle_rate_limit(self, headers: Dict):
        """Update rate limit tracking from response headers."""
        self._rate_limit_remaining = int(headers.get('X-RateLimit-Remaining', 100))
        self._rate_limit_reset = int(headers.get('X-RateLimit-Reset', 0))
    
    async def _wait_for_rate_limit(self):
        """Wait if rate limit is approaching."""
        if self._rate_limit_remaining < 5:
            wait_time = max(0, self._rate_limit_reset - int(datetime.now().timestamp()))
            if wait_time > 0:
                logger.warning(f"Rate limit approaching, waiting {wait_time}s")
                await asyncio.sleep(wait_time + 1)
    
    @abstractmethod
    async def close(self):
        """Close connection to exchange."""
        pass
