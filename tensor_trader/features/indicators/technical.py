"""Technical indicators for feature engineering."""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Close price series
        period: RSI period (default 14)
    
    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Close price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                              std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.
    
    Args:
        prices: Close price series
        period: Moving average period
        std_dev: Standard deviation multiplier
    
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate SuperTrend indicator.
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period
        multiplier: ATR multiplier
    
    Returns:
        DataFrame with supertrend, direction, upper_band, lower_band columns
    """
    hl2 = (df['high'] + df['low']) / 2
    
    # Calculate ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate bands
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize supertrend
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(period, len(df)):
        if i == period:
            if df['close'].iloc[i] <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1  # Downtrend
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1   # Uptrend
        else:
            if supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                if df['close'].iloc[i] > upper_band.iloc[i]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
            else:
                if df['close'].iloc[i] < lower_band.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
    
    result = df.copy()
    result['supertrend'] = supertrend
    result['supertrend_direction'] = direction
    result['supertrend_upper'] = upper_band
    result['supertrend_lower'] = lower_band
    return result


def calculate_ichimoku(df: pd.DataFrame, tenkan_period: int = 9, 
                       kijun_period: int = 26, senkou_b_period: int = 52) -> pd.DataFrame:
    """Calculate Ichimoku Cloud indicator.
    
    Args:
        df: DataFrame with high, low, close columns
        tenkan_period: Tenkan-sen period
        kijun_period: Kijun-sen period
        senkou_b_period: Senkou Span B period
    
    Returns:
        DataFrame with ichimoku components
    """
    # Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 for 9 periods
    tenkan_high = df['high'].rolling(window=tenkan_period).max()
    tenkan_low = df['low'].rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 for 26 periods
    kijun_high = df['high'].rolling(window=kijun_period).max()
    kijun_low = df['low'].rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods forward
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 for 52 periods, shifted 26 periods forward
    senkou_b_high = df['high'].rolling(window=senkou_b_period).max()
    senkou_b_low = df['low'].rolling(window=senkou_b_period).min()
    senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span): Close price shifted 26 periods backward
    chikou_span = df['close'].shift(-kijun_period)
    
    result = df.copy()
    result['tenkan_sen'] = tenkan_sen
    result['kijun_sen'] = kijun_sen
    result['senkou_span_a'] = senkou_span_a
    result['senkou_span_b'] = senkou_span_b
    result['chikou_span'] = chikou_span
    
    # Cloud color: 1 = bullish (A above B), -1 = bearish (B above A)
    result['ichimoku_cloud_color'] = np.where(
        result['senkou_span_a'] > result['senkou_span_b'], 1, -1
    )
    
    # Price relative to cloud
    result['price_above_cloud'] = np.where(
        df['close'] > result[['senkou_span_a', 'senkou_span_b']].max(axis=1), 1,
        np.where(df['close'] < result[['senkou_span_a', 'senkou_span_b']].min(axis=1), -1, 0)
    )
    
    # TK Cross: Tenkan above Kijun = bullish
    result['tk_cross'] = np.where(tenkan_sen > kijun_sen, 1, -1)
    
    return result


def calculate_donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Calculate Donchian Channels.
    
    Args:
        df: DataFrame with high, low, close columns
        period: Lookback period
    
    Returns:
        DataFrame with donchian_upper, donchian_lower, donchian_middle
    """
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    
    result = df.copy()
    result['donchian_upper'] = upper
    result['donchian_lower'] = lower
    result['donchian_middle'] = middle
    return result


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period
    
    Returns:
        ATR series
    """
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with all indicators added
    """
    result = df.copy()
    
    # RSI
    result['rsi'] = calculate_rsi(df['close'])
    result['rsi_7'] = calculate_rsi(df['close'], 7)
    result['rsi_21'] = calculate_rsi(df['close'], 21)
    
    # MACD
    macd, signal, hist = calculate_macd(df['close'])
    result['macd'] = macd
    result['macd_signal'] = signal
    result['macd_histogram'] = hist
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'])
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    result['bb_width'] = (bb_upper - bb_lower) / bb_middle
    result['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Moving Averages
    for period in [7, 20, 50, 200]:
        result[f'sma_{period}'] = calculate_sma(df['close'], period)
        result[f'ema_{period}'] = calculate_ema(df['close'], period)
    
    # SuperTrend
    result = calculate_supertrend(result)
    
    # Ichimoku
    result = calculate_ichimoku(result)
    
    # Donchian Channels
    result = calculate_donchian_channels(result)
    
    # ATR
    result['atr'] = calculate_atr(df)
    result['atr_percent'] = result['atr'] / df['close'] * 100
    
    # Price-based features
    result['returns'] = df['close'].pct_change()
    result['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    result['volatility'] = result['returns'].rolling(window=20).std()
    
    # Candlestick features
    result['body'] = abs(df['close'] - df['open'])
    result['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    result['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    result['candle_range'] = df['high'] - df['low']
    
    return result
