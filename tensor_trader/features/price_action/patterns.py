"""Price action pattern detection."""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Pattern:
    """Represents a price action pattern."""
    index: int
    timestamp: pd.Timestamp
    type: str
    direction: str  # 'bullish' or 'bearish'
    confidence: float
    metadata: Dict = None


def detect_crosses(df: pd.DataFrame, fast_col: str = 'sma_7', 
                   slow_col: str = 'sma_20') -> pd.DataFrame:
    """Detect moving average crosses.
    
    Args:
        df: DataFrame with moving average columns
        fast_col: Fast moving average column name
        slow_col: Slow moving average column name
    
    Returns:
        DataFrame with cross signals added
    """
    result = df.copy()
    
    # Calculate cross conditions
    fast_above_slow = (df[fast_col] > df[slow_col]).astype(bool)
    fast_above_slow_prev = fast_above_slow.shift(1).fillna(False).astype(bool)
    
    # Golden cross: fast crosses above slow (bullish)
    result['golden_cross'] = ((fast_above_slow) & (~fast_above_slow_prev)).astype(int)
    
    # Death cross: fast crosses below slow (bearish)
    result['death_cross'] = ((~fast_above_slow) & (fast_above_slow_prev)).astype(int)
    
    # Distance between MAs
    result['ma_distance'] = ((df[fast_col] - df[slow_col]) / df[slow_col] * 100)
    
    # MA alignment
    result['ma_aligned_bullish'] = (
        (df['sma_7'] > df['sma_20']) & 
        (df['sma_20'] > df['sma_50'])
    ).astype(int)
    
    result['ma_aligned_bearish'] = (
        (df['sma_7'] < df['sma_20']) & 
        (df['sma_20'] < df['sma_50'])
    ).astype(int)
    
    return result


def detect_waves(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Detect Elliott Wave-like patterns (simplified).
    
    Identifies impulse and corrective waves based on price swings.
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period for wave detection
    
    Returns:
        DataFrame with wave features added
    """
    result = df.copy()
    
    # Calculate price swings
    result['price_swing_high'] = result['high'].rolling(window=period, center=True).max() == result['high']
    result['price_swing_low'] = result['low'].rolling(window=period, center=True).min() == result['low']
    
    # Wave amplitude
    result['wave_amplitude'] = result['high'].rolling(window=period).max() - result['low'].rolling(window=period).min()
    result['wave_amplitude_pct'] = result['wave_amplitude'] / result['close'] * 100
    
    # Wave momentum
    result['wave_momentum'] = result['close'].diff(period)
    result['wave_momentum_pct'] = result['wave_momentum'] / result['close'].shift(period) * 100
    
    # Wave phase
    result['wave_phase'] = 0
    result.loc[result['price_swing_high'], 'wave_phase'] = 1  # Peak
    result.loc[result['price_swing_low'], 'wave_phase'] = -1  # Trough
    
    # Wave trend strength
    result['wave_trend_strength'] = abs(result['wave_momentum_pct']) / result['wave_amplitude_pct']
    
    return result


def calculate_fibonacci_levels(high: float, low: float, trend: str = 'up') -> Dict[str, float]:
    """Calculate Fibonacci retracement/extension levels.
    
    Args:
        high: High price of the swing
        low: Low price of the swing
        trend: 'up' for uptrend, 'down' for downtrend
    
    Returns:
        Dictionary of Fibonacci levels
    """
    diff = high - low
    
    if trend == 'up':
        levels = {
            '0.0': low,
            '0.236': low + diff * 0.236,
            '0.382': low + diff * 0.382,
            '0.5': low + diff * 0.5,
            '0.618': low + diff * 0.618,
            '0.786': low + diff * 0.786,
            '1.0': high,
            '1.272': high + diff * 0.272,
            '1.618': high + diff * 0.618,
            '2.0': high + diff,
        }
    else:
        levels = {
            '0.0': high,
            '0.236': high - diff * 0.236,
            '0.382': high - diff * 0.382,
            '0.5': high - diff * 0.5,
            '0.618': high - diff * 0.618,
            '0.786': high - diff * 0.786,
            '1.0': low,
            '1.272': low - diff * 0.272,
            '1.618': low - diff * 0.618,
            '2.0': low - diff,
        }
    
    return levels


def detect_fibonacci_signals(df: pd.DataFrame, swing_period: int = 20) -> pd.DataFrame:
    """Detect price reactions at Fibonacci levels.
    
    Args:
        df: DataFrame with OHLCV data
        swing_period: Period for identifying swings
    
    Returns:
        DataFrame with Fibonacci features added
    """
    result = df.copy()
    
    # Find recent swing high and low
    result['swing_high'] = result['high'].rolling(window=swing_period, min_periods=1).max()
    result['swing_low'] = result['low'].rolling(window=swing_period, min_periods=1).min()
    
    # Calculate Fibonacci levels for each row
    result['fib_0'] = result['swing_low']
    result['fib_236'] = result['swing_low'] + (result['swing_high'] - result['swing_low']) * 0.236
    result['fib_382'] = result['swing_low'] + (result['swing_high'] - result['swing_low']) * 0.382
    result['fib_500'] = result['swing_low'] + (result['swing_high'] - result['swing_low']) * 0.5
    result['fib_618'] = result['swing_low'] + (result['swing_high'] - result['swing_low']) * 0.618
    result['fib_786'] = result['swing_low'] + (result['swing_high'] - result['swing_low']) * 0.786
    result['fib_1000'] = result['swing_high']
    
    # Distance to nearest Fibonacci level
    fib_cols = ['fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_1000']
    result['dist_to_fib'] = result[fib_cols].apply(
        lambda row: min(abs(row - result.loc[row.name, 'close'])), axis=1
    )
    result['dist_to_fib_pct'] = result['dist_to_fib'] / result['close'] * 100
    
    # Price at key Fibonacci level (within 1%)
    tolerance = result['close'] * 0.01
    result['at_fib_618'] = abs(result['close'] - result['fib_618']) <= tolerance
    result['at_fib_382'] = abs(result['close'] - result['fib_382']) <= tolerance
    result['at_fib_500'] = abs(result['close'] - result['fib_500']) <= tolerance
    
    return result


def detect_long_short_boxes(df: pd.DataFrame, consolidation_periods: int = 10,
                           breakout_threshold: float = 0.02) -> pd.DataFrame:
    """Detect consolidation boxes and breakouts.
    
    Args:
        df: DataFrame with OHLCV data
        consolidation_periods: Number of periods for box formation
        breakout_threshold: Percentage threshold for breakout confirmation
    
    Returns:
        DataFrame with box features added
    """
    result = df.copy()
    
    # Calculate box ranges
    result['box_high'] = result['high'].rolling(window=consolidation_periods).max()
    result['box_low'] = result['low'].rolling(window=consolidation_periods).min()
    result['box_range'] = result['box_high'] - result['box_low']
    result['box_range_pct'] = result['box_range'] / result['close'] * 100
    
    # Box consolidation (tight range)
    result['is_consolidating'] = result['box_range_pct'] < 2.0  # Less than 2% range
    
    # Long box breakout (bullish)
    result['long_box_breakout'] = (
        (result['close'] > result['box_high'].shift(1) * (1 + breakout_threshold)) &
        result['is_consolidating'].shift(1)
    ).astype(int)
    
    # Short box breakdown (bearish)
    result['short_box_breakdown'] = (
        (result['close'] < result['box_low'].shift(1) * (1 - breakout_threshold)) &
        result['is_consolidating'].shift(1)
    ).astype(int)
    
    # Box position
    result['box_position'] = (result['close'] - result['box_low']) / result['box_range']
    result['box_position'] = result['box_position'].fillna(0.5)
    
    return result


def detect_bullish_bearish_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish and bearish candlestick patterns.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with pattern signals added
    """
    result = df.copy()
    
    # Calculate candle components
    result['body'] = result['close'] - result['open']
    result['body_pct'] = abs(result['body']) / result['close'] * 100
    result['upper_wick'] = result['high'] - result[['open', 'close']].max(axis=1)
    result['lower_wick'] = result[['open', 'close']].min(axis=1) - result['low']
    result['total_range'] = result['high'] - result['low']
    
    # Doji (small body)
    result['is_doji'] = result['body_pct'] < 0.5
    
    # Hammer (bullish reversal) - small body at top, long lower wick
    result['is_hammer'] = (
        (result['body'] > 0) &  # Bullish candle
        (result['lower_wick'] > result['body'].abs() * 2) &  # Long lower wick
        (result['upper_wick'] < result['body'].abs() * 0.5)  # Small upper wick
    ).astype(int)
    
    # Shooting star (bearish reversal) - small body at bottom, long upper wick
    result['is_shooting_star'] = (
        (result['body'] < 0) &  # Bearish candle
        (result['upper_wick'] > result['body'].abs() * 2) &  # Long upper wick
        (result['lower_wick'] < result['body'].abs() * 0.5)  # Small lower wick
    ).astype(int)
    
    # Engulfing patterns
    result['prev_body'] = result['body'].shift(1)
    
    # Bullish engulfing
    result['bullish_engulfing'] = (
        (result['body'] > 0) &  # Current bullish
        (result['prev_body'] < 0) &  # Previous bearish
        (result['open'] < result['close'].shift(1)) &  # Open below prev close
        (result['close'] > result['open'].shift(1))  # Close above prev open
    ).astype(int)
    
    # Bearish engulfing
    result['bearish_engulfing'] = (
        (result['body'] < 0) &  # Current bearish
        (result['prev_body'] > 0) &  # Previous bullish
        (result['open'] > result['close'].shift(1)) &  # Open above prev close
        (result['close'] < result['open'].shift(1))  # Close below prev open
    ).astype(int)
    
    # Morning star (bullish)
    result['morning_star'] = (
        (result['body'].shift(2) < 0) &  # First bearish
        (result['is_doji'].shift(1)) &  # Second doji
        (result['body'] > 0) &  # Third bullish
        (result['close'] > (result['open'].shift(2) + result['close'].shift(2)) / 2)
    ).astype(int)
    
    # Evening star (bearish)
    result['evening_star'] = (
        (result['body'].shift(2) > 0) &  # First bullish
        (result['is_doji'].shift(1)) &  # Second doji
        (result['body'] < 0) &  # Third bearish
        (result['close'] < (result['open'].shift(2) + result['close'].shift(2)) / 2)
    ).astype(int)
    
    # Marubozu (strong trend candle with no wicks)
    result['bullish_marubozu'] = (
        (result['body'] > 0) &
        (result['upper_wick'] < result['body'] * 0.05) &
        (result['lower_wick'] < result['body'] * 0.05)
    ).astype(int)
    
    result['bearish_marubozu'] = (
        (result['body'] < 0) &
        (result['upper_wick'] < abs(result['body']) * 0.05) &
        (result['lower_wick'] < abs(result['body']) * 0.05)
    ).astype(int)
    
    # Overall bullish/bearish signals
    result['bullish_signal'] = (
        result['is_hammer'] | 
        result['bullish_engulfing'] | 
        result['morning_star'] | 
        result['bullish_marubozu']
    ).astype(int)
    
    result['bearish_signal'] = (
        result['is_shooting_star'] | 
        result['bearish_engulfing'] | 
        result['evening_star'] | 
        result['bearish_marubozu']
    ).astype(int)
    
    return result


def calculate_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all price action features.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with all price action features added
    """
    result = df.copy()
    
    # Moving average crosses
    if 'sma_7' in result.columns and 'sma_20' in result.columns:
        result = detect_crosses(result)
    
    # Wave patterns
    result = detect_waves(result)
    
    # Fibonacci levels
    result = detect_fibonacci_signals(result)
    
    # Box patterns
    result = detect_long_short_boxes(result)
    
    # Candlestick patterns
    result = detect_bullish_bearish_patterns(result)
    
    # Price momentum
    result['momentum_1'] = result['close'].pct_change(1)
    result['momentum_3'] = result['close'].pct_change(3)
    result['momentum_5'] = result['close'].pct_change(5)
    result['momentum_10'] = result['close'].pct_change(10)
    
    # Price vs moving averages
    for col in result.columns:
        if col.startswith('sma_') or col.startswith('ema_'):
            result[f'price_vs_{col}'] = (result['close'] - result[col]) / result[col] * 100
    
    return result
