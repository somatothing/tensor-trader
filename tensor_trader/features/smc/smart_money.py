"""Smart Money Concepts (SMC) feature engineering."""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SwingPoint:
    """Represents a swing high or low."""
    index: int
    timestamp: pd.Timestamp
    price: float
    type: str  # 'high' or 'low'


@dataclass
class FairValueGap:
    """Represents a Fair Value Gap (FVG)."""
    index: int
    timestamp: pd.Timestamp
    top: float
    bottom: float
    type: str  # 'bullish' or 'bearish'
    mitigated: bool = False


@dataclass
class SupplyDemandZone:
    """Represents a Supply or Demand zone."""
    index: int
    timestamp: pd.Timestamp
    top: float
    bottom: float
    type: str  # 'supply' or 'demand'
    active: bool = True


def find_swing_points(df: pd.DataFrame, left_bars: int = 5, 
                      right_bars: int = 5) -> Tuple[List[SwingPoint], List[SwingPoint]]:
    """Find swing highs and lows in price data.
    
    Args:
        df: DataFrame with high, low, close columns
        left_bars: Number of bars to look left
        right_bars: Number of bars to look right
    
    Returns:
        Tuple of (swing_highs, swing_lows)
    """
    swing_highs = []
    swing_lows = []
    
    for i in range(left_bars, len(df) - right_bars):
        # Check for swing high
        is_swing_high = all(
            df['high'].iloc[i] > df['high'].iloc[i - j] 
            for j in range(1, left_bars + 1)
        ) and all(
            df['high'].iloc[i] > df['high'].iloc[i + j]
            for j in range(1, right_bars + 1)
        )
        
        if is_swing_high:
            swing_highs.append(SwingPoint(
                index=i,
                timestamp=df.index[i],
                price=df['high'].iloc[i],
                type='high'
            ))
        
        # Check for swing low
        is_swing_low = all(
            df['low'].iloc[i] < df['low'].iloc[i - j]
            for j in range(1, left_bars + 1)
        ) and all(
            df['low'].iloc[i] < df['low'].iloc[i + j]
            for j in range(1, right_bars + 1)
        )
        
        if is_swing_low:
            swing_lows.append(SwingPoint(
                index=i,
                timestamp=df.index[i],
                price=df['low'].iloc[i],
                type='low'
            ))
    
    return swing_highs, swing_lows


def detect_fvg(df: pd.DataFrame, min_gap: float = 0.0) -> List[FairValueGap]:
    """Detect Fair Value Gaps (FVGs) in price data.
    
    FVG is created when:
    - Bullish: Current candle's low > previous candle's high
    - Bearish: Current candle's high < previous candle's low
    
    Args:
        df: DataFrame with open, high, low, close columns
        min_gap: Minimum gap size to consider
    
    Returns:
        List of FairValueGap objects
    """
    fvgs = []
    
    for i in range(2, len(df)):
        # Bullish FVG: current low > previous high
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
            if gap_size >= min_gap:
                fvgs.append(FairValueGap(
                    index=i,
                    timestamp=df.index[i],
                    top=df['low'].iloc[i],
                    bottom=df['high'].iloc[i-2],
                    type='bullish'
                ))
        
        # Bearish FVG: current high < previous low
        elif df['high'].iloc[i] < df['low'].iloc[i-2]:
            gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
            if gap_size >= min_gap:
                fvgs.append(FairValueGap(
                    index=i,
                    timestamp=df.index[i],
                    top=df['low'].iloc[i-2],
                    bottom=df['high'].iloc[i],
                    type='bearish'
                ))
    
    return fvgs


def detect_bos_choch(df: pd.DataFrame, swing_highs: List[SwingPoint], 
                     swing_lows: List[SwingPoint]) -> Tuple[List[Dict], List[Dict]]:
    """Detect Break of Structure (BOS) and Change of Character (CHOCH).
    
    BOS: Price breaks above previous high (bullish) or below previous low (bearish)
    CHOCH: Price breaks below previous high in uptrend or above previous low in downtrend
    
    Args:
        df: DataFrame with price data
        swing_highs: List of swing high points
        swing_lows: List of swing low points
    
    Returns:
        Tuple of (bos_events, choch_events)
    """
    bos_events = []
    choch_events = []
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return bos_events, choch_events
    
    # Combine and sort swing points
    all_swings = sorted(swing_highs + swing_lows, key=lambda x: x.index)
    
    # Track market structure
    trend = 0  # 1 = uptrend, -1 = downtrend, 0 = undefined
    last_high = None
    last_low = None
    
    for i, swing in enumerate(all_swings):
        if swing.type == 'high':
            if last_high is not None:
                # Check for BOS (break above previous high in uptrend)
                if trend == 1 and swing.price > last_high.price:
                    bos_events.append({
                        'index': swing.index,
                        'timestamp': swing.timestamp,
                        'type': 'bullish_bos',
                        'price': swing.price,
                        'previous_price': last_high.price
                    })
                # Check for CHOCH (break below previous high in downtrend)
                elif trend == -1 and swing.price > last_high.price:
                    choch_events.append({
                        'index': swing.index,
                        'timestamp': swing.timestamp,
                        'type': 'bullish_choch',
                        'price': swing.price,
                        'previous_price': last_high.price
                    })
                    trend = 1  # Trend changed to bullish
            last_high = swing
            
        elif swing.type == 'low':
            if last_low is not None:
                # Check for BOS (break below previous low in downtrend)
                if trend == -1 and swing.price < last_low.price:
                    bos_events.append({
                        'index': swing.index,
                        'timestamp': swing.timestamp,
                        'type': 'bearish_bos',
                        'price': swing.price,
                        'previous_price': last_low.price
                    })
                # Check for CHOCH (break above previous low in uptrend)
                elif trend == 1 and swing.price < last_low.price:
                    choch_events.append({
                        'index': swing.index,
                        'timestamp': swing.timestamp,
                        'type': 'bearish_choch',
                        'price': swing.price,
                        'previous_price': last_low.price
                    })
                    trend = -1  # Trend changed to bearish
            last_low = swing
        
        # Determine initial trend
        if trend == 0 and last_high and last_low:
            if last_high.index > last_low.index and last_high.price > all_swings[0].price:
                trend = 1
            elif last_low.index > last_high.index and last_low.price < all_swings[0].price:
                trend = -1
    
    return bos_events, choch_events


def identify_supply_demand_zones(df: pd.DataFrame, swing_highs: List[SwingPoint],
                                 swing_lows: List[SwingPoint], 
                                 lookback: int = 10) -> Tuple[List[SupplyDemandZone], List[SupplyDemandZone]]:
    """Identify Supply and Demand zones.
    
    Supply zones are created near swing highs with strong selling pressure.
    Demand zones are created near swing lows with strong buying pressure.
    
    Args:
        df: DataFrame with price data
        swing_highs: List of swing high points
        swing_lows: List of swing low points
        lookback: Number of candles to look back for zone formation
    
    Returns:
        Tuple of (supply_zones, demand_zones)
    """
    supply_zones = []
    demand_zones = []
    
    # Identify supply zones (near swing highs)
    for swing in swing_highs:
        if swing.index < lookback:
            continue
        
        # Look for consolidation before the swing high
        start_idx = max(0, swing.index - lookback)
        zone_high = df['high'].iloc[start_idx:swing.index].max()
        zone_low = df['low'].iloc[start_idx:swing.index].min()
        
        # Strong move up followed by rejection
        if zone_high > zone_low * 1.005:  # At least 0.5% range
            supply_zones.append(SupplyDemandZone(
                index=swing.index,
                timestamp=swing.timestamp,
                top=zone_high,
                bottom=zone_low,
                type='supply'
            ))
    
    # Identify demand zones (near swing lows)
    for swing in swing_lows:
        if swing.index < lookback:
            continue
        
        start_idx = max(0, swing.index - lookback)
        zone_high = df['high'].iloc[start_idx:swing.index].max()
        zone_low = df['low'].iloc[start_idx:swing.index].min()
        
        # Strong move down followed by bounce
        if zone_high > zone_low * 1.005:
            demand_zones.append(SupplyDemandZone(
                index=swing.index,
                timestamp=swing.timestamp,
                top=zone_high,
                bottom=zone_low,
                type='demand'
            ))
    
    return supply_zones, demand_zones


def calculate_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all SMC features and add to DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with SMC features added
    """
    result = df.copy()
    
    # Find swing points
    swing_highs, swing_lows = find_swing_points(df)
    
    # Detect FVGs
    fvgs = detect_fvg(df)
    
    # Detect BOS and CHOCH
    bos_events, choch_events = detect_bos_choch(df, swing_highs, swing_lows)
    
    # Identify supply/demand zones
    supply_zones, demand_zones = identify_supply_demand_zones(df, swing_highs, swing_lows)
    
    # Create feature columns
    result['is_swing_high'] = 0
    result['is_swing_low'] = 0
    
    for swing in swing_highs:
        result.loc[result.index[swing.index], 'is_swing_high'] = 1
    for swing in swing_lows:
        result.loc[result.index[swing.index], 'is_swing_low'] = 1
    
    # FVG features
    result['fvg_bullish'] = 0
    result['fvg_bearish'] = 0
    result['fvg_top'] = np.nan
    result['fvg_bottom'] = np.nan
    
    for fvg in fvgs:
        if fvg.type == 'bullish':
            result.loc[result.index[fvg.index], 'fvg_bullish'] = 1
        else:
            result.loc[result.index[fvg.index], 'fvg_bearish'] = 1
        result.loc[result.index[fvg.index], 'fvg_top'] = fvg.top
        result.loc[result.index[fvg.index], 'fvg_bottom'] = fvg.bottom
    
    # BOS/CHOCH features
    result['bos_bullish'] = 0
    result['bos_bearish'] = 0
    result['choch_bullish'] = 0
    result['choch_bearish'] = 0
    
    for event in bos_events:
        col = f'bos_{event["type"].split("_")[0]}'
        if col in result.columns:
            result.loc[result.index[event['index']], col] = 1
    
    for event in choch_events:
        col = f'choch_{event["type"].split("_")[0]}'
        if col in result.columns:
            result.loc[result.index[event['index']], col] = 1
    
    # Market structure
    result['market_structure'] = 0  # 0 = neutral, 1 = bullish, -1 = bearish
    for event in bos_events + choch_events:
        if 'bullish' in event['type']:
            result.loc[result.index[event['index']]:, 'market_structure'] = 1
        else:
            result.loc[result.index[event['index']]:, 'market_structure'] = -1
    
    # Distance to nearest supply/demand zone
    result['dist_to_supply'] = np.nan
    result['dist_to_demand'] = np.nan
    
    for i in range(len(result)):
        current_price = result['close'].iloc[i]
        
        # Find nearest active supply zone
        for zone in supply_zones:
            if zone.index <= i and current_price < zone.bottom:
                dist = (zone.bottom - current_price) / current_price * 100
                if pd.isna(result['dist_to_supply'].iloc[i]) or dist < result['dist_to_supply'].iloc[i]:
                    result.loc[result.index[i], 'dist_to_supply'] = dist
        
        # Find nearest active demand zone
        for zone in demand_zones:
            if zone.index <= i and current_price > zone.top:
                dist = (current_price - zone.top) / current_price * 100
                if pd.isna(result['dist_to_demand'].iloc[i]) or dist < result['dist_to_demand'].iloc[i]:
                    result.loc[result.index[i], 'dist_to_demand'] = dist
    
    return result
