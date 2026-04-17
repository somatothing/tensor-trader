"""Extended technical indicators for 250+ feature pipeline."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.signal import argrelextrema


def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume-based indicators."""
    result = df.copy()
    
    # Volume SMA/EMA
    for period in [10, 20, 50]:
        result[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        result[f'volume_ema_{period}'] = df['volume'].ewm(span=period, adjust=False).mean()
    
    # Volume Rate of Change
    result['volume_roc'] = df['volume'].pct_change(10) * 100
    
    # On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    result['obv'] = obv
    
    # OBV EMA
    result['obv_ema'] = result['obv'].ewm(span=20, adjust=False).mean()
    
    # Volume-Weighted Average Price (VWAP)
    result['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    result['vwap_deviation'] = (df['close'] - result['vwap']) / result['vwap'] * 100
    
    # Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    money_flow_ratio = (
        raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum() /
        raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    )
    result['mfi'] = 100 - (100 / (1 + money_flow_ratio))
    
    # Chaikin Money Flow (CMF)
    result['cmf'] = (
        ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    ).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Accumulation/Distribution Line (ADL)
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    result['adl'] = money_flow_volume.cumsum()
    
    # Volume Profile (simplified)
    result['volume_profile'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    return result


def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum oscillators and indicators."""
    result = df.copy()
    
    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    result['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    result['stoch_d'] = result['stoch_k'].rolling(window=3).mean()
    result['stoch_slow_d'] = result['stoch_d'].rolling(window=3).mean()
    
    # Williams %R
    result['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
    
    # Commodity Channel Index (CCI)
    tp = (df['high'] + df['low'] + df['close']) / 3
    result['cci'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
    
    # Rate of Change (ROC)
    for period in [10, 20, 50]:
        result[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
    
    # Percentage Price Oscillator (PPO)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    result['ppo'] = (ema_12 - ema_26) / ema_26 * 100
    result['ppo_signal'] = result['ppo'].ewm(span=9, adjust=False).mean()
    result['ppo_histogram'] = result['ppo'] - result['ppo_signal']
    
    # Awesome Oscillator
    median_price = (df['high'] + df['low']) / 2
    ao_fast = median_price.rolling(window=5).mean()
    ao_slow = median_price.rolling(window=34).mean()
    result['awesome_oscillator'] = ao_fast - ao_slow
    
    # Ultimate Oscillator
    bp = df['close'] - df[['close', 'low']].min(axis=1)
    tr = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
    avg_7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
    avg_14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
    avg_28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()
    result['ultimate_oscillator'] = 100 * (4 * avg_7 + 2 * avg_14 + avg_28) / 7
    
    # Relative Vigor Index (RVI)
    rvi_numerator = (df['close'] - df['open']).rolling(window=10).mean()
    rvi_denominator = (df['high'] - df['low']).rolling(window=10).mean()
    result['rvi'] = rvi_numerator / rvi_denominator
    result['rvi_signal'] = result['rvi'].rolling(window=4).mean()
    
    # True Strength Index (TSI)
    pc = df['close'].diff()
    double_smoothed_pc = pc.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    double_smoothed_abs_pc = abs(pc).ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    result['tsi'] = 100 * double_smoothed_pc / double_smoothed_abs_pc
    
    # Detrended Price Oscillator (DPO)
    result['dpo'] = df['close'] - df['close'].shift(21).rolling(window=21).mean()
    
    return result


def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility-based indicators."""
    result = df.copy()
    
    # Keltner Channels
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    atr = result.get('atr', calculate_atr(df))
    result['keltner_middle'] = typical_price.ewm(span=20, adjust=False).mean()
    result['keltner_upper'] = result['keltner_middle'] + 2 * atr
    result['keltner_lower'] = result['keltner_middle'] - 2 * atr
    result['keltner_position'] = (df['close'] - result['keltner_lower']) / (result['keltner_upper'] - result['keltner_lower'])
    
    # Historical Volatility
    for period in [10, 20, 50]:
        result[f'hist_vol_{period}'] = df['close'].pct_change().rolling(window=period).std() * np.sqrt(365) * 100
    
    # Parkinson Volatility
    result['parkinson_vol'] = np.sqrt(
        (np.log(df['high'] / df['low']) ** 2).rolling(window=20).mean() / (4 * np.log(2))
    ) * np.sqrt(365) * 100
    
    # Garman-Klass Volatility
    log_hl = np.log(df['high'] / df['low']) ** 2
    log_co = np.log(df['close'] / df['open']) ** 2
    result['garman_klass_vol'] = np.sqrt(
        0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    ).rolling(window=20).mean() * np.sqrt(365) * 100
    
    # Yang-Zhang Volatility
    log_oc = np.log(df['open'] / df['close'].shift(1)) ** 2
    log_co = np.log(df['close'] / df['open']) ** 2
    log_hl = np.log(df['high'] / df['close']) * np.log(df['high'] / df['open'])
    log_lh = np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
    result['yang_zhang_vol'] = np.sqrt(
        log_oc.rolling(window=20).mean() + 
        0.5 * (log_hl + log_lh).rolling(window=20).mean() +
        0.23 * log_co.rolling(window=20).mean()
    ) * np.sqrt(365) * 100
    
    # Volatility Percentile
    result['vol_percentile'] = result['atr'].rolling(window=100).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1])
    )
    
    # Volatility Regime
    result['vol_regime'] = np.where(
        result['atr'] > result['atr'].rolling(window=50).mean(), 
        1,  # High volatility
        -1  # Low volatility
    )
    
    return result


def calculate_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trend strength and direction indicators."""
    result = df.copy()
    
    # ADX (Average Directional Index)
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff(-1).abs()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * plus_dm.rolling(window=14).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=14).mean() / atr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    result['adx'] = dx.rolling(window=14).mean()
    result['plus_di'] = plus_di
    result['minus_di'] = minus_di
    result['di_diff'] = plus_di - minus_di
    
    # Parabolic SAR
    af = 0.02
    max_af = 0.2
    sar = df['close'].copy()
    ep = df['close'].copy()
    trend = 1  # 1 = uptrend, -1 = downtrend
    
    for i in range(1, len(df)):
        if trend == 1:
            sar.iloc[i] = sar.iloc[i-1] + af * (ep.iloc[i-1] - sar.iloc[i-1])
            if df['low'].iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = df['low'].iloc[i]
                af = 0.02
            else:
                if df['high'].iloc[i] > ep.iloc[i-1]:
                    ep.iloc[i] = df['high'].iloc[i]
                    af = min(af + 0.02, max_af)
        else:
            sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep.iloc[i-1])
            if df['high'].iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = df['high'].iloc[i]
                af = 0.02
            else:
                if df['low'].iloc[i] < ep.iloc[i-1]:
                    ep.iloc[i] = df['low'].iloc[i]
                    af = min(af + 0.02, max_af)
    
    result['parabolic_sar'] = sar
    result['sar_trend'] = trend
    
    # Linear Regression
    for period in [20, 50]:
        x = np.arange(period)
        slope = df['close'].rolling(window=period).apply(
            lambda y: np.polyfit(x[-len(y):], y, 1)[0] if len(y) >= 2 else 0
        )
        intercept = df['close'].rolling(window=period).apply(
            lambda y: np.polyfit(x[-len(y):], y, 1)[1] if len(y) >= 2 else 0
        )
        result[f'linreg_slope_{period}'] = slope
        result[f'linreg_intercept_{period}'] = intercept
        result[f'linreg_value_{period}'] = slope * (period - 1) + intercept
        result[f'linreg_r2_{period}'] = df['close'].rolling(window=period).apply(
            lambda y: np.corrcoef(x[-len(y):], y)[0, 1] ** 2 if len(y) >= 2 else 0
        )
    
    # Trend Intensity Index
    for period in [20, 50]:
        sma = df['close'].rolling(window=period).mean()
        pos_dev = ((df['close'] - sma) / sma * 100).where(df['close'] > sma, 0).rolling(window=period).sum()
        neg_dev = ((sma - df['close']) / sma * 100).where(df['close'] < sma, 0).rolling(window=period).sum()
        result[f'tii_{period}'] = pos_dev / (pos_dev + neg_dev) * 100
    
    # Vortex Indicator
    for period in [14, 20]:
        vm_plus = abs(df['high'] - df['low'].shift(1))
        vm_minus = abs(df['low'] - df['high'].shift(1))
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        result[f'vi_plus_{period}'] = vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()
        result[f'vi_minus_{period}'] = vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()
    
    return result


def calculate_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical and distribution features."""
    result = df.copy()
    
    returns = df['close'].pct_change().dropna()
    
    # Skewness and Kurtosis
    for period in [20, 50, 100]:
        result[f'skewness_{period}'] = returns.rolling(window=period).skew()
        result[f'kurtosis_{period}'] = returns.rolling(window=period).kurt()
    
    # Z-Score
    for period in [20, 50]:
        result[f'zscore_{period}'] = (
            (df['close'] - df['close'].rolling(window=period).mean()) / 
            df['close'].rolling(window=period).std()
        )
    
    # Percentile Rank
    for period in [20, 50, 100]:
        result[f'percentile_{period}'] = df['close'].rolling(window=period).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1])
        )
    
    # Entropy (price distribution)
    for period in [20, 50]:
        def calc_entropy(x):
            hist, _ = np.histogram(x, bins=10, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0
        result[f'entropy_{period}'] = df['close'].rolling(window=period).apply(calc_entropy)
    
    # Fractal Dimension
    for period in [20, 50]:
        def fractal_dim(x):
            if len(x) < 2:
                return 0
            lengths = np.abs(np.diff(x))
            return np.log(np.sum(lengths)) / np.log(len(x))
        result[f'fractal_dim_{period}'] = df['close'].rolling(window=period).apply(fractal_dim)
    
    # Hurst Exponent (simplified)
    for period in [50, 100]:
        def hurst_exponent(x):
            if len(x) < 2:
                return 0.5
            lags = range(2, min(len(x) // 2, 20))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            if len(tau) < 2 or any(t <= 0 for t in tau):
                return 0.5
            return np.polyfit(np.log(lags), np.log(tau), 1)[0]
        result[f'hurst_{period}'] = df['close'].rolling(window=period).apply(hurst_exponent)
    
    return result


def calculate_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market structure and order flow features."""
    result = df.copy()
    
    # Order Flow Imbalance (simplified using price/volume)
    result['buy_volume'] = df['volume'] * (df['close'] > df['open']).astype(int)
    result['sell_volume'] = df['volume'] * (df['close'] < df['open']).astype(int)
    result['volume_imbalance'] = (
        (result['buy_volume'] - result['sell_volume']) / 
        (result['buy_volume'] + result['sell_volume'] + 1e-10)
    )
    
    # Delta (close - open) * volume
    result['delta'] = (df['close'] - df['open']) * df['volume']
    result['delta_cumulative'] = result['delta'].cumsum()
    
    # Cumulative Delta EMA
    result['delta_ema'] = result['delta_cumulative'].ewm(span=20, adjust=False).mean()
    
    # Price Action Score
    result['pa_score'] = (
        (df['close'] > df['open']).astype(int) * 2 +  # Bullish candle
        (df['close'] > df['close'].shift(1)).astype(int) * 1 +  # Higher close
        (df['volume'] > df['volume'].rolling(window=20).mean()).astype(int) * 1  # High volume
    )
    
    # Range Analysis
    result['true_range'] = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    result['range_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Gap Analysis
    result['gap'] = df['open'] - df['close'].shift(1)
    result['gap_pct'] = result['gap'] / df['close'].shift(1) * 100
    result['is_gap_up'] = (result['gap'] > 0).astype(int)
    result['is_gap_down'] = (result['gap'] < 0).astype(int)
    
    # Session High/Low
    for period in [24, 48]:  # Assuming hourly data, 24 = 1 day
        result[f'session_high_{period}'] = df['high'].rolling(window=period).max()
        result[f'session_low_{period}'] = df['low'].rolling(window=period).min()
        result[f'session_range_{period}'] = result[f'session_high_{period}'] - result[f'session_low_{period}']
        result[f'position_in_session_{period}'] = (
            (df['close'] - result[f'session_low_{period}']) / 
            (result[f'session_range_{period}'] + 1e-10)
        )
    
    return result


def calculate_multitimeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features that aggregate multiple timeframes."""
    result = df.copy()
    
    # Multi-timeframe momentum
    for short, long in [(5, 20), (10, 50), (20, 100)]:
        result[f'mtf_momentum_{short}_{long}'] = (
            (df['close'] - df['close'].shift(short)) / df['close'].shift(short) -
            (df['close'] - df['close'].shift(long)) / df['close'].shift(long)
        ) * 100
    
    # Trend Consistency (how many periods price is above/below MA)
    for period in [20, 50]:
        ma = df['close'].rolling(window=period).mean()
        above_ma = (df['close'] > ma).astype(int)
        result[f'trend_consistency_{period}'] = above_ma.rolling(window=period).sum()
    
    # Volatility Contraction/Expansion
    for period in [10, 20]:
        atr = result.get('atr', calculate_atr(df))
        result[f'vol_contraction_{period}'] = (
            atr < atr.rolling(window=period).quantile(0.2)
        ).astype(int)
        result[f'vol_expansion_{period}'] = (
            atr > atr.rolling(window=period).quantile(0.8)
        ).astype(int)
    
    # Time-based features
    if isinstance(df.index, pd.DatetimeIndex):
        result['hour'] = df.index.hour
        result['day_of_week'] = df.index.dayofweek
        result['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    return result


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_all_extended_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all extended indicators (250+ features)."""
    result = df.copy()
    
    # Volume indicators
    result = calculate_volume_indicators(result)
    
    # Momentum indicators
    result = calculate_momentum_indicators(result)
    
    # Volatility indicators
    result = calculate_volatility_indicators(result)
    
    # Trend indicators
    result = calculate_trend_indicators(result)
    
    # Statistical features
    result = calculate_statistical_features(result)
    
    # Market structure features
    result = calculate_market_structure_features(result)
    
    # Multi-timeframe features
    result = calculate_multitimeframe_features(result)
    
    return result
