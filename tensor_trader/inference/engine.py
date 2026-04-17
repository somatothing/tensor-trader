"""Inference engine for real-time trading decisions."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from datetime import datetime

from ..features.pipeline import FeaturePipeline
from ..models.boosting.xgboost_model import MarketXGBoost
from ..models.tree.decision_tree import MarketDecisionTree
from ..models.gnn.spread_tensor import SpreadTensorModel

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Trading signal direction."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradingDecision:
    """Trading decision output."""
    direction: SignalDirection
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: datetime
    symbol: str
    timeframe: str
    model_predictions: Dict[str, Any]
    risk_reward_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'direction': self.direction.name,
            'confidence': round(self.confidence, 4),
            'entry_price': round(self.entry_price, 4),
            'stop_loss': round(self.stop_loss, 4),
            'take_profit': round(self.take_profit, 4),
            'position_size': round(self.position_size, 4),
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'model_predictions': self.model_predictions,
            'risk_reward_ratio': round(self.risk_reward_ratio, 2),
        }


class InferenceEngine:
    """Engine for real-time inference and decision making."""
    
    def __init__(self, 
                 models: Dict[str, Any],
                 feature_pipeline: FeaturePipeline,
                 confidence_threshold: float = 0.7,
                 risk_per_trade: float = 0.02,
                 min_risk_reward: float = 1.5):
        """
        Initialize inference engine.
        
        Args:
            models: Dictionary of loaded models {'xgboost': model, 'tree': model, 'spread_tensor': model}
            feature_pipeline: Feature pipeline instance
            confidence_threshold: Minimum confidence to trigger trade
            risk_per_trade: Risk percentage per trade (0.02 = 2%)
            min_risk_reward: Minimum risk/reward ratio
        """
        self.models = models
        self.feature_pipeline = feature_pipeline
        self.confidence_threshold = confidence_threshold
        self.risk_per_trade = risk_per_trade
        self.min_risk_reward = min_risk_reward
        self.decision_history: List[TradingDecision] = []
        
    def compute_features(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """Compute features from OHLCV data."""
        features = self.feature_pipeline.transform(ohlcv_data)
        
        # Extract feature columns (exclude OHLCV)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in features.columns if c not in exclude_cols]
        
        return features[feature_cols].values
    
    def ensemble_predict(self, X: np.ndarray) -> Tuple[SignalDirection, float, Dict[str, Any]]:
        """
        Ensemble prediction using all models.
        
        Returns:
            Tuple of (direction, confidence, model_details)
        """
        predictions = {}
        probabilities = {}
        
        # XGBoost prediction
        if 'xgboost' in self.models:
            try:
                xgb_model = self.models['xgboost']
                xgb_pred = xgb_model.predict(X)
                xgb_proba = xgb_model.predict_proba(X)
                predictions['xgboost'] = int(xgb_pred[0])
                probabilities['xgboost'] = xgb_proba[0].tolist()
            except Exception as e:
                logger.error(f"XGBoost prediction error: {e}")
                predictions['xgboost'] = 0
        
        # Decision Tree prediction
        if 'tree' in self.models:
            try:
                tree_model = self.models['tree']
                tree_pred = tree_model.predict(X)
                tree_proba = tree_model.predict_proba(X)
                predictions['tree'] = int(tree_pred[0])
                probabilities['tree'] = tree_proba[0].tolist()
            except Exception as e:
                logger.error(f"Tree prediction error: {e}")
                predictions['tree'] = 0
        
        # Spread-Tensor prediction
        if 'spread_tensor' in self.models:
            try:
                tensor_model = self.models['spread_tensor']
                tensor_pred = tensor_model.predict(X)
                predictions['spread_tensor'] = int(tensor_pred[0])
            except Exception as e:
                logger.error(f"Spread-Tensor prediction error: {e}")
                predictions['spread_tensor'] = 0
        
        # Ensemble voting with confidence weighting
        if not predictions:
            return SignalDirection.HOLD, 0.0, {}
        
        # Weight by model confidence (XGBoost gets higher weight)
        weights = {'xgboost': 0.5, 'tree': 0.3, 'spread_tensor': 0.2}
        weighted_sum = sum(
            predictions.get(model, 0) * weights.get(model, 0.25)
            for model in predictions.keys()
        )
        
        # Calculate confidence as agreement between models
        pred_values = list(predictions.values())
        agreement = sum(1 for p in pred_values if p == pred_values[0]) / len(pred_values)
        
        # Get max probability from XGBoost as base confidence
        if 'xgboost' in probabilities and probabilities['xgboost']:
            base_confidence = max(probabilities['xgboost'])
        else:
            base_confidence = 0.5
        
        # Final confidence combines agreement and base confidence
        confidence = (agreement * 0.4 + base_confidence * 0.6)
        
        # Determine direction
        if weighted_sum > 0.3:
            direction = SignalDirection.BUY
        elif weighted_sum < -0.3:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD
            confidence = 0.0
        
        model_details = {
            'predictions': predictions,
            'probabilities': probabilities,
            'weighted_sum': round(weighted_sum, 4),
            'agreement': round(agreement, 4),
        }
        
        return direction, confidence, model_details
    
    def calculate_levels(self, 
                         direction: SignalDirection,
                         entry_price: float,
                         atr: float,
                         recent_swing_low: float,
                         recent_swing_high: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        if direction == SignalDirection.BUY:
            # Stop loss below recent swing low or 2x ATR
            stop_loss = min(entry_price - 2 * atr, recent_swing_low * 0.998)
            # Take profit at 2:1 risk-reward minimum
            risk = entry_price - stop_loss
            take_profit = entry_price + risk * self.min_risk_reward
        elif direction == SignalDirection.SELL:
            # Stop loss above recent swing high or 2x ATR
            stop_loss = max(entry_price + 2 * atr, recent_swing_high * 1.002)
            # Take profit at 2:1 risk-reward minimum
            risk = stop_loss - entry_price
            take_profit = entry_price - risk * self.min_risk_reward
        else:
            return entry_price, entry_price
        
        return stop_loss, take_profit
    
    def make_decision(self, 
                      ohlcv_data: pd.DataFrame,
                      symbol: str,
                      timeframe: str,
                      account_balance: float) -> Optional[TradingDecision]:
        """
        Make a trading decision based on current market data.
        
        Args:
            ohlcv_data: Recent OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe string
            account_balance: Current account balance
        
        Returns:
            TradingDecision or None if no trade
        """
        if len(ohlcv_data) < 50:
            logger.warning(f"Insufficient data: {len(ohlcv_data)} candles")
            return None
        
        # Compute features
        try:
            X = self.compute_features(ohlcv_data)
            if X.shape[0] == 0:
                logger.warning("No features computed")
                return None
            
            # Use last row for prediction
            X_last = X[-1:]
        except Exception as e:
            logger.error(f"Feature computation error: {e}")
            return None
        
        # Get ensemble prediction
        direction, confidence, model_details = self.ensemble_predict(X_last)
        
        # Check confidence threshold
        if confidence < self.confidence_threshold or direction == SignalDirection.HOLD:
            logger.info(f"No trade: confidence={confidence:.4f}, direction={direction.name}")
            return None
        
        # Get current price
        entry_price = ohlcv_data['close'].iloc[-1]
        
        # Calculate ATR for position sizing
        high_low = ohlcv_data['high'] - ohlcv_data['low']
        high_close = np.abs(ohlcv_data['high'] - ohlcv_data['close'].shift())
        low_close = np.abs(ohlcv_data['low'] - ohlcv_data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = ranges.max(axis=1).rolling(14).mean().iloc[-1]
        
        # Get recent swing levels
        recent_swing_high = ohlcv_data['high'].rolling(20).max().iloc[-1]
        recent_swing_low = ohlcv_data['low'].rolling(20).min().iloc[-1]
        
        # Calculate SL/TP
        stop_loss, take_profit = self.calculate_levels(
            direction, entry_price, atr, recent_swing_low, recent_swing_high
        )
        
        # Calculate position size based on risk
        risk_amount = account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            logger.warning("Zero price risk, cannot calculate position size")
            return None
        
        position_size = risk_amount / price_risk
        
        # Calculate risk-reward ratio
        potential_profit = abs(take_profit - entry_price)
        potential_loss = abs(entry_price - stop_loss)
        risk_reward = potential_profit / potential_loss if potential_loss > 0 else 0
        
        if risk_reward < self.min_risk_reward:
            logger.info(f"Risk-reward too low: {risk_reward:.2f}")
            return None
        
        decision = TradingDecision(
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            model_predictions=model_details,
            risk_reward_ratio=risk_reward
        )
        
        self.decision_history.append(decision)
        logger.info(f"Trading decision: {direction.name} {symbol} @ {entry_price:.4f} "
                   f"(confidence: {confidence:.4f}, RR: {risk_reward:.2f})")
        
        return decision
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get statistics about decisions."""
        if not self.decision_history:
            return {'total_decisions': 0}
        
        directions = [d.direction.name for d in self.decision_history]
        confidences = [d.confidence for d in self.decision_history]
        
        return {
            'total_decisions': len(self.decision_history),
            'buy_count': directions.count('BUY'),
            'sell_count': directions.count('SELL'),
            'hold_count': directions.count('HOLD'),
            'avg_confidence': np.mean(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
        }


class MultiTimeframeInference:
    """Inference across multiple timeframes with spread-tensor aggregation."""
    
    def __init__(self, 
                 models: Dict[str, Any],
                 feature_pipeline: FeaturePipeline,
                 timeframes: List[str] = None):
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '1d']
        self.engines = {
            tf: InferenceEngine(models, feature_pipeline)
            for tf in self.timeframes
        }
        self.spread_tensor = SpreadTensorModel(input_dim=1)  # Will be set dynamically
        
    def aggregate_decisions(self, 
                           decisions: Dict[str, Optional[TradingDecision]],
                           primary_tf: str = '1m') -> Optional[TradingDecision]:
        """
        Aggregate decisions across timeframes using spread-tensor method.
        
        Higher timeframes have more weight in the final decision.
        """
        weights = {
            '1m': 0.1,
            '5m': 0.15,
            '15m': 0.2,
            '1h': 0.25,
            '4h': 0.2,
            '1d': 0.1,
        }
        
        valid_decisions = {
            tf: d for tf, d in decisions.items() 
            if d is not None and d.direction != SignalDirection.HOLD
        }
        
        if not valid_decisions:
            return None
        
        # Calculate weighted consensus
        weighted_direction = 0
        weighted_confidence = 0
        total_weight = 0
        
        for tf, decision in valid_decisions.items():
            weight = weights.get(tf, 0.1)
            direction_val = 1 if decision.direction == SignalDirection.BUY else -1
            
            weighted_direction += direction_val * decision.confidence * weight
            weighted_confidence += decision.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        normalized_direction = weighted_direction / total_weight
        normalized_confidence = weighted_confidence / total_weight
        
        # Use primary timeframe for price levels
        primary_decision = valid_decisions.get(primary_tf, list(valid_decisions.values())[0])
        
        # Update direction based on consensus
        if normalized_direction > 0.3:
            final_direction = SignalDirection.BUY
        elif normalized_direction < -0.3:
            final_direction = SignalDirection.SELL
        else:
            return None
        
        # Create aggregated decision
        return TradingDecision(
            direction=final_direction,
            confidence=normalized_confidence,
            entry_price=primary_decision.entry_price,
            stop_loss=primary_decision.stop_loss,
            take_profit=primary_decision.take_profit,
            position_size=primary_decision.position_size,
            timestamp=primary_decision.timestamp,
            symbol=primary_decision.symbol,
            timeframe=primary_tf,
            model_predictions={
                'aggregated': True,
                'timeframe_decisions': {tf: d.to_dict() if d else None 
                                       for tf, d in decisions.items()},
                'weighted_direction': round(normalized_direction, 4),
            },
            risk_reward_ratio=primary_decision.risk_reward_ratio
        )
