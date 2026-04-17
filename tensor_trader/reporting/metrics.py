"""Comprehensive trading metrics and reporting system."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import json
import os


@dataclass
class TradeRecord:
    """Individual trade record."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration_minutes: float = 0.0
    exit_reason: str = ""  # 'tp', 'sl', 'signal', 'manual'
    confidence: float = 0.0
    model_predictions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyPerformance:
    """Daily performance metrics."""
    date: datetime
    starting_equity: float
    ending_equity: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0


@dataclass
class PositionRecord:
    """Open position record."""
    position_id: str
    symbol: str
    side: str
    entry_price: float
    current_price: float
    size: float
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0


class MetricsCalculator:
    """Calculate comprehensive trading metrics."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
        """Calculate annualized Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / periods_per_year
        downside_deviation = np.std(downside_returns)
        
        return excess_returns / downside_deviation * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if max_drawdown == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 365
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """Calculate maximum drawdown and its duration."""
        if len(equity_curve) < 2:
            return 0.0, 0, 0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find peak before max drawdown
        peak_idx = np.argmax(equity_curve[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        
        return max_dd, peak_idx, max_dd_idx
    
    @staticmethod
    def calculate_drawdowns(equity_curve: np.ndarray) -> Dict[str, Any]:
        """Calculate various drawdown statistics."""
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'current_drawdown': 0.0,
                'recovery_time': 0
            }
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        
        max_dd, peak_idx, dd_idx = MetricsCalculator.calculate_max_drawdown(equity_curve)
        
        # Calculate average drawdown (excluding zeros)
        non_zero_dd = drawdown[drawdown < 0]
        avg_dd = np.mean(non_zero_dd) if len(non_zero_dd) > 0 else 0.0
        
        # Current drawdown
        current_dd = drawdown[-1]
        
        # Recovery time (if currently in drawdown)
        recovery_time = 0
        if current_dd < 0:
            last_peak_idx = len(equity_curve) - 1
            while last_peak_idx > 0 and equity_curve[last_peak_idx] < peak[last_peak_idx]:
                last_peak_idx -= 1
            recovery_time = len(equity_curve) - last_peak_idx
        
        return {
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'max_drawdown_duration': dd_idx - peak_idx,
            'current_drawdown': current_dd,
            'recovery_time': recovery_time
        }
    
    @staticmethod
    def calculate_profit_factor(gross_profit: float, gross_loss: float) -> float:
        """Calculate profit factor."""
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 0.0
        return gross_profit / abs(gross_loss)
    
    @staticmethod
    def calculate_expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate mathematical expectancy per trade."""
        loss_rate = 1 - win_rate
        return (win_rate * avg_win) - (loss_rate * abs(avg_loss))
    
    @staticmethod
    def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal Kelly criterion position size."""
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        return max(0, min(kelly, 0.5))  # Cap at 50%
    
    @staticmethod
    def calculate_value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)."""
        if len(returns) < 2:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_conditional_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall."""
        var = MetricsCalculator.calculate_value_at_risk(returns, confidence)
        return np.mean(returns[returns <= var]) if len(returns[returns <= var]) > 0 else var
    
    @staticmethod
    def calculate_beta(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate beta relative to benchmark."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 1.0
        
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    @staticmethod
    def calculate_alpha(returns: np.ndarray, benchmark_returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate alpha (excess return)."""
        beta = MetricsCalculator.calculate_beta(returns, benchmark_returns)
        return np.mean(returns) - (risk_free_rate + beta * (np.mean(benchmark_returns) - risk_free_rate))
    
    @staticmethod
    def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate Information Ratio."""
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns)
        
        return np.mean(active_returns) / tracking_error if tracking_error > 0 else 0.0


class TradeMetricsReporter:
    """Comprehensive trade metrics reporting system."""
    
    def __init__(self, initial_equity: float = 10000.0):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        
        self.trades: List[TradeRecord] = []
        self.positions: Dict[str, PositionRecord] = {}
        self.daily_performance: Dict[str, DailyPerformance] = {}
        self.equity_curve: deque = deque(maxlen=10000)
        
        self.calculator = MetricsCalculator()
        
        # Running statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.total_fees = 0.0
        
        # Time series data
        self.returns_history: deque = deque(maxlen=10000)
        self.trade_durations: deque = deque(maxlen=1000)
        
    def update_equity(self, new_equity: float):
        """Update equity and track peak."""
        self.current_equity = new_equity
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': new_equity
        })
        
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
    
    def add_trade(self, trade: TradeRecord):
        """Add completed trade to history."""
        self.trades.append(trade)
        self.total_trades += 1
        
        if trade.pnl > 0:
            self.winning_trades += 1
            self.gross_profit += trade.pnl
        else:
            self.losing_trades += 1
            self.gross_loss += abs(trade.pnl)
        
        self.total_fees += trade.fees
        self.trade_durations.append(trade.duration_minutes)
        
        # Update daily performance
        date_key = trade.timestamp.strftime('%Y-%m-%d')
        if date_key not in self.daily_performance:
            self.daily_performance[date_key] = DailyPerformance(
                date=trade.timestamp.replace(hour=0, minute=0, second=0, microsecond=0),
                starting_equity=self.current_equity
            )
        
        daily = self.daily_performance[date_key]
        daily.total_trades += 1
        if trade.pnl > 0:
            daily.winning_trades += 1
            daily.gross_profit += trade.pnl
        else:
            daily.losing_trades += 1
            daily.gross_loss += abs(trade.pnl)
        daily.net_pnl += trade.pnl
        daily.ending_equity = self.current_equity
    
    def update_position(self, position: PositionRecord):
        """Update or add open position."""
        self.positions[position.position_id] = position
    
    def close_position(self, position_id: str, exit_price: float, exit_time: datetime):
        """Close a position and record the trade."""
        if position_id not in self.positions:
            return
        
        pos = self.positions[position_id]
        pnl = (exit_price - pos.entry_price) * pos.size if pos.side == 'long' else (pos.entry_price - exit_price) * pos.size
        pnl_pct = pnl / (pos.entry_price * pos.size) * 100 if pos.entry_price > 0 else 0
        
        trade = TradeRecord(
            trade_id=position_id,
            timestamp=exit_time,
            symbol=pos.symbol,
            side='sell' if pos.side == 'long' else 'buy',
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_minutes=(exit_time - pos.entry_time).total_seconds() / 60,
            confidence=pos.confidence
        )
        
        self.add_trade(trade)
        del self.positions[position_id]
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary trading metrics."""
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0
            }
        
        win_rate = self.winning_trades / self.total_trades
        profit_factor = self.calculator.calculate_profit_factor(self.gross_profit, self.gross_loss)
        total_pnl = self.gross_profit - self.gross_loss - self.total_fees
        total_return_pct = (self.current_equity - self.initial_equity) / self.initial_equity * 100
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'total_fees': self.total_fees,
            'total_return_pct': total_return_pct,
            'current_equity': self.current_equity,
            'initial_equity': self.initial_equity
        }
    
    def get_advanced_metrics(self) -> Dict[str, Any]:
        """Get advanced risk-adjusted metrics."""
        if len(self.returns_history) < 2:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0
            }
        
        returns = np.array(list(self.returns_history))
        equity = np.array([e['equity'] for e in self.equity_curve])
        
        sharpe = self.calculator.calculate_sharpe_ratio(returns)
        sortino = self.calculator.calculate_sortino_ratio(returns)
        drawdown_stats = self.calculator.calculate_drawdowns(equity)
        calmar = self.calculator.calculate_calmar_ratio(returns, drawdown_stats['max_drawdown'])
        
        avg_win = self.gross_profit / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.gross_loss / self.losing_trades if self.losing_trades > 0 else 0
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': drawdown_stats['max_drawdown'],
            'avg_drawdown': drawdown_stats['avg_drawdown'],
            'current_drawdown': drawdown_stats['current_drawdown'],
            'expectancy': self.calculator.calculate_expectancy(
                self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                avg_win, avg_loss
            ),
            'kelly_criterion': self.calculator.calculate_kelly_criterion(
                self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                avg_win, avg_loss
            ),
            'var_95': self.calculator.calculate_value_at_risk(returns),
            'cvar_95': self.calculator.calculate_conditional_var(returns)
        }
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get detailed trade statistics."""
        if not self.trades:
            return {}
        
        pnls = [t.pnl for t in self.trades]
        durations = list(self.trade_durations)
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        return {
            'avg_trade_pnl': np.mean(pnls),
            'median_trade_pnl': np.median(pnls),
            'std_trade_pnl': np.std(pnls),
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'avg_duration_minutes': np.mean(durations) if durations else 0,
            'median_duration_minutes': np.median(durations) if durations else 0,
            'consecutive_wins': self._calculate_consecutive('win'),
            'consecutive_losses': self._calculate_consecutive('loss')
        }
    
    def _calculate_consecutive(self, trade_type: str) -> int:
        """Calculate maximum consecutive wins or losses."""
        if not self.trades:
            return 0
        
        max_consecutive = 0
        current = 0
        
        for trade in self.trades:
            is_match = (trade_type == 'win' and trade.pnl > 0) or (trade_type == 'loss' and trade.pnl < 0)
            
            if is_match:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        
        return max_consecutive
    
    def get_timeframe_analysis(self, timeframe: str = '1d') -> Dict[str, Any]:
        """Analyze performance by time of day/week."""
        if not self.trades:
            return {}
        
        hourly_performance = {}
        daily_performance = {}
        
        for trade in self.trades:
            hour = trade.timestamp.hour
            day = trade.timestamp.strftime('%A')
            
            if hour not in hourly_performance:
                hourly_performance[hour] = {'trades': 0, 'pnl': 0.0, 'wins': 0}
            if day not in daily_performance:
                daily_performance[day] = {'trades': 0, 'pnl': 0.0, 'wins': 0}
            
            hourly_performance[hour]['trades'] += 1
            hourly_performance[hour]['pnl'] += trade.pnl
            if trade.pnl > 0:
                hourly_performance[hour]['wins'] += 1
            
            daily_performance[day]['trades'] += 1
            daily_performance[day]['pnl'] += trade.pnl
            if trade.pnl > 0:
                daily_performance[day]['wins'] += 1
        
        # Calculate win rates
        for hour in hourly_performance:
            hourly_performance[hour]['win_rate'] = (
                hourly_performance[hour]['wins'] / hourly_performance[hour]['trades']
                if hourly_performance[hour]['trades'] > 0 else 0
            )
        
        for day in daily_performance:
            daily_performance[day]['win_rate'] = (
                daily_performance[day]['wins'] / daily_performance[day]['trades']
                if daily_performance[day]['trades'] > 0 else 0
            )
        
        return {
            'hourly_performance': hourly_performance,
            'daily_performance': daily_performance,
            'best_hour': max(hourly_performance.items(), key=lambda x: x[1]['pnl'])[0] if hourly_performance else None,
            'best_day': max(daily_performance.items(), key=lambda x: x[1]['pnl'])[0] if daily_performance else None
        }
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive trading report."""
        return {
            'summary': self.get_summary_metrics(),
            'advanced_metrics': self.get_advanced_metrics(),
            'trade_statistics': self.get_trade_statistics(),
            'timeframe_analysis': self.get_timeframe_analysis(),
            'open_positions': len(self.positions),
            'equity_curve': list(self.equity_curve)[-100:],  # Last 100 points
            'timestamp': datetime.now().isoformat()
        }
    
    def export_report(self, filepath: str, format: str = 'json'):
        """Export report to file."""
        report = self.generate_full_report()
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'csv':
            # Export trades to CSV
            if self.trades:
                df = pd.DataFrame([{
                    'trade_id': t.trade_id,
                    'timestamp': t.timestamp,
                    'symbol': t.symbol,
                    'side': t.side,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'size': t.size,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'duration_minutes': t.duration_minutes
                } for t in self.trades])
                df.to_csv(filepath, index=False)
        
        return report


class EquityTracker:
    """Track equity curve and calculate running statistics."""
    
    def __init__(self, initial_equity: float = 10000.0):
        self.initial_equity = initial_equity
        self.equity_points: List[Dict[str, Any]] = []
        self.peak_equity = initial_equity
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
    def add_point(self, equity: float, timestamp: Optional[datetime] = None):
        """Add equity point."""
        point = {
            'timestamp': timestamp or datetime.now(),
            'equity': equity,
            'drawdown': 0.0
        }
        
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        self.current_drawdown = (equity - self.peak_equity) / self.peak_equity
        self.max_drawdown = min(self.max_drawdown, self.current_drawdown)
        point['drawdown'] = self.current_drawdown
        
        self.equity_points.append(point)
    
    def get_returns(self) -> np.ndarray:
        """Calculate returns from equity curve."""
        if len(self.equity_points) < 2:
            return np.array([])
        
        equities = np.array([p['equity'] for p in self.equity_points])
        return np.diff(equities) / equities[:-1]
    
    def get_drawdown_series(self) -> List[float]:
        """Get drawdown series."""
        return [p['drawdown'] for p in self.equity_points]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get equity summary."""
        if not self.equity_points:
            return {}
        
        returns = self.get_returns()
        
        return {
            'initial_equity': self.initial_equity,
            'current_equity': self.equity_points[-1]['equity'],
            'peak_equity': self.peak_equity,
            'total_return_pct': (self.equity_points[-1]['equity'] - self.initial_equity) / self.initial_equity * 100,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'volatility': np.std(returns) * np.sqrt(365) if len(returns) > 0 else 0,
            'points_count': len(self.equity_points)
        }
