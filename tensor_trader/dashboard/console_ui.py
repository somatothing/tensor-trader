"""Console UI with Rich library for real-time dashboard and progress animations."""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.live import Live
from rich.style import Style
from rich.align import Align
from rich import box


@dataclass
class TradeMetrics:
    """Trade metrics data structure."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    
    # Time series data
    equity_curve: deque = field(default_factory=lambda: deque(maxlen=1000))
    pnl_history: deque = field(default_factory=lambda: deque(maxlen=100))
    trade_history: List[Dict] = field(default_factory=list)


@dataclass
class TimeframeData:
    """Data for a single timeframe."""
    timeframe: str
    current_price: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    change_pct: float = 0.0
    rsi: float = 0.0
    signal: str = "NEUTRAL"
    last_update: Optional[datetime] = None


@dataclass
class SystemStatus:
    """System status information."""
    exchange: str = ""
    symbol: str = ""
    is_connected: bool = False
    is_trading: bool = False
    mode: str = "PAPER"
    uptime: str = "00:00:00"
    memory_usage: str = "0 MB"
    cpu_usage: str = "0%"
    last_error: str = ""
    model_status: Dict[str, str] = field(default_factory=dict)


class ConsoleDashboard:
    """Real-time console dashboard with Rich library."""
    
    def __init__(self, refresh_rate: float = 1.0):
        self.console = Console()
        self.refresh_rate = refresh_rate
        self.layout = self._create_layout()
        
        # Data storage
        self.metrics = TradeMetrics()
        self.timeframes: Dict[str, TimeframeData] = {
            "1m": TimeframeData("1m"),
            "5m": TimeframeData("5m"),
            "15m": TimeframeData("15m"),
            "1h": TimeframeData("1h"),
            "1d": TimeframeData("1d"),
        }
        self.status = SystemStatus()
        self.positions: List[Dict] = []
        self.logs: deque = deque(maxlen=50)
        
        # Progress trackers
        self.progress_bars: Dict[str, Progress] = {}
        self.tasks: Dict[str, Any] = {}
        
        # Start time
        self.start_time = datetime.now()
        
    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()
        
        # Header
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Main area
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="center", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Left panel - Timeframes
        layout["left"].split_column(
            Layout(name="timeframes"),
            Layout(name="system_status", size=12)
        )
        
        # Center panel - Charts and positions
        layout["center"].split_column(
            Layout(name="equity_chart", size=15),
            Layout(name="positions")
        )
        
        # Right panel - Metrics and logs
        layout["right"].split_column(
            Layout(name="metrics", size=20),
            Layout(name="logs")
        )
        
        return layout
    
    def _create_header(self) -> Panel:
        """Create header panel."""
        title = Text("🚀 TENSOR TRADER - Real-Time Dashboard", style="bold cyan")
        subtitle = Text(
            f"{self.status.symbol} @ {self.status.exchange} | "
            f"Mode: {self.status.mode} | "
            f"Status: {'🟢 LIVE' if self.status.is_trading else '🔴 STOPPED'}",
            style="dim"
        )
        
        content = Align.center(Text.assemble(title, "\n", subtitle))
        return Panel(content, box=box.DOUBLE, border_style="cyan")
    
    def _create_timeframes_panel(self) -> Panel:
        """Create multiplex timeframe panel."""
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta")
        table.add_column("TF", style="cyan", width=4)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Change", justify="right", width=8)
        table.add_column("RSI", justify="right", width=5)
        table.add_column("Signal", width=8)
        
        for tf_name, tf_data in self.timeframes.items():
            price = f"{tf_data.close:,.2f}" if tf_data.close else "-"
            change = f"{tf_data.change_pct:+.2f}%" if tf_data.change_pct else "-"
            change_style = "green" if tf_data.change_pct > 0 else "red" if tf_data.change_pct < 0 else "white"
            rsi = f"{tf_data.rsi:.1f}" if tf_data.rsi else "-"
            
            signal_style = {
                "BUY": "green",
                "SELL": "red",
                "NEUTRAL": "yellow",
                "HOLD": "white"
            }.get(tf_data.signal, "white")
            
            table.add_row(
                tf_name,
                price,
                Text(change, style=change_style),
                rsi,
                Text(tf_data.signal, style=signal_style)
            )
        
        return Panel(table, title="📊 Multiplex Timeframes", border_style="magenta", box=box.ROUNDED)
    
    def _create_system_status_panel(self) -> Panel:
        """Create system status panel."""
        table = Table(box=None, show_header=False)
        table.add_column("Label", style="dim", width=12)
        table.add_column("Value", style="cyan")
        
        uptime = self._calculate_uptime()
        connection = "🟢 Connected" if self.status.is_connected else "🔴 Disconnected"
        
        table.add_row("Exchange:", self.status.exchange or "-")
        table.add_row("Symbol:", self.status.symbol or "-")
        table.add_row("Uptime:", uptime)
        table.add_row("Connection:", connection)
        table.add_row("Memory:", self.status.memory_usage)
        table.add_row("CPU:", self.status.cpu_usage)
        
        if self.status.last_error:
            table.add_row("Last Error:", Text(self.status.last_error, style="red"))
        
        return Panel(table, title="⚙️ System Status", border_style="blue", box=box.ROUNDED)
    
    def _create_equity_chart(self) -> Panel:
        """Create equity curve visualization."""
        if not self.metrics.equity_curve:
            return Panel("No equity data available", title="📈 Equity Curve", border_style="green")
        
        # Create ASCII chart
        equity_list = list(self.metrics.equity_curve)
        if len(equity_list) < 2:
            return Panel("Collecting data...", title="📈 Equity Curve", border_style="green")
        
        min_eq = min(equity_list)
        max_eq = max(equity_list)
        range_eq = max_eq - min_eq if max_eq != min_eq else 1
        
        # Create sparkline
        width = 60
        height = 8
        
        chart_lines = []
        for i in range(height):
            level = max_eq - (i * range_eq / height)
            line = ""
            for j, val in enumerate(equity_list[-width:]):
                if val >= level:
                    line += "█"
                else:
                    line += " "
            chart_lines.append(line)
        
        chart = "\n".join(chart_lines)
        
        # Add stats
        stats = (
            f"Current: ${equity_list[-1]:,.2f} | "
            f"Peak: ${max_eq:,.2f} | "
            f"Drawdown: {self.metrics.current_drawdown:.2f}%"
        )
        
        content = f"{chart}\n{stats}"
        return Panel(content, title="📈 Equity Curve", border_style="green", box=box.ROUNDED)
    
    def _create_positions_panel(self) -> Panel:
        """Create open positions panel."""
        table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold yellow")
        table.add_column("Symbol", width=10)
        table.add_column("Side", width=6)
        table.add_column("Size", justify="right", width=10)
        table.add_column("Entry", justify="right", width=10)
        table.add_column("Current", justify="right", width=10)
        table.add_column("PnL", justify="right", width=10)
        table.add_column("PnL %", justify="right", width=8)
        
        if not self.positions:
            table.add_row("-", "-", "-", "-", "-", "-", "-")
        else:
            for pos in self.positions:
                pnl = pos.get('unrealized_pnl', 0)
                pnl_pct = pos.get('pnl_pct', 0)
                pnl_style = "green" if pnl >= 0 else "red"
                
                table.add_row(
                    pos.get('symbol', '-'),
                    pos.get('side', '-'),
                    f"{pos.get('size', 0):.4f}",
                    f"{pos.get('entry_price', 0):,.2f}",
                    f"{pos.get('current_price', 0):,.2f}",
                    Text(f"${pnl:,.2f}", style=pnl_style),
                    Text(f"{pnl_pct:+.2f}%", style=pnl_style)
                )
        
        return Panel(table, title="📋 Open Positions", border_style="yellow", box=box.ROUNDED)
    
    def _create_metrics_panel(self) -> Panel:
        """Create trading metrics panel."""
        table = Table(box=None, show_header=False)
        table.add_column("Metric", style="dim", width=15)
        table.add_column("Value", justify="right")
        
        m = self.metrics
        
        # Color coding
        def colored_value(value, threshold=0, inverse=False):
            if inverse:
                color = "green" if value <= threshold else "red"
            else:
                color = "green" if value >= threshold else "red"
            return Text(f"{value:.2f}", style=color)
        
        table.add_row("Total Trades:", f"{m.total_trades}")
        table.add_row("Win Rate:", f"{m.win_rate:.1f}%")
        table.add_row("Profit Factor:", f"{m.profit_factor:.2f}")
        table.add_row("Sharpe Ratio:", f"{m.sharpe_ratio:.2f}")
        table.add_row("Max Drawdown:", f"{m.max_drawdown:.2f}%")
        table.add_row("Current DD:", f"{m.current_drawdown:.2f}%")
        table.add_row("Avg Win:", f"${m.avg_win:.2f}")
        table.add_row("Avg Loss:", f"${m.avg_loss:.2f}")
        table.add_row("Total PnL:", Text(f"${m.total_pnl:,.2f}", 
                                         style="green" if m.total_pnl >= 0 else "red"))
        
        return Panel(table, title="💰 Trading Metrics", border_style="green", box=box.ROUNDED)
    
    def _create_logs_panel(self) -> Panel:
        """Create system logs panel."""
        if not self.logs:
            content = "No logs yet..."
        else:
            content = "\n".join(list(self.logs)[-15:])
        
        return Panel(content, title="📝 System Logs", border_style="dim", box=box.ROUNDED)
    
    def _create_footer(self) -> Panel:
        """Create footer panel."""
        text = Text(
            "Press Ctrl+C to exit | Tensor Trader v1.0 | Real-time Trading Dashboard",
            style="dim",
            justify="center"
        )
        return Panel(text, box=box.SIMPLE)
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime."""
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def update(self):
        """Update all dashboard components."""
        self.layout["header"].update(self._create_header())
        self.layout["left"]["timeframes"].update(self._create_timeframes_panel())
        self.layout["left"]["system_status"].update(self._create_system_status_panel())
        self.layout["center"]["equity_chart"].update(self._create_equity_chart())
        self.layout["center"]["positions"].update(self._create_positions_panel())
        self.layout["right"]["metrics"].update(self._create_metrics_panel())
        self.layout["right"]["logs"].update(self._create_logs_panel())
        self.layout["footer"].update(self._create_footer())
    
    def update_timeframe(self, timeframe: str, data: Dict[str, Any]):
        """Update timeframe data."""
        if timeframe in self.timeframes:
            tf = self.timeframes[timeframe]
            tf.current_price = data.get('price', tf.current_price)
            tf.open = data.get('open', tf.open)
            tf.high = data.get('high', tf.high)
            tf.low = data.get('low', tf.low)
            tf.close = data.get('close', tf.close)
            tf.volume = data.get('volume', tf.volume)
            tf.change_pct = data.get('change_pct', tf.change_pct)
            tf.rsi = data.get('rsi', tf.rsi)
            tf.signal = data.get('signal', tf.signal)
            tf.last_update = datetime.now()
    
    def update_metrics(self, **kwargs):
        """Update trading metrics."""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
    
    def update_equity(self, equity: float):
        """Update equity curve."""
        self.metrics.equity_curve.append(equity)
    
    def add_trade(self, trade: Dict):
        """Add a trade to history."""
        self.metrics.trade_history.append(trade)
        self.metrics.total_trades += 1
        
        pnl = trade.get('pnl', 0)
        if pnl > 0:
            self.metrics.winning_trades += 1
            self.metrics.avg_win = (self.metrics.avg_win * (self.metrics.winning_trades - 1) + pnl) / self.metrics.winning_trades
        elif pnl < 0:
            self.metrics.losing_trades += 1
            self.metrics.avg_loss = (self.metrics.avg_loss * (self.metrics.losing_trades - 1) + abs(pnl)) / self.metrics.losing_trades
        
        # Update derived metrics
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades * 100
        
        if self.metrics.avg_loss > 0:
            self.metrics.profit_factor = (self.metrics.avg_win * self.metrics.winning_trades) / (self.metrics.avg_loss * self.metrics.losing_trades)
        
        self.metrics.total_pnl += pnl
        self.metrics.pnl_history.append(pnl)
    
    def update_positions(self, positions: List[Dict]):
        """Update open positions."""
        self.positions = positions
    
    def update_status(self, **kwargs):
        """Update system status."""
        for key, value in kwargs.items():
            if hasattr(self.status, key):
                setattr(self.status, key, value)
    
    def log(self, message: str, level: str = "INFO"):
        """Add log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_colors = {
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "SUCCESS": "green"
        }
        color = level_colors.get(level, "white")
        self.logs.append(f"[{timestamp}] [{level}] {message}")
    
    def create_progress(self, name: str, total: int, description: str = "") -> Progress:
        """Create a progress bar for long-running operations."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
        self.progress_bars[name] = progress
        self.tasks[name] = progress.add_task(description or name, total=total)
        return progress
    
    def update_progress(self, name: str, advance: float = 1):
        """Update progress bar."""
        if name in self.progress_bars and name in self.tasks:
            self.progress_bars[name].update(self.tasks[name], advance=advance)
    
    def complete_progress(self, name: str):
        """Mark progress as complete."""
        if name in self.progress_bars and name in self.tasks:
            self.progress_bars[name].update(self.tasks[name], completed=True)
    
    async def run(self):
        """Run the dashboard in live mode."""
        with Live(self.layout, console=self.console, refresh_per_second=4) as live:
            try:
                while True:
                    self.update()
                    await asyncio.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive trading report."""
        m = self.metrics
        
        report = {
            "summary": {
                "total_trades": m.total_trades,
                "winning_trades": m.winning_trades,
                "losing_trades": m.losing_trades,
                "win_rate": m.win_rate,
                "total_pnl": m.total_pnl,
                "profit_factor": m.profit_factor,
                "sharpe_ratio": m.sharpe_ratio,
                "max_drawdown": m.max_drawdown,
            },
            "timeframes": {
                tf_name: {
                    "price": tf_data.close,
                    "change_pct": tf_data.change_pct,
                    "signal": tf_data.signal
                }
                for tf_name, tf_data in self.timeframes.items()
            },
            "positions": self.positions,
            "system_status": {
                "uptime": self._calculate_uptime(),
                "is_connected": self.status.is_connected,
                "is_trading": self.status.is_trading
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return report


class ProgressAnimation:
    """Helper class for console progress animations."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
    
    def training_progress(self, epochs: int, current: int, loss: float, val_loss: float):
        """Display training progress."""
        progress = (current / epochs) * 100
        bar_width = 30
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        self.console.print(
            f"\r[blue]Training:[/blue] [{bar}] {progress:.1f}% | "
            f"Epoch {current}/{epochs} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f}",
            end=""
        )
    
    def data_fetch_progress(self, symbol: str, timeframe: str, fetched: int, total: int):
        """Display data fetching progress."""
        progress = (fetched / total) * 100 if total > 0 else 0
        self.console.print(
            f"[cyan]Fetching {symbol} {timeframe}:[/cyan] {fetched}/{total} candles ({progress:.1f}%)"
        )
    
    def feature_calculation_progress(self, current: int, total: int, feature_name: str):
        """Display feature calculation progress."""
        progress = (current / total) * 100
        self.console.print(
            f"[magenta]Features:[/magenta] {current}/{total} | Current: {feature_name} ({progress:.1f}%)"
        )
    
    def inference_progress(self, symbol: str, confidence: float, signal: str):
        """Display inference progress."""
        color = "green" if signal == "BUY" else "red" if signal == "SELL" else "yellow"
        self.console.print(
            f"[bold {color}]Signal:[/bold {color}] {signal} | Confidence: {confidence:.2%} | {symbol}"
        )
    
    def trading_loop_status(self, iteration: int, next_update: float):
        """Display trading loop status."""
        self.console.print(
            f"[green]Trading Loop:[/green] Iteration {iteration} | Next update in {next_update:.1f}s"
        )


# Convenience functions
def create_dashboard(refresh_rate: float = 1.0) -> ConsoleDashboard:
    """Create a new dashboard instance."""
    return ConsoleDashboard(refresh_rate=refresh_rate)


def create_progress_animator(console: Optional[Console] = None) -> ProgressAnimation:
    """Create a new progress animator."""
    return ProgressAnimation(console)
