# Tensor Trader

A sophisticated algorithmic trading framework leveraging spread-tensor methods, ensemble machine learning, and graph neural networks for multi-timeframe market analysis.

## Features

- **Multi-Exchange Support**: Bitget, Hyperliquid, MT5, cTrader
- **Multiplexed Timeframes**: 1m, 5m, 15m, 1h, 1d
- **Advanced Feature Engineering**:
  - Technical Indicators: RSI, MACD, MA, SMA, Bollinger Bands, SuperTrend, Ichimoku Cloud
  - Smart Money Concepts: BOS, CHOCH, FVG, Supply/Demand zones
  - Price Action Patterns: Fibonacci, Donchian Channels, Crosses
- **ML Model Suite**:
  - Decision Tree Classifiers
  - Gradient Boosting (XGBoost)
  - Graph Neural Networks (PyTorch Geometric)
- **Model Serving**: FastAPI + ONNX export
- **Live Trading Connectors**: Real-time inference with order execution

## Architecture

```
tensor_trader/
├── data/           # Data fetchers and processors
├── features/       # Technical indicators and SMC patterns
├── models/         # ML models (Tree, Boosting, GNN)
├── training/       # Training and hyperparameter optimization
├── connectors/     # Exchange connectors
├── serving/        # FastAPI inference server
└── utils/          # Utilities
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data fetcher
python -m tensor_trader.data.fetchers.bitget_fetcher

# Train models
python -m tensor_trader.training.train

# Start serving
python -m tensor_trader.models.serving.api
```

## License

MIT
