# Tensor-Trader Scaffold Plan

## Goal
Scaffold a multi-exchange, multi-timeframe trading system ("Tensor-Trader") utilizing advanced ML (GNN, Gradient Boosting, Decision Trees) and technical analysis, with model serving and ONNX export for live inference on Bitget, Hyperliquid, MT5, and cTrader.

## Research Summary
- **APIs**: Bitget (SDK: `bitget-python`), Hyperliquid (SDK: `hyperliquid-python-sdk`).
- **ML Stack**: XGBoost/LightGBM for Gradient Boosting, PyTorch Geometric (PyG) for Graph Neural Networks (GNN), Scikit-learn for Decision Trees.
- **Inference**: ONNX Runtime for cross-platform model execution (MT5 supports ONNX natively).
- **Technical Indicators**: `pandas-ta` or `talib` for RSI, MACD, etc.
- **Data**: Multi-timeframe (1m to 1d) multiplexing required for 1m decision making.

## Approach
1. **Infrastructure**: Modular Python project with GitHub Actions for CI/CD.
2. **Data Engine**: Multiplexed data fetcher for Bitget/Hyperliquid across 1m, 5m, 15m, 1h, 1d.
3. **Feature Engineering**: Comprehensive technical analysis suite (SMC, Fibonacci, Ichimoku, etc.).
4. **Model Suite**: 
   - Decision Tree/Gradient Boosting for baseline predictions.
   - GNN for spatial-temporal relationship between assets/timeframes.
   - "Spread-Tensor" method: Representing market state as a high-dimensional tensor of spreads and features.
5. **Serving**: FastAPI-based model server and ONNX export pipeline.
6. **Execution**: Integration layers for MT5, cTrader, Bitget, and Hyperliquid.

## Subtasks
1. **Project Setup**: Initialize Git, directory structure, and GitHub Actions workflows. (Output: `/Users/somatothing/Desktop/devs/repo_name/boards/.github/workflows/build.yml`)
2. **Data Layer**: Implement Bitget/Hyperliquid multiplexed fetcher for 1m, 5m, 15m, 1h, 1d. (Output: `src/data/fetcher.py`)
3. **Feature Engineering**: Implement technical indicators (RSI, MACD, SMC, Fibonacci, Ichimoku). (Output: `src/features/indicators.py`)
4. **Model Architecture**: Implement Decision Tree, Gradient Boosting, and GNN modules. (Output: `src/models/architectures.py`)
5. **Training Pipeline**: Implement training loop with dynamic hyperparameter macro (entropy-based). (Output: `src/models/train.py`)
6. **Export & Serving**: Implement ONNX export and FastAPI serving layer. (Output: `src/serving/onnx_exporter.py`, `src/serving/api.py`)
7. **Live Inference**: Implement execution connectors for MT5, cTrader, Bitget, Hyperliquid. (Output: `src/execution/connectors.py`)

## Deliverables
| File Path | Description |
|-----------|-------------|
| `src/data/fetcher.py` | Multi-exchange data ingestion |
| `src/features/indicators.py` | Technical analysis & SMC features |
| `src/models/architectures.py` | GNN, XGBoost, and Tree models |
| `src/serving/onnx_exporter.py` | Model export to ONNX |
| `src/execution/connectors.py` | Live trading execution logic |

## Evaluation Criteria
- Successful data fetch from at least one exchange.
- Successful feature generation for all requested indicators.
- Model training completion with saved ONNX artifact.
- End-to-end scaffold structure verified by CI.

## Notes
- Ichimoku Cloud division-by-zero protection is critical.
- MT5/cTrader often require specific bridge or DLL logic for Python; ONNX is the primary bridge.
