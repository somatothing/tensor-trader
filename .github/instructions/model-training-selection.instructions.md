---
description: "Use when implementing or changing feature pipelines, training, evaluation, model registry logic, checkpoint discovery, inference, or best-model selection for Tensor Trader."
applyTo:
  - "tensor_trader/features/**"
  - "tensor_trader/training/**"
  - "tensor_trader/models/**"
  - "tensor_trader/inference/**"
  - "tensor_trader/serving/**"
---
# Model Training And Selection

- Optimize for `1m` decision making while treating `5m`, `15m`, `1h`, and `1d` inputs as multiplex context for confirmation, regime detection, and feature enrichment.
- Design feature pipelines so they can scale toward 250+ features while preserving source-column traceability, timeframe provenance, and parity between training and live inference.
- Make leakage checks explicit for lookahead labels, resampling, rolling windows, and cross-timeframe joins.
- Maintain multiple candidate model families when practical, including tree, boosting, GNN, spread-tensor, trader, arbitrage, and DeFi-oriented tracks.
- Persist artifacts with enough metadata to support dynamic discovery and fair comparison: metrics, feature schema, timeframes, training window, and configuration snapshot.
- Prefer dynamic checkpoint and metadata discovery over hardcoded artifact names or fixed winners.
- Select the default model using backtest, PnL, and risk-adjusted trading outcomes first; use classification metrics such as F1, precision, and recall as secondary diagnostics.
- When recommending a winner, state the comparison window, the ranking metrics, and why that artifact beat the alternatives.
- Treat TensorFlow, TensorBoard, spread-tensor batching, and container workflows as preferred extensions, but do not replace the current tree, boosting, GNN, or ONNX paths unless the user asks for that migration.
- Before considering training or inference work complete, verify feature availability, artifact reproducibility, saved metrics, and model-loading behavior end to end.
