# Tensor Trader Guidelines

## Project Focus
- Treat this repository as a Python trading and ML system centered on `tensor_trader/` and the workspace root above it.
- Preserve the current module boundaries in `data/`, `features/`, `training/`, `inference/`, `models/`, `serving/`, and `connectors/` unless the user explicitly asks for a restructure.
- Prefer planning and targeted scaffolding before broad rewrites when work affects multiple subsystems.

## Multi-Timeframe Market Workflow
- Default the market data strategy to maximum reliable OHLCV history collection with safe pagination, exchange limits awareness, and deterministic backfill behavior.
- Treat synchronized `1m`, `5m`, `15m`, `1h`, and `1d` OHLCV streams as the baseline context for model training.
- Optimize decision making for `1m` execution while using the higher timeframes as multiplex inputs for context, regime detection, and confirmation.

## Feature and Tensor Strategy
- Prefer feature engineering that can scale toward 250+ features by combining technical indicators, smart money concepts, price action, spread tensors, and cross-timeframe aggregates.
- Keep feature generation traceable: new features should have clear source columns, timeframe provenance, and compatibility with both training and live inference paths.
- Treat TensorFlow, TensorBoard, spread-tensor batching, and container-oriented tensor workflows as preferred extension points, but do not replace the current tree, boosting, GNN, or ONNX paths unless the user asks for that migration.

## Model Training and Selection
- Maintain multiple candidate model families when possible, including tree, boosting, GNN, spread-tensor, arbitrage, and DeFi-oriented variants.
- Do not hardcode a permanent winner. Persist model artifacts with metadata and metrics so the system can dynamically discover checkpoints and choose the best validated model.
- When proposing model selection logic, surface the exact metrics used, the comparison window, and why the chosen model won.

## Storage and Local Services
- When shared state, caching, queues, or model registry coordination are needed, default to a localhost Redis workflow first.
- Keep Redis and storage logic behind explicit abstractions, preferably in `tensor_trader/data/storage/` or closely related service modules, rather than scattering direct client calls.
- Prefer container-friendly local development flows for services, batches, and supporting infrastructure.

## Implementation Rules
- Keep modules exchange-agnostic where practical so Bitget, Hyperliquid, MT5, and cTrader can share core logic.
- Prefer dynamic file and artifact discovery over hardcoded paths when dealing with checkpoints, metadata, exports, or batch outputs.
- For training or inference work, validate timeframe alignment, leakage risk, feature availability, saved metrics, and reproducible artifact selection before considering the task complete.