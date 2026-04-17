---
description: "Use when implementing or changing OHLCV fetchers, exchange pagination, candle synchronization, data ingestion, backfills, or storage workflows for Tensor Trader."
applyTo:
  - "tensor_trader/data/**"
  - "tensor_trader/connectors/**"
---
# Multi-Timeframe Data Workflow

- Start from exchange-aware fetchers and keep shared logic exchange-agnostic where practical.
- Prefer maximum reliable OHLCV history collection over short sample fetches.
- Treat synchronized `1m`, `5m`, `15m`, `1h`, and `1d` OHLCV streams as the default training context.
- Use safe pagination and deterministic backfill loops with explicit exchange-limit awareness, bounded retries, and resume-friendly checkpoints.
- Normalize timestamps, symbol naming, and candle schemas before downstream feature or label generation.
- Separate raw fetch, normalization, multiplex assembly, and persistence so higher layers do not absorb exchange-specific quirks.
- When fetch jobs need durable coordination, cache, deduplication, or restart-safe progress, use localhost Redis as the default backing service.
- Keep storage and stateful coordination behind abstractions in `tensor_trader/data/storage/` or a closely related service layer.
- Validate data completeness with missing-interval checks, expected-candle counts, and reproducible backfill windows.
