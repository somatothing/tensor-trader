---
name: redis-container-batch-workflow
description: "Plan or implement a localhost Redis and container batch workflow for Tensor Trader without rewriting model code."
argument-hint: "Goal, scope, and constraints for the Redis/container workflow"
agent: agent
---
Related workspace guidance: [Tensor Trader Guidelines](../copilot-instructions.md)

Plan or implement a local Redis and container batch workflow for this Tensor Trader workspace.

Use any extra user-provided arguments as the controlling scope and constraints.

Requirements:
- Review the workspace root and `tensor_trader/` before making changes.
- Keep model architectures and prediction logic untouched unless the user explicitly asks to modify them.
- Use localhost Redis as the default backing service for stateful local storage, cache, queue, batch coordination, or model registry responsibilities.
- Keep Redis and storage logic behind explicit abstractions, preferably in `tensor_trader/data/storage/` or a nearby service module.
- Prefer container-friendly workflows for fetch, feature generation, spread-tensor batches, training, evaluation, and serving.
- Preserve dynamic discovery of checkpoints, metadata, and batch outputs instead of hardcoded paths.
- If the request is planning-only, produce a phased implementation plan with files to touch, validation steps, and rollout order.
- If the request includes implementation, make the smallest infrastructure-scoped change set first and validate it before expanding scope.