# Roadmap: Inference, Optimization, and Agent Capabilities

This document outlines a prioritized roadmap to expand the repository's inference, optimization, and agent capabilities. The roadmap is organized into phases with suggested milestones and expected outcomes.

Phase 1 — Foundations (0–4 weeks)
- Add micro-benchmarks and profiling scripts
  - Scripts to measure latency, memory, and throughput for key pipelines (base SDXL generation, refiner, ControlNet).
  - Use torch.profiler where possible and simple timers otherwise.
  - Add reproducible test configs (small/medium/large) and commit baseline results under `docs/benchmarks/`.
- Add a CPU-only CI smoke workflow
  - GitHub Actions workflow that runs `pytest` and a quick smoke benchmark on push and PR.
- ONNX export experiments
  - Try exporting smaller model components (e.g., VAE or parts of UNet) to ONNX and run verification on CPU with ONNX Runtime.

Phase 2 — Inference runtime integration (4–12 weeks)
- Integrate ONNX Runtime and TensorRT where applicable
  - Convert suitable model parts to ONNX, and run them via ONNX Runtime with CUDA/TensorRT execution providers.
  - Optionally add TensorRT engine building for NVIDIA hardware and measure speedups.
- Quantization & compression pipelines
  - Add post-training quantization flows (ONNX/ORT static quantization or PyTorch quantization workflows).
  - Evaluate accuracy/performance tradeoffs and provide reproducible recipes.
- Add a minimal local LLM path for agent workflows
  - Add a small LLM inference path (e.g., llama.cpp or vLLM) and a toy agent that composes prompts and calls the image generation API.

Phase 3 — Systems & low-level optimizations (12–24 weeks)
- Prototype C++/CUDA optimizations for hot paths
  - Use profiling data to identify bottlenecks and prototype PyTorch C++ extensions or CUDA kernels for critical ops.
- Triton/TensorRT production experiments
  - Integrate with Triton Inference Server or build TensorRT custom plugins for specialized operations.
- Automated performance regression testing
  - Add nightly profiling snapshots and performance regression checks; store baseline artifacts for comparison.

Phase 4 — Agentic systems and demos (ongoing)
- Agent orchestration layer
  - Build a lightweight agent framework that composes LLMs and perception models for multi-step workflows (planning, tool use, memory, evaluation).
  - Provide a GRPC/REST adapter to allow agents to call local image/vision services.
- On-device demo bundles
  - Produce quantized model bundles + minimal runtimes for running demos on RTX-enabled edge devices. Provide scripts to build the bundles and run the demos.

Quick experiments to prioritize (pick 1–2 for immediate impact)
- ONNX export + ORT CPU run for a UNet/VAE subgraph; report latency and memory vs PyTorch baseline.
- Tiny agent demo: a local llama.cpp-based planner that calls `/api/generate` and iterates on control weights.
- Benchmarks: add `bench.py` that runs generation at 256/512/1024 resolutions and records latency and peak memory.

How this helps
- Demonstrates practical experience with inference runtimes (ORT, TensorRT), model compression (quantization), and end-to-end profiling — all valuable for edge and GPU-accelerated deployments.
- Provides concrete artifacts (benchmarks, export scripts, demos) that show optimization impact and system-level thinking.

If you want I can start with Phase 1: add `docs/benchmarks/bench.py`, a CPU-only GitHub Actions `ci.yml`, and an initial ONNX export script for a model component. Which one should I begin with?