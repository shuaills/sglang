# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About SGLang

SGLang is a fast serving framework for large language models and vision language models with two main components:
- **Backend Runtime (SRT)**: High-performance serving engine with RadixAttention, batching, parallelism, and optimizations
- **Frontend Language**: Programming interface for LLM applications with chained generation, control flow, and multi-modal support

## Repository Structure

- `python/sglang/srt/`: Backend serving runtime - the core inference engine
- `python/sglang/lang/`: Frontend language APIs and backends
- `sgl-kernel/`: Custom CUDA kernels (separate package)
- `sgl-router/`: Load balancing and routing components (Rust)
- `test/`: Unit tests split into `srt/` (backend) and `lang/` (frontend)
- `benchmark/`: Performance benchmarks and evaluation scripts
- `docs/`: Documentation source
- `examples/`: Usage examples and tutorials

## Development Commands

### Installation
```bash
# Install from source for development
pip install -e "python[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Formatting
```bash
# Format modified files only (recommended)
make format

# Run all pre-commit checks
pre-commit run --all-files
```

### Testing

#### Backend Runtime Tests
```bash
cd test/srt

# Run single test file
python3 test_srt_endpoint.py

# Run specific test method
python3 -m unittest test_srt_endpoint.TestSRTEndpoint.test_simple_decode

# Run test suite
python3 run_suite.py --suite per-commit
```

#### Frontend Language Tests
```bash
cd test/lang

# Run single test file
python3 test_srt_backend.py

# Run test suite
python3 run_suite.py --suite per-commit
```

#### Accuracy Tests
```bash
# Quick sanity check with GSM8K
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
```

### Building and Deployment
```bash
# Launch server for testing
python3 -m sglang.launch_server --model <model_name>

# Build documentation (if needed)
cd docs/
# Follow docs/README.md for build instructions
```

## Architecture Overview

### Core Components
- **SRT Engine** (`python/sglang/srt/`): Multi-GPU serving runtime with advanced batching and memory management
- **Model Executor** (`python/sglang/srt/model_executor/`): Model loading and forward pass execution
- **Memory Cache** (`python/sglang/srt/mem_cache/`): RadixAttention prefix caching system
- **Managers** (`python/sglang/srt/managers/`): Request scheduling and batch management
- **Frontend API** (`python/sglang/lang/`): Programming interface and compiler

### Key Design Patterns
- **Hardware Abstraction**: New hardware support added via separate files (e.g., `allocator_ascend.py`)
- **Performance Critical**: All SRT code runs on request critical path - optimize aggressively
- **Pure Functions**: Prefer immutable operations, avoid in-place argument modification
- **Modular Design**: Separate packages for kernels (sgl-kernel) and routing (sgl-router)

## Multi-Package Development

When modifying `sgl-kernel`:
1. Submit PR with kernel changes (without using in main package)
2. Bump sgl-kernel version to trigger PyPI release
3. Update `python/pyproject.toml` to use new version and implement changes

## Code Style Guidelines

- Avoid code duplication (extract shared functions for >5 lines)
- Minimize device synchronization (`tensor.item()`, `tensor.cpu()`)
- Keep files under 2000 lines
- Cache runtime checks in model forward pass as boolean values
- For new features: common path (NVIDIA/existing) should be first branch in conditionals

## Hardware Support

SGLang supports multiple hardware platforms:
- **NVIDIA GPUs**: Primary platform (`srt` extra)
- **AMD GPUs**: ROCm support (`srt_hip` extra)
- **Intel**: XPU and Gaudi support (`srt_xpu`, `srt_hpu` extras)
- **CPU**: CPU-only inference (`srt_cpu` extra)
- **Ascend NPU**: Huawei NPU support (`srt_npu` extra)