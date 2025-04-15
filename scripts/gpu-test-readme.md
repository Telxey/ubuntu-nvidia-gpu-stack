# GPU Testing Toolkit

This toolkit provides utilities for testing and benchmarking NVIDIA GPUs on Ubuntu.

## Overview

The GPU Testing Toolkit includes several Python scripts for:

- Getting detailed GPU information
- Benchmarking GPU performance
- Monitoring GPU usage in real-time
- Stress testing GPU performance and stability

## Installation

The `create-gpu-test-env.sh` script in this directory will set up a complete testing environment at `~/gpu-test`. 

To install:

```bash
sudo ./create-gpu-test-env.sh
```

This script:
1. Creates the `~/gpu-test` directory
2. Sets up a Python virtual environment with PyTorch and other dependencies
3. Installs all testing scripts
4. Configures permissions and makes utilities executable

## Available Tools

After installation, the following tools will be available in `~/gpu-test`:

### 1. GPU Information (`gpu_info.py`)

Shows detailed information about your GPU, including hardware specifications, driver version, CUDA capabilities, memory usage, PCIe configuration, and more.

```bash
cd ~/gpu-test
source venv/bin/activate
python gpu_info.py
```

### 2. GPU Benchmark (`gpu_benchmark.py`)

Measures GPU performance in several areas: matrix multiplication, memory transfer speed, and convolution operations. Generates an HTML report with plots.

```bash
python gpu_benchmark.py --matrix-size 5000 --memory-size 1.0 --batch-size 64 --iterations 5
```

### 3. GPU Monitor (`gpu_monitor.py`)

Provides real-time monitoring of GPU metrics: temperature, utilization, memory usage, and power consumption. Can log data to a CSV file for analysis.

```bash
python gpu_monitor.py --interval 1 --log gpu_stats.csv
```

### 4. GPU Stress Test (`gpu_stress_test.py`)

Runs intensive workloads to test GPU stability: memory allocation, compute operations, mixed workloads, and CUDA kernel operations.

```bash
python gpu_stress_test.py --duration 300 --type all --log-file stress_log.csv
```

### Running All Tests

A convenience script is provided to run all tests in sequence:

```bash
cd ~/gpu-test
./run-all-tests.sh
```

This will collect GPU information, run benchmarks, perform a quick stress test, and save all results to a `reports` directory.

## Documentation

For complete documentation with detailed usage instructions, advanced options, and troubleshooting, see the full guide in the `/docs` directory: `gpu-testing-guide.md`