# GPU Testing Guide

This guide provides detailed instructions for testing and benchmarking NVIDIA GPUs on Ubuntu using the GPU testing toolkit included in this repository.

## Overview

The GPU testing toolkit provides a comprehensive set of tools to:

- Gather detailed information about your GPU hardware and drivers
- Benchmark GPU performance for various workloads
- Monitor GPU usage, temperature, and other metrics in real-time
- Stress test your GPU to ensure stability

## Installation

1. Install the base NVIDIA GPU stack using our installation script:

   ```bash
   sudo ./scripts/install-nvidia-gpu-stack.sh
   ```

2. Set up the GPU testing environment:

   ```bash
   sudo ./scripts/create-gpu-test-env.sh
   ```

   This script will:
   - Create a directory at `~/gpu-test`
   - Set up a Python virtual environment with PyTorch and other necessary packages
   - Install several Python testing scripts
   - Configure proper permissions

## Testing Tools

### 1. GPU Information (`gpu_info.py`)

This tool provides detailed information about your GPU hardware, drivers, and configuration.

**Usage:**
```bash
cd ~/gpu-test
source venv/bin/activate
python gpu_info.py
```

**Output includes:**
- GPU model, VRAM, and driver version
- CUDA version and capabilities
- Current memory allocation
- PCIe configuration
- Temperature and power settings
- GPU topology (for multi-GPU systems)

### 2. GPU Benchmarking (`gpu_benchmark.py`)

Runs performance benchmarks to measure your GPU's computational capabilities.

**Usage:**
```bash
cd ~/gpu-test
source venv/bin/activate
python gpu_benchmark.py [options]
```

**Available options:**
- `--device DEVICE`: CUDA device to use (default: 'cuda:0')
- `--dtype {float32,float64}`: Data type for benchmarks
- `--matrix-size SIZE`: Size of matrices for multiplication test
- `--memory-size SIZE`: Memory transfer size in GB
- `--batch-size SIZE`: Batch size for convolution test
- `--iterations N`: Number of iterations per test
- `--output FILE`: Path to save benchmark report
- `--skip-matrix`: Skip matrix multiplication benchmark
- `--skip-memory`: Skip memory transfer benchmark
- `--skip-conv`: Skip convolution benchmark

**Example:**
```bash
# Run full benchmark with larger matrices and save report
python gpu_benchmark.py --matrix-size 8000 --iterations 10 --output benchmark_report.html
```

**Benchmark metrics:**
- Matrix multiplication GFLOPS
- Memory transfer bandwidth (host to device and device to host)
- Convolution performance
- Operation latency

### 3. GPU Monitoring (`gpu_monitor.py`)

Provides real-time monitoring of GPU metrics with historical tracking.

**Usage:**
```bash
cd ~/gpu-test
source venv/bin/activate
python gpu_monitor.py [options]
```

**Available options:**
- `--interval SECONDS`: Monitoring interval in seconds (default: 1.0)
- `--log FILE`: Path to save monitoring data as CSV
- `--history N`: Number of historical readings to keep (default: 60)

**Example:**
```bash
# Monitor with 2-second intervals and save logs
python gpu_monitor.py --interval 2 --log gpu_metrics.csv
```

**Monitored metrics:**
- GPU utilization percentage
- Memory utilization
- Temperature
- Power consumption
- Clock speeds
- Historical averages and maximums

Press Ctrl+C to stop monitoring.

### 4. GPU Stress Testing (`gpu_stress_test.py`)

Runs intensive workloads to stress test your GPU for performance and stability.

**Usage:**
```bash
cd ~/gpu-test
source venv/bin/activate
python gpu_stress_test.py [options]
```

**Available options:**
- `--device DEVICE`: CUDA device to use (default: 'cuda:0')
- `--duration SECONDS`: Duration of each test in seconds (default: 60)
- `--type {all,memory,compute,mixed,cuda}`: Type of stress test to run
- `--memory FRACTION`: Fraction of GPU memory to use (0.0-1.0)
- `--log-interval SECONDS`: Interval between logging metrics
- `--log-file FILE`: File to save test metrics

**Example:**
```bash
# Run a comprehensive 5-minute stress test
python gpu_stress_test.py --duration 300 --type all --log-file stress_results.csv
```

**Test types:**
- `memory`: Tests memory allocation and access patterns
- `compute`: Runs intensive matrix operations
- `mixed`: Combines memory and compute workloads
- `cuda`: Tests CUDA kernels with convolutional operations
- `all`: Runs all test types in sequence

### Running All Tests

A convenience script is provided to run all tests in sequence:

```bash
cd ~/gpu-test
./run-all-tests.sh
```

This will:
1. Collect GPU information
2. Run benchmarks
3. Perform a quick stress test
4. Save all results to a `reports` directory

## Interpreting Results

### Benchmark Results

The benchmark tool generates HTML reports with visualizations showing:

- GFLOPS for matrix operations
- Memory bandwidth in GB/s
- Iteration times for each test

Higher values indicate better performance for GFLOPS and bandwidth. Compare your results with published values for your GPU model to determine if it's performing as expected.

### Monitoring Data

The monitoring tool provides insights into:

- Maximum GPU temperature (should generally stay below 85°C)
- Utilization patterns
- Memory usage
- Power consumption relative to TDP

These metrics help identify potential bottlenecks or thermal issues.

### Stress Test Results

A successful stress test indicates your GPU can handle sustained workloads without:

- Crashes or hangs
- Memory errors
- Thermal throttling
- Driver failures

If issues occur during stress testing, check:
- System cooling
- Power supply adequacy
- Driver compatibility
- GPU mounting and connections

## Troubleshooting

### Common Issues

1. **CUDA not available error:**
   ```
   # Verify CUDA installation
   nvcc --version
   
   # Check if PyTorch can see CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Out of memory errors:**
   - Reduce matrix sizes or memory fraction
   - Close other GPU-using applications
   - Restart the system to clear GPU memory

3. **Performance lower than expected:**
   - Check PCIe link speed and width
   - Ensure GPU is properly cooled
   - Verify you're using the latest drivers
   - Check system power management settings

4. **Driver issues:**
   ```bash
   # Reinstall drivers if needed
   sudo apt install --reinstall nvidia-driver-550-server
   ```

## Advanced Usage

### Custom Benchmarks

You can create custom benchmark configurations to test specific workloads:

```bash
# Test only memory performance with larger transfers
python gpu_benchmark.py --skip-matrix --skip-conv --memory-size 4.0 --iterations 20

# Focus on compute with larger matrices
python gpu_benchmark.py --skip-memory --matrix-size 10000 --batch-size 128
```

### Multi-GPU Systems

For systems with multiple GPUs:

```bash
# Benchmark a specific GPU
python gpu_benchmark.py --device cuda:1

# Monitor all GPUs
python gpu_monitor.py

# Stress test a specific GPU
python gpu_stress_test.py --device cuda:1 --type all
```

### Integration with Other Tools

The GPU testing tools can be integrated with system monitoring and logging:

```bash
# Run benchmark and capture system metrics
python gpu_benchmark.py & nvidia-smi dmon -i 0 -s u -d 1 -f system_metrics.log

# Set up scheduled testing
(crontab -l ; echo "0 1 * * * cd ~/gpu-test && ./run-all-tests.sh") | crontab -
```

## Additional Resources

- [NVIDIA Driver Documentation](https://docs.nvidia.com/datacenter/tesla/index.html)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)