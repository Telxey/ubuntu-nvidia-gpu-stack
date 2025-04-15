#!/bin/bash
# GPU Testing Environment Setup Script
# Creates ~/gpu-test directory with Python scripts for GPU testing
# Author: Ubuntu-NVIDIA-GPU-Stack Team

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function for logging
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Check if nvidia-smi is available
check_gpu() {
    log "Checking for NVIDIA GPU..."
    if ! command -v nvidia-smi &> /dev/null; then
        error "NVIDIA drivers not installed or not in PATH"
    fi
    
    if ! nvidia-smi &> /dev/null; then
        error "NVIDIA GPU not detected or drivers not functioning properly"
    fi
    
    success "NVIDIA GPU detected and drivers working"
}

# Create gpu-test directory in user's home
create_test_directory() {
    local USER_HOME=$(eval echo ~$SUDO_USER)
    local TEST_DIR="${USER_HOME}/gpu-test"
    
    log "Creating GPU test directory at ${TEST_DIR}..."
    
    # Create directory with correct ownership
    mkdir -p "${TEST_DIR}"
    chown -R $SUDO_USER:$SUDO_USER "${TEST_DIR}"
    
    success "Test directory created at ${TEST_DIR}"
    
    # Return the test directory path for later use
    echo "${TEST_DIR}"
}

# Create Python environment
setup_python_env() {
    local TEST_DIR=$1
    log "Setting up Python environment in ${TEST_DIR}..."
    
    # Create virtual environment
    sudo -u $SUDO_USER python3 -m venv "${TEST_DIR}/venv"
    
    # Install packages
    sudo -u $SUDO_USER "${TEST_DIR}/venv/bin/pip" install --upgrade pip
    sudo -u $SUDO_USER "${TEST_DIR}/venv/bin/pip" install numpy torch torchvision matplotlib pandas scikit-learn jupyterlab
    
    success "Python environment created successfully"
}

# Create GPU test scripts
create_test_scripts() {
    local TEST_DIR=$1
    log "Creating GPU test scripts..."
    
    # Basic GPU info script
    cat > "${TEST_DIR}/gpu_info.py" << 'EOF'
#!/usr/bin/env python3
"""
Basic GPU Information Script
Displays detailed information about available NVIDIA GPUs
"""

import torch
import platform
import os
import subprocess
import json
from datetime import datetime

def run_command(cmd):
    """Run shell command and return output"""
    try:
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
        return output.strip()
    except subprocess.CalledProcessError:
        return "Command failed"

def get_nvidia_smi_info():
    """Get detailed GPU info using nvidia-smi"""
    try:
        # Get GPU info in JSON format
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.current.sm,clocks.max.sm,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader",
            shell=True, universal_newlines=True
        )
        
        # Parse the output
        gpus = []
        for i, line in enumerate(output.strip().split('\n')):
            values = [val.strip() for val in line.split(',')]
            
            if len(values) >= 14:
                gpu = {
                    "index": i,
                    "name": values[0],
                    "driver_version": values[1],
                    "memory_total": values[2],
                    "memory_free": values[3],
                    "memory_used": values[4],
                    "temperature": values[5],
                    "gpu_utilization": values[6],
                    "memory_utilization": values[7],
                    "power_draw": values[8],
                    "power_limit": values[9],
                    "sm_clock": values[10],
                    "max_sm_clock": values[11],
                    "pcie_gen": values[12],
                    "pcie_width": values[13]
                }
                gpus.append(gpu)
                
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def main():
    print(f"=== GPU Information Report ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hostname: {platform.node()}")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    
    # PyTorch info
    print(f"\n=== PyTorch Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"GPU count: {device_count}")
        
        for i in range(device_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    
    # Get detailed GPU information from nvidia-smi
    print(f"\n=== NVIDIA System Management Interface Information ===")
    gpus = get_nvidia_smi_info()
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"\nGPU {i}: {gpu['name']}")
            print(f"  Driver Version: {gpu['driver_version']}")
            print(f"  Memory: {gpu['memory_used']} / {gpu['memory_total']} (Free: {gpu['memory_free']})")
            print(f"  Temperature: {gpu['temperature']}")
            print(f"  Utilization: GPU {gpu['gpu_utilization']}, Memory {gpu['memory_utilization']}")
            print(f"  Power: {gpu['power_draw']} / {gpu['power_limit']}")
            print(f"  SM Clock: {gpu['sm_clock']} (Max: {gpu['max_sm_clock']})")
            print(f"  PCIe: Gen {gpu['pcie_gen']}, Width x{gpu['pcie_width']}")
    else:
        print("No detailed GPU information available from nvidia-smi")
    
    # System GPU topology
    print(f"\n=== GPU Topology ===")
    try:
        topo = run_command("nvidia-smi topo -m")
        print(topo)
    except:
        print("GPU topology information not available")
        
    # NVML Library Info
    print(f"\n=== NVML Library Information ===")
    try:
        nvml_version = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1")
        print(f"NVML Version: {nvml_version}")
    except:
        print("NVML information not available")
    
    print("\n=== Report Complete ===")

if __name__ == "__main__":
    main()
EOF

    # GPU benchmark script
    cat > "${TEST_DIR}/gpu_benchmark.py" << 'EOF'
#!/usr/bin/env python3
"""
GPU Benchmark Script
Runs several standard tests to measure GPU performance
"""

import torch
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class GPUBenchmark:
    def __init__(self, device='cuda:0', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.results = {}
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot run GPU benchmark.")
        
        print(f"Running benchmarks on {torch.cuda.get_device_name(device)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
    def matrix_multiplication(self, size=5000, iterations=5):
        """Benchmark matrix multiplication"""
        print(f"\nRunning matrix multiplication benchmark ({size}x{size}, {iterations} iterations)...")
        
        # Generate random matrices
        torch.cuda.empty_cache()
        a = torch.randn(size, size, dtype=self.dtype, device=self.device)
        b = torch.randn(size, size, dtype=self.dtype, device=self.device)
        
        # Warmup
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for i in range(iterations):
            start = time.time()
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            print(f"  Iteration {i+1}/{iterations}: {times[-1]:.4f} seconds")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        gflops = 2 * size**3 / (avg_time * 1e9)
        
        print(f"Matrix multiplication: {avg_time:.4f} ± {std_time:.4f} seconds, {gflops:.2f} GFLOPS")
        
        self.results['matrix_mul'] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'gflops': gflops,
            'times': times
        }
        
    def memory_benchmark(self, size_gb=1, iterations=3):
        """Benchmark memory transfer speed"""
        print(f"\nRunning memory transfer benchmark ({size_gb}GB, {iterations} iterations)...")
        
        # Calculate size in elements
        bytes_per_element = 4 if self.dtype == torch.float32 else 8  # float32 or float64
        elements = int(size_gb * 1024**3 / bytes_per_element)
        
        h2d_times = []
        d2h_times = []
        
        for i in range(iterations):
            # Host to device
            torch.cuda.empty_cache()
            x_cpu = torch.randn(elements, dtype=self.dtype)
            
            start = time.time()
            x_gpu = x_cpu.to(self.device)
            torch.cuda.synchronize()
            h2d_time = time.time() - start
            h2d_times.append(h2d_time)
            
            # Device to host
            start = time.time()
            _ = x_gpu.cpu()
            torch.cuda.synchronize()
            d2h_time = time.time() - start
            d2h_times.append(d2h_time)
            
            # Calculate bandwidth
            h2d_bandwidth = size_gb / h2d_time
            d2h_bandwidth = size_gb / d2h_time
            
            print(f"  Iteration {i+1}/{iterations}:")
            print(f"    Host to Device: {h2d_time:.4f} seconds, {h2d_bandwidth:.2f} GB/s")
            print(f"    Device to Host: {d2h_time:.4f} seconds, {d2h_bandwidth:.2f} GB/s")
            
            # Free memory
            del x_cpu, x_gpu
        
        avg_h2d = np.mean(h2d_times)
        avg_d2h = np.mean(d2h_times)
        h2d_bandwidth = size_gb / avg_h2d
        d2h_bandwidth = size_gb / avg_d2h
        
        print(f"Memory transfer:")
        print(f"  Host to Device: {avg_h2d:.4f} seconds, {h2d_bandwidth:.2f} GB/s")
        print(f"  Device to Host: {avg_d2h:.4f} seconds, {d2h_bandwidth:.2f} GB/s")
        
        self.results['memory'] = {
            'h2d_avg_time': avg_h2d,
            'h2d_bandwidth': h2d_bandwidth,
            'h2d_times': h2d_times,
            'd2h_avg_time': avg_d2h,
            'd2h_bandwidth': d2h_bandwidth,
            'd2h_times': d2h_times
        }
        
    def conv_benchmark(self, batch_size=64, iterations=5):
        """Benchmark convolutional operations"""
        print(f"\nRunning convolution benchmark (batch size {batch_size}, {iterations} iterations)...")
        
        # Create a typical CNN input tensor: [batch, channels, height, width]
        torch.cuda.empty_cache()
        input_tensor = torch.randn(batch_size, 3, 224, 224, dtype=self.dtype, device=self.device)
        
        # Create a convolutional layer
        conv_layer = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1).to(device=self.device, dtype=self.dtype)
        
        # Warmup
        _ = conv_layer(input_tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for i in range(iterations):
            start = time.time()
            _ = conv_layer(input_tensor)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
            print(f"  Iteration {i+1}/{iterations}: {times[-1]:.4f} seconds")
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate operations (approximate)
        # For each output pixel: kernel_h * kernel_w * in_channels multiplications and additions
        kernel_size = 3 * 3
        in_channels = 3
        out_channels = 64
        output_size = 224 * 224  # Assuming same padding
        ops_per_image = 2 * kernel_size * in_channels * out_channels * output_size
        total_ops = ops_per_image * batch_size
        gflops = total_ops / (avg_time * 1e9)
        
        print(f"Convolution: {avg_time:.4f} ± {std_time:.4f} seconds, {gflops:.2f} GFLOPS")
        
        self.results['conv'] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'gflops': gflops,
            'times': times
        }
    
    def generate_report(self, save_path=None):
        """Generate a report with benchmark results"""
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"gpu_benchmark_report_{timestamp}.html"
            
        # Create plots
        plt.figure(figsize=(15, 10))
        
        # Matrix multiplication results
        if 'matrix_mul' in self.results:
            plt.subplot(2, 2, 1)
            plt.bar(['Matrix Multiplication'], [self.results['matrix_mul']['gflops']])
            plt.ylabel('GFLOPS')
            plt.title('Matrix Multiplication Performance')
            
        # Memory transfer results
        if 'memory' in self.results:
            plt.subplot(2, 2, 2)
            plt.bar(['Host to Device', 'Device to Host'], 
                    [self.results['memory']['h2d_bandwidth'], self.results['memory']['d2h_bandwidth']])
            plt.ylabel('GB/s')
            plt.title('Memory Transfer Bandwidth')
            
        # Convolution results
        if 'conv' in self.results:
            plt.subplot(2, 2, 3)
            plt.bar(['Convolution'], [self.results['conv']['gflops']])
            plt.ylabel('GFLOPS')
            plt.title('Convolution Performance')
            
        # Combined iteration times
        plt.subplot(2, 2, 4)
        
        if 'matrix_mul' in self.results:
            plt.plot(self.results['matrix_mul']['times'], label='Matrix Mul')
            
        if 'conv' in self.results:
            plt.plot(self.results['conv']['times'], label='Convolution')
            
        if 'memory' in self.results:
            plt.plot(self.results['memory']['h2d_times'], label='Host to Device')
            plt.plot(self.results['memory']['d2h_times'], label='Device to Host')
            
        plt.xlabel('Iteration')
        plt.ylabel('Time (s)')
        plt.title('Benchmark Iteration Times')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path.replace('.html', '.png'))
        print(f"Saved benchmark plot to {save_path.replace('.html', '.png')}")
        
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GPU Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .plot {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GPU Benchmark Report</h1>
                <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Device: {torch.cuda.get_device_name(self.device)}</p>
                <p>CUDA Version: {torch.version.cuda}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Benchmark</th>
                        <th>Result</th>
                    </tr>
        """
        
        if 'matrix_mul' in self.results:
            html += f"""
                    <tr>
                        <td>Matrix Multiplication</td>
                        <td>{self.results['matrix_mul']['gflops']:.2f} GFLOPS</td>
                    </tr>
            """
            
        if 'memory' in self.results:
            html += f"""
                    <tr>
                        <td>Memory (Host to Device)</td>
                        <td>{self.results['memory']['h2d_bandwidth']:.2f} GB/s</td>
                    </tr>
                    <tr>
                        <td>Memory (Device to Host)</td>
                        <td>{self.results['memory']['d2h_bandwidth']:.2f} GB/s</td>
                    </tr>
            """
            
        if 'conv' in self.results:
            html += f"""
                    <tr>
                        <td>Convolution</td>
                        <td>{self.results['conv']['gflops']:.2f} GFLOPS</td>
                    </tr>
            """
            
        html += """
                </table>
            </div>
            
            <div class="plot">
                <h2>Benchmark Results</h2>
                <img src="{}" alt="Benchmark Results" style="width: 100%;">
            </div>
        </body>
        </html>
        """.format(save_path.replace('.html', '.png'))
        
        with open(save_path, 'w') as f:
            f.write(html)
            
        print(f"Saved benchmark report to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Benchmark Tool')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device to use')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'], 
                        help='Data type for benchmarks')
    parser.add_argument('--matrix-size', type=int, default=5000, help='Size of matrices for matrix multiplication')
    parser.add_argument('--memory-size', type=float, default=1.0, help='Size of memory transfer in GB')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for convolution benchmark')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations for each benchmark')
    parser.add_argument('--output', type=str, default=None, help='Path to save benchmark report')
    parser.add_argument('--skip-matrix', action='store_true', help='Skip matrix multiplication benchmark')
    parser.add_argument('--skip-memory', action='store_true', help='Skip memory transfer benchmark')
    parser.add_argument('--skip-conv', action='store_true', help='Skip convolution benchmark')
    
    args = parser.parse_args()
    
    # Set dtype
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    
    try:
        benchmark = GPUBenchmark(device=args.device, dtype=dtype)
        
        if not args.skip_matrix:
            benchmark.matrix_multiplication(size=args.matrix_size, iterations=args.iterations)
            
        if not args.skip_memory:
            benchmark.memory_benchmark(size_gb=args.memory_size, iterations=args.iterations)
            
        if not args.skip_conv:
            benchmark.conv_benchmark(batch_size=args.batch_size, iterations=args.iterations)
            
        benchmark.generate_report(save_path=args.output)
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
EOF

    # GPU monitoring script
    cat > "${TEST_DIR}/gpu_monitor.py" << 'EOF'
#!/usr/bin/env python3
"""
GPU Monitoring Script
Continuously monitors GPU status and reports usage statistics
"""

import subprocess
import time
import argparse
import datetime
import json
import os
import signal
import sys
from collections import deque

class GPUMonitor:
    def __init__(self, log_file=None, interval=1, history_size=60):
        self.interval = interval
        self.log_file = log_file
        self.history_size = history_size
        self.running = True
        self.history = {}
        
        # Set up signal handler for graceful termination
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Initialize log file if specified
        if log_file:
            with open(log_file, 'w') as f:
                f.write("timestamp,gpu_id,name,temperature,gpu_util,mem_util,mem_used,mem_total,power\n")
    
    def signal_handler(self, sig, frame):
        print("\nShutting down GPU monitor...")
        self.running = False
    
    def get_gpu_info(self):
        """Get current GPU information using nvidia-smi"""
        try:
            cmd = "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
            
            gpus = []
            for line in output.strip().split('\n'):
                values = [val.strip() for val in line.split(',')]
                if len(values) >= 8:
                    gpu = {
                        "index": int(values[0]),
                        "name": values[1],
                        "temperature": float(values[2]),
                        "gpu_util": float(values[3]),
                        "mem_util": float(values[4]),
                        "mem_used": float(values[5]),
                        "mem_total": float(values[6]),
                        "power": float(values[7]) if values[7] else 0
                    }
                    gpus.append(gpu)
            
            return gpus
        except (subprocess.SubprocessError, ValueError) as e:
            print(f"Error getting GPU info: {e}")
            return []
    
    def update_history(self, gpus):
        """Update history with new GPU data"""
        timestamp = datetime.datetime.now()
        
        for gpu in gpus:
            idx = gpu["index"]
            if idx not in self.history:
                self.history[idx] = {
                    "timestamps": deque(maxlen=self.history_size),
                    "temperatures": deque(maxlen=self.history_size),
                    "gpu_utils": deque(maxlen=self.history_size),
                    "mem_utils": deque(maxlen=self.history_size),
                    "mem_useds": deque(maxlen=self.history_size),
                    "powers": deque(maxlen=self.history_size)
                }
            
            self.history[idx]["timestamps"].append(timestamp)
            self.history[idx]["temperatures"].append(gpu["temperature"])
            self.history[idx]["gpu_utils"].append(gpu["gpu_util"])
            self.history[idx]["mem_utils"].append(gpu["mem_util"])
            self.history[idx]["mem_useds"].append(gpu["mem_used"])
            self.history[idx]["powers"].append(gpu["power"])
    
    def log_to_file(self, gpus):
        """Log GPU data to file"""
        if not self.log_file:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            for gpu in gpus:
                f.write(f"{timestamp},{gpu['index']},{gpu['name']},{gpu['temperature']},{gpu['gpu_util']},{gpu['mem_util']},{gpu['mem_used']},{gpu['mem_total']},{gpu['power']}\n")
    
    def print_summary(self, gpus):
        """Print current GPU status summary"""
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"=== GPU Monitor ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===")
        print(f"Monitoring interval: {self.interval}s | Press Ctrl+C to exit")
        print("-" * 80)
        
        header = f"{'GPU':<4} {'Name':<20} {'Temp(°C)':<10} {'GPU Util(%)':<12} {'Mem Util(%)':<12} {'Memory Used':<15} {'Power(W)':<10}"
        print(header)
        print("-" * 80)
        
        for gpu in gpus:
            gpu_line = (
                f"{gpu['index']:<4} "
                f"{gpu['name'][:18]:<20} "
                f"{gpu['temperature']:<10.1f} "
                f"{gpu['gpu_util']:<12.1f} "
                f"{gpu['mem_util']:<12.1f} "
                f"{gpu['mem_used']:.1f}/{gpu['mem_total']:.1f} MB  "
                f"{gpu['power']:<10.2f}"
            )
            print(gpu_line)
        
        print("-" * 80)
        
        # Print history summary if available
        for idx in self.history:
            if len(self.history[idx]["temperatures"]) > 1:
                hist = self.history[idx]
                avg_temp = sum(hist["temperatures"]) / len(hist["temperatures"])
                max_temp = max(hist["temperatures"])
                avg_gpu_util = sum(hist["gpu_utils"]) / len(hist["gpu_utils"])
                max_gpu_util = max(hist["gpu_utils"])
                avg_mem_used = sum(hist["mem_useds"]) / len(hist["mem_useds"])
                max_mem_used = max(hist["mem_useds"])
                
                print(f"GPU {idx} Summary (last {len(hist['temperatures'])} readings):")
                print(f"  Temperature: Avg {avg_temp:.1f}°C, Max {max_temp:.1f}°C")
                print(f"  GPU Utilization: Avg {avg_gpu_util:.1f}%, Max {max_gpu_util:.1f}%")
                print(f"  Memory Used: Avg {avg_mem_used:.1f} MB, Max {max_mem_used:.1f} MB")
                print()
    
    def run(self):
        """Main monitoring loop"""
        print("Starting GPU monitoring...")
        
        while self.running:
            try:
                # Get current GPU info
                gpus = self.get_gpu_info()
                
                if not gpus:
                    print("No GPUs found or nvidia-smi not available")
                    time.sleep(self.interval)
                    continue
                
                # Update history
                self.update_history(gpus)
                
                # Log to file if enabled
                self.log_to_file(gpus)
                
                # Print current status
                self.print_summary(gpus)
                
                # Wait for next update
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)
        
        print("GPU monitoring stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Monitoring Tool')
    parser.add_argument('--interval', type=float, default=1.0, help='Monitoring interval in seconds')
    parser.add_argument('--log', type=str, default=None, help='Log file path')
    parser.add_argument('--history', type=int, default=60, help='Number of historical readings to keep')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(log_file=args.log, interval=args.interval, history_size=args.history)
    monitor.run()
EOF

    # GPU stress test script
    cat > "${TEST_DIR}/gpu_stress_test.py" << 'EOF'
#!/usr/bin/env python3
"""
GPU Stress Test Script
Runs intensive workloads to stress test GPU performance and stability
"""

import torch
import argparse
import time
import signal
import sys
import random
import os
import threading
import subprocess
from datetime import datetime

class GPUStressTest:
    def __init__(self, device='cuda:0', duration=60, test_type='all', 
                 memory_fraction=0.8, log_interval=5, log_file=None):
        self.device = device
        self.duration = duration
        self.test_type = test_type
        self.memory_fraction = memory_fraction
        self.log_interval = log_interval
        self.log_file = log_file
        self.running = True
        self.tests = []
        self.log_thread = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot run GPU stress test.")
            
        # Get GPU properties
        self.gpu_name = torch.cuda.get_device_name(device)
        self.gpu_mem = torch.cuda.get_device_properties(device).total_memory
        
        print(f"Running stress test on {self.gpu_name}")
        print(f"Test type: {test_type}")
        print(f"Duration: {duration} seconds")
        print(f"Memory fraction: {memory_fraction}")
        
        # Initialize log file if specified
        if log_file:
            with open(log_file, 'w') as f:
                f.write("timestamp,test_type,gpu_util,mem_util,temperature,power,clock\n")
    
    def signal_handler(self, sig, frame):
        print("\nShutting down stress test...")
        self.running = False
        if self.log_thread and self.log_thread.is_alive():
            self.log_thread.join()
        sys.exit(0)
    
    def get_gpu_info(self):
        """Get current GPU information using nvidia-smi"""
        try:
            cmd = "nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.sm --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
            
            values = [val.strip() for val in output.strip().split(',')]
            if len(values) >= 6:
                return {
                    "gpu_util": float(values[1]),
                    "mem_util": float(values[2]),
                    "temperature": float(values[3]),
                    "power": float(values[4]) if values[4] else 0,
                    "clock": float(values[5]) if values[5] else 0
                }
            return None
        except subprocess.SubprocessError:
            return None
    
    def log_metrics(self, test_name):
        """Monitor and log GPU metrics during testing"""
        print(f"Starting metrics logging for {test_name}...")
        start_time = time.time()
        
        while self.running and (time.time() - start_time < self.duration):
            try:
                gpu_info = self.get_gpu_info()
                if gpu_info:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] {test_name}: " + 
                          f"GPU: {gpu_info['gpu_util']:.1f}%, " + 
                          f"Memory: {gpu_info['mem_util']:.1f}%, " +
                          f"Temp: {gpu_info['temperature']:.1f}°C, " +
                          f"Power: {gpu_info['power']:.2f}W, " +
                          f"Clock: {gpu_info['clock']:.0f}MHz")
                    
                    if self.log_file:
                        with open(self.log_file, 'a') as f:
                            f.write(f"{timestamp},{test_name},{gpu_info['gpu_util']},{gpu_info['mem_util']}," +
                                    f"{gpu_info['temperature']},{gpu_info['power']},{gpu_info['clock']}\n")
            except Exception as e:
                print(f"Error logging metrics: {e}")
                
            time.sleep(self.log_interval)
        
        print(f"Finished logging metrics for {test_name}")
    
    def memory_test(self):
        """Test GPU memory allocation and access"""
        test_name = "memory_test"
        print(f"\nRunning {test_name}...")
        
        # Calculate memory to use based on available memory
        self.log_thread = threading.Thread(target=self.log_metrics, args=(test_name,))
        self.log_thread.start()
        
        try:
            # Get memory size
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            target_mem = int(total_mem * self.memory_fraction)
            
            # Allocate memory gradually
            chunk_size = 1024 * 1024 * 128  # 128MB chunks
            tensors = []
            
            print(f"Target memory usage: {target_mem / (1024**2):.0f}MB")
            allocated = 0
            
            start_time = time.time()
            while self.running and (time.time() - start_time < self.duration):
                try:
                    if allocated < target_mem:
                        # Allocate more memory
                        remaining = target_mem - allocated
                        size = min(chunk_size, remaining)
                        tensor = torch.rand(size // 4, device=self.device)  # 4 bytes per float32
                        tensors.append(tensor)
                        allocated += size
                        print(f"Allocated {allocated / (1024**2):.0f}MB / {target_mem / (1024**2):.0f}MB")
                    
                    # Access memory to ensure it's used
                    for i, tensor in enumerate(tensors):
                        if random.random() < 0.2:  # 20% chance to access each tensor
                            tensor.add_(0.01).clamp_(0, 1)
                    
                    # Small delay
                    time.sleep(1)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Out of memory error: {e}")
                        # If we run out of memory, stop allocating but keep accessing
                        time.sleep(5)
                    else:
                        raise
            
            # Clean up
            del tensors
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in memory test: {e}")
        
        if self.log_thread:
            self.log_thread.join()
        
        print(f"Completed {test_name}")
    
    def compute_test(self):
        """Test GPU compute performance with heavy matrix operations"""
        test_name = "compute_test"
        print(f"\nRunning {test_name}...")
        
        self.log_thread = threading.Thread(target=self.log_metrics, args=(test_name,))
        self.log_thread.start()
        
        try:
            # Create large matrices
            matrix_size = 10000
            a = torch.randn(matrix_size, matrix_size, device=self.device)
            b = torch.randn(matrix_size, matrix_size, device=self.device)
            
            start_time = time.time()
            operation_count = 0
            
            while self.running and (time.time() - start_time < self.duration):
                # Matrix multiplication
                c = torch.matmul(a, b)
                # Some additional operations to keep the GPU busy
                c = torch.nn.functional.relu(c)
                d = c.transpose(0, 1)
                e = torch.matmul(d, a)
                
                operation_count += 1
                if operation_count % 10 == 0:
                    print(f"Completed {operation_count} matrix operations")
                    
                # Ensure we're using the results
                del c, d, e
                torch.cuda.synchronize()
            
            # Clean up
            del a, b
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in compute test: {e}")
        
        if self.log_thread:
            self.log_thread.join()
            
        print(f"Completed {test_name}")
    
    def mixed_test(self):
        """Test GPU with a mix of memory and compute operations"""
        test_name = "mixed_test"
        print(f"\nRunning {test_name}...")
        
        self.log_thread = threading.Thread(target=self.log_metrics, args=(test_name,))
        self.log_thread.start()
        
        try:
            # Calculate memory to use
            total_mem = torch.cuda.get_device_properties(self.device).total_memory
            target_mem = int(total_mem * self.memory_fraction * 0.7)  # Use 70% of target for base memory
            
            # Allocate memory
            base_tensor = torch.rand(target_mem // 4, device=self.device)  # 4 bytes per float32
            
            # Create matrices for compute operations
            matrix_size = 5000
            a = torch.randn(matrix_size, matrix_size, device=self.device)
            b = torch.randn(matrix_size, matrix_size, device=self.device)
            
            start_time = time.time()
            operation_count = 0
            
            while self.running and (time.time() - start_time < self.duration):
                # Matrix operations
                c = torch.matmul(a, b)
                d = torch.nn.functional.relu(c)
                
                # Memory operations
                base_tensor.add_(0.01).clamp_(0, 1)
                
                # Create and destroy additional tensors
                temp_size = int(target_mem * 0.3 // 4)  # Use remaining 30% for temp tensors
                temp = torch.rand(temp_size, device=self.device)
                temp.mul_(2).add_(c.mean())
                
                operation_count += 1
                if operation_count % 10 == 0:
                    print(f"Completed {operation_count} mixed operations")
                    
                # Clean up temporary tensors
                del c, d, temp
                torch.cuda.synchronize()
            
            # Final cleanup
            del base_tensor, a, b
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in mixed test: {e}")
        
        if self.log_thread:
            self.log_thread.join()
            
        print(f"Completed {test_name}")
    
    def cuda_kernel_test(self):
        """Test GPU with CUDA kernels using convolutional operations"""
        test_name = "cuda_kernel_test"
        print(f"\nRunning {test_name}...")
        
        self.log_thread = threading.Thread(target=self.log_metrics, args=(test_name,))
        self.log_thread.start()
        
        try:
            # Create input data (typical image batch)
            batch_size = 64
            input_tensor = torch.randn(batch_size, 3, 224, 224, device=self.device)
            
            # Create several convolutional layers
            conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(self.device)
            conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1).to(self.device)
            conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1).to(self.device)
            pool = torch.nn.MaxPool2d(2)
            
            start_time = time.time()
            operation_count = 0
            
            while self.running and (time.time() - start_time < self.duration):
                # Forward pass
                x = conv1(input_tensor)
                x = torch.nn.functional.relu(x)
                x = pool(x)
                
                x = conv2(x)
                x = torch.nn.functional.relu(x)
                x = pool(x)
                
                x = conv3(x)
                x = torch.nn.functional.relu(x)
                
                # Ensure we're using the results
                loss = x.sum()
                loss.backward()
                
                operation_count += 1
                if operation_count % 10 == 0:
                    print(f"Completed {operation_count} convolution operations")
                
                # Clear gradients for next iteration
                conv1.zero_grad()
                conv2.zero_grad()
                conv3.zero_grad()
                
                torch.cuda.synchronize()
            
            # Clean up
            del input_tensor, conv1, conv2, conv3, pool, x
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in CUDA kernel test: {e}")
        
        if self.log_thread:
            self.log_thread.join()
            
        print(f"Completed {test_name}")
    
    def run(self):
        """Run the selected stress tests"""
        print("\n=== Starting GPU Stress Test ===")
        print(f"Device: {self.gpu_name}")
        print(f"Test duration: {self.duration} seconds per test")
        
        start_time = time.time()
        
        if self.test_type == 'all' or self.test_type == 'memory':
            self.memory_test()
            
        if self.test_type == 'all' or self.test_type == 'compute':
            self.compute_test()
            
        if self.test_type == 'all' or self.test_type == 'mixed':
            self.mixed_test()
            
        if self.test_type == 'all' or self.test_type == 'cuda':
            self.cuda_kernel_test()
        
        total_time = time.time() - start_time
        print(f"\n=== GPU Stress Test Completed ===")
        print(f"Total time: {total_time:.2f} seconds")
        print("If your GPU completed all tests without errors or crashes, it appears to be stable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Stress Test Tool')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device to use')
    parser.add_argument('--duration', type=int, default=60, help='Duration of each test in seconds')
    parser.add_argument('--type', type=str, default='all', 
                        choices=['all', 'memory', 'compute', 'mixed', 'cuda'], 
                        help='Type of stress test to run')
    parser.add_argument('--memory', type=float, default=0.8, 
                        help='Fraction of GPU memory to use (0.0-1.0)')
    parser.add_argument('--log-interval', type=int, default=5, 
                        help='Interval in seconds between logging GPU metrics')
    parser.add_argument('--log-file', type=str, default=None, 
                        help='File to log GPU metrics to')
    
    args = parser.parse_args()
    
    try:
        stress_test = GPUStressTest(
            device=args.device,
            duration=args.duration,
            test_type=args.type,
            memory_fraction=args.memory,
            log_interval=args.log_interval,
            log_file=args.log_file
        )
        stress_test.run()
    except Exception as e:
        print(f"Error running stress test: {e}")
EOF

    # Create a README file
    cat > "${TEST_DIR}/README.md" << 'EOF'
# GPU Testing Toolkit

This toolkit provides utilities for testing and benchmarking NVIDIA GPUs on Ubuntu.

## Overview

The GPU Testing Toolkit includes several Python scripts for:

- Getting detailed GPU information
- Benchmarking GPU performance
- Monitoring GPU usage in real-time
- Stress testing GPU performance and stability

## Prerequisites

- Ubuntu Linux with NVIDIA GPU drivers installed
- Python 3.6+ with pip
- NVIDIA CUDA Toolkit

## Setup

1. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

2. Verify setup:
   ```
   python gpu_info.py
   ```

## Tools

### 1. GPU Information (`gpu_info.py`)

Shows detailed information about your GPU, including:
- Hardware specifications
- Driver version
- CUDA capabilities
- Memory usage
- PCIe configuration

Usage:
```
python gpu_info.py
```

### 2. GPU Benchmark (`gpu_benchmark.py`)

Measures GPU performance in several areas:
- Matrix multiplication
- Memory transfer speed
- Convolution operations

Generates HTML report with plots.

Usage:
```
python gpu_benchmark.py --matrix-size 5000 --memory-size 1.0 --batch-size 64 --iterations 5
```

Options:
- `--device`: CUDA device to use (default: 'cuda:0')
- `--dtype`: Data type for benchmarks (float32 or float64)
- `--matrix-size`: Size of matrices for matrix multiplication
- `--memory-size`: Size of memory transfer in GB
- `--batch-size`: Batch size for convolution benchmark
- `--iterations`: Number of iterations for each benchmark
- `--output`: Path to save benchmark report
- `--skip-matrix`: Skip matrix multiplication benchmark
- `--skip-memory`: Skip memory transfer benchmark
- `--skip-conv`: Skip convolution benchmark

### 3. GPU Monitor (`gpu_monitor.py`)

Provides real-time monitoring of GPU metrics:
- Temperature
- Utilization (GPU and Memory)
- Memory usage
- Power consumption

Displays live updates and optionally logs data to a CSV file.

Usage:
```
python gpu_monitor.py --interval 1 --log gpu_stats.csv
```

Options:
- `--interval`: Monitoring interval in seconds
- `--log`: Log file path
- `--history`: Number of historical readings to keep

### 4. GPU Stress Test (`gpu_stress_test.py`)

Runs intensive workloads to test GPU performance and stability:
- Memory allocation and access
- Compute operations
- Mixed workloads
- CUDA kernel operations

Usage:
```
python gpu_stress_test.py --duration 300 --type all --memory 0.8
```

Options:
- `--device`: CUDA device to use (default: 'cuda:0')
- `--duration`: Duration of each test in seconds
- `--type`: Type of stress test to run (all, memory, compute, mixed, cuda)
- `--memory`: Fraction of GPU memory to use (0.0-1.0)
- `--log-interval`: Interval in seconds between logging GPU metrics
- `--log-file`: File to log GPU metrics to

## Example Workflows

### Basic Performance Check
```
python gpu_info.py
python gpu_benchmark.py --output benchmark_report.html
```

### Stability Testing
```
python gpu_stress_test.py --duration 1800 --type all --log-file stress_log.csv
```

### Monitoring During Workload
```
python gpu_monitor.py --interval 2 --log workload_stats.csv
```

## Troubleshooting

If you encounter issues:

1. Verify NVIDIA drivers are installed and working:
   ```
   nvidia-smi
   ```

2. Check CUDA installation:
   ```
   nvcc --version
   ```

3. Make sure PyTorch is installed with CUDA support:
   ```
   python -c "import torch; print(torch.cuda.is_available())"
   ```
EOF

    # Create a simple shell script to run all tests
    cat > "${TEST_DIR}/run-all-tests.sh" << 'EOF'
#!/bin/bash
# Run all GPU tests in sequence

set -e

# Activate the virtual environment
source venv/bin/activate

# Define output directory
OUTPUT_DIR="reports"
mkdir -p "$OUTPUT_DIR"

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "===== Running GPU Tests ====="
echo "Date: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo

# Step 1: Run GPU Info
echo "Step 1: Getting GPU information..."
python gpu_info.py > "$OUTPUT_DIR/gpu_info_${TIMESTAMP}.txt"
echo "GPU information saved to $OUTPUT_DIR/gpu_info_${TIMESTAMP}.txt"
echo

# Step 2: Run benchmark
echo "Step 2: Running GPU benchmark..."
python gpu_benchmark.py --output "$OUTPUT_DIR/benchmark_${TIMESTAMP}.html"
echo "Benchmark completed. Report saved to $OUTPUT_DIR/benchmark_${TIMESTAMP}.html"
echo

# Step 3: Run quick stress test
echo "Step 3: Running quick GPU stress test (2 minutes)..."
python gpu_stress_test.py --duration 120 --type all --log-file "$OUTPUT_DIR/stress_${TIMESTAMP}.csv"
echo "Stress test completed. Log saved to $OUTPUT_DIR/stress_${TIMESTAMP}.csv"
echo

echo "All tests completed successfully!"
echo "Results are available in the $OUTPUT_DIR directory"
EOF

    # Make script executable
    chmod +x "${TEST_DIR}/run-all-tests.sh"
    
    # Set permissions
    chown -R $SUDO_USER:$SUDO_USER "${TEST_DIR}"
    chmod -R 755 "${TEST_DIR}"
    
    success "Test scripts created successfully"
}

# Main function
main() {
    # Check if script is run as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (sudo)"
    fi
    
    if [[ -z "$SUDO_USER" ]]; then
        error "No SUDO_USER environment variable. Run with 'sudo' instead of as root."
    fi
    
    log "Setting up GPU Python testing environment"
    
    # Check for NVIDIA GPU
    check_gpu
    
    # Create test directory
    TEST_DIR=$(create_test_directory)
    
    # Setup Python environment
    setup_python_env "$TEST_DIR"
    
    # Create test scripts
    create_test_scripts "$TEST_DIR"
    
    success "GPU testing environment setup complete!"
    log "Test directory created at: $TEST_DIR"
    log "To run all tests, use: cd $TEST_DIR && ./run-all-tests.sh"
}

# Run main function
main