#!/bin/bash
# Simple script to check NVIDIA GPU status
# Author: CustomUbuntu Team
# Date: April 12, 2025

echo "=== GPU Hardware Information ==="
nvidia-smi --query-gpu=gpu_name,driver_version,memory.total,power.draw,temperature.gpu --format=csv,noheader

echo -e "\n=== GPU Utilization ==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader

echo -e "\n=== Running GPU Processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

echo -e "\n=== PCIe Information ==="
nvidia-smi --query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader

# Check if PyTorch can access GPU
echo -e "\n=== PyTorch GPU Access Check ==="
/opt/gpu-manager-venv/bin/python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA Available: Yes')
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA Available: No')
"
