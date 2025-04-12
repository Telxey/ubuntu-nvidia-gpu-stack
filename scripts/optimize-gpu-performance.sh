#!/bin/bash
# GPU Performance Optimization Script
# Author: CustomUbuntu Team
# Date: April 12, 2025

# Set persistence mode (keeps GPU initialized even when not in use)
echo "Setting persistence mode..."
sudo nvidia-smi -pm 1

# Set compute mode to default (allows multiple compute applications to run on one GPU)
echo "Setting compute mode..."
sudo nvidia-smi -c 0

# Disable autoboost and set GPU clocks to maximum
# Note: This should only be done after careful testing in your environment
echo "Setting power management..."
sudo nvidia-smi -pl 250  # Replace with your GPU's appropriate power limit

# Enable persistence daemon for better performance
echo "Ensuring nvidia-persistenced is running..."
if ! systemctl is-active --quiet nvidia-persistenced; then
    sudo systemctl start nvidia-persistenced
    sudo systemctl enable nvidia-persistenced
fi

# Check current PCIe link status
echo "Current PCIe status:"
nvidia-smi --query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv

# Apply best practice NUMA settings for multi-GPU systems
if [ $(lscpu | grep -c "NUMA node(s)") -gt 1 ] && [ $(nvidia-smi --query-gpu=count --format=csv,noheader) -gt 1 ]; then
    echo "Multi-GPU NUMA system detected, optimizing NUMA settings..."
    echo "always" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
    echo "For best performance, ensure applications use CUDA_VISIBLE_DEVICES to match GPUs to their local NUMA node"
fi

echo "GPU optimization complete!"
