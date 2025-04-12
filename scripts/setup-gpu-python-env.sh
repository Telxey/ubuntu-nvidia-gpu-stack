#!/bin/bash
# Script to setup Python environment for GPU computing
# Author: CustomUbuntu Team
# Date: April 12, 2025

# Default installation path
VENV_PATH=${1:-"/opt/gpu-manager-venv"}

# Check if running as root for system-wide installation
if [[ $EUID -ne 0 ]] && [[ $VENV_PATH == "/opt/"* ]]; then
    echo "This script must be run as root for system-wide installation"
    echo "Usage: sudo $0 [/path/to/venv]"
    echo "For user installation: $0 ~/my-gpu-venv"
    exit 1
fi

echo "Setting up GPU Python environment at: $VENV_PATH"

# Make sure required packages are installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found, installing..."
    apt update
    apt install -y python3 python3-venv python3-pip
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv $VENV_PATH

# Activate and upgrade pip
echo "Upgrading pip..."
$VENV_PATH/bin/pip install --upgrade pip

# Install PyTorch with CUDA support and other required packages
echo "Installing GPU libraries and dependencies..."
$VENV_PATH/bin/pip install torch nvidia-ml-py==12.535.133 psutil==5.9.8 pyyaml==6.0.1

# Verify installation
echo "Verifying installation..."
$VENV_PATH/bin/python -c "
import torch
import nvidia_ml_py
import psutil
import yaml

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'nvidia-ml-py: {nvidia_ml_py.__version__}')
print(f'psutil: {psutil.__version__}')
print(f'PyYAML: {yaml.__version__}')
"

echo -e "\nSetup complete! To use this environment:"
echo "source $VENV_PATH/bin/activate"
echo "deactivate # to exit the environment when done"
