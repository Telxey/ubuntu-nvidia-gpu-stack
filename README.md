<p align="center">
   <img src="https://assets.ubuntu.com/v1/594d0a0c-Canonical%20Ubuntu%20Dark.svg" alt="ubuntu" height="160" width="900"></a>
</p>                                         


# Ubuntu NVIDIA GPU Stack

Custom Ubuntu 24.04 Server ISO with NVIDIA GPU management stack.

## Quick Installation

Install the complete NVIDIA GPU stack on any Ubuntu server with a single command:

```bash
# Clone the repository
git clone https://github.com/Telxey/ubuntu-nvidia-gpu-stack.git
cd ubuntu-nvidia-gpu-stack

# Run the installation script
sudo ./scripts/install-nvidia-gpu-stack.sh
```

### What Gets Installed

- NVIDIA Driver 550-server
- CUDA Toolkit 12.0
- Python environment with PyTorch and NVIDIA libraries
- GPU Manager configuration
- Filebrowser for remote file management (port 8080)
- Optimization scripts and system settings

### After Installation

After rebooting, verify your installation:

```bash
# Verify NVIDIA driver installation
nvidia-smi

# Check GPU status
/usr/local/bin/gpu-manager/check-gpu-status.sh

# Optimize GPU performance
sudo /usr/local/bin/gpu-manager/optimize-gpu-performance.sh
```

Access the Filebrowser at `http://your-server-ip:8080`

### GPU
● All GPU manager services are running properly:

  1. GPU Manager Services:
    - gpu-manager.service - Active ✓
    - gpu-activator.service - Active ✓
    - force-gpu-speed.service - Active ✓
    - nvidia-persistenced - Active ✓
  2. Log Files:
    - Active logs at /var/log/gpu-manager/
    - Recent entries showing GPU activation and PCIe optimization
  3. Python Environment:
    - Located at /opt/gpu-manager/venv/ (not at /opt/gpu-manager-venv)
    - Running the GPU management scripts
  4. GPU Management Scripts:
    - Located at /usr/local/bin/gpu-manager/
    - Both continuous_gpu_activator.py and pcie_analyzer_updated.py are running
  5. Configuration:
    - Config at /etc/gpu-manager/config.yaml

  All services are functioning correctly, keeping your GPUs optimized with PCIe Gen 4 settings and
   persistent mode.

## Documentation

For detailed information about the stack, see [docs/custom-ubuntu-nvidia-gpu-stack.md](docs/custom-ubuntu-nvidia-gpu-stack.md)

<p align="right">
   <a href="https://www.buymeacoffee.com/telxey" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-black.png" alt="Buy Me A Coffee" height="41" width="174"></a>
</p>
