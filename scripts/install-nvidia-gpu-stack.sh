#!/bin/bash
# NVIDIA GPU Stack Installation Script
# This script installs the NVIDIA GPU stack on Ubuntu
# Author: Ubuntu-NVIDIA-GPU-Stack Team From Telxey
# Date: April 14, 2025

set -e

# Color codes for prettier output
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

check_requirements() {
    log "Checking system requirements..."
    
    # Check if running on Ubuntu
    if ! grep -q "Ubuntu" /etc/os-release; then
        error "This script is designed for Ubuntu systems only"
    fi
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (sudo)"
    fi
    
    # Check for GPU
    if ! lspci | grep -i nvidia > /dev/null; then
        warning "No NVIDIA GPU detected. Continuing anyway, but driver installation may fail"
    fi
    
    success "System requirements check passed"
}

install_dependencies() {
    log "Installing dependencies..."
    
    apt update
    apt install -y \
        build-essential \
        gcc \
        g++ \
        make \
        dkms \
        linux-headers-$(uname -r) \
        python3 \
        python3-pip \
        python3-venv \
        wget \
        curl

    success "Dependencies installed"
}

install_nvidia_drivers() {
    log "Installing NVIDIA drivers..."
    
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt update
    apt install -y nvidia-driver-550-server
    
    log "Setting up NVIDIA persistence mode service..."
    apt install -y nvidia-persistenced
    systemctl enable nvidia-persistenced
    systemctl start nvidia-persistenced
    
    success "NVIDIA drivers installed"
}

install_cuda() {
    log "Installing CUDA Toolkit..."
    
    local CUDA_VERSION="12.0"
    local CUDA_PKG="cuda-toolkit-12-0"
    
    # Add CUDA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt update
    apt install -y $CUDA_PKG
    
    # Set up environment variables
    echo 'export PATH=/usr/local/cuda/bin:$PATH' > /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
    chmod +x /etc/profile.d/cuda.sh
    
    success "CUDA Toolkit $CUDA_VERSION installed"
}

setup_gpu_manager() {
    log "Setting up GPU Manager infrastructure..."
    
    # Create user and group
    groupadd -f gpu-manager
    useradd -m -g gpu-manager -s /bin/bash gpu-manager || true
    
    # Create directory structure
    mkdir -p /usr/local/bin/gpu-manager
    mkdir -p /etc/gpu-manager
    mkdir -p /var/log/gpu-manager
    
    # Set permissions
    chown -R gpu-manager:gpu-manager /var/log/gpu-manager
    chmod 755 /usr/local/bin/gpu-manager
    
    # Copy configuration
    if [ -f "$(dirname "$0")/../config/gpu-manager-config.yaml" ]; then
        cp "$(dirname "$0")/../config/gpu-manager-config.yaml" /etc/gpu-manager/config.yaml
    else
        # Create default config if not available
        cat > /etc/gpu-manager/config.yaml << 'EOF'
# GPU Manager Configuration Template

logging:
  level: info
  file: /var/log/gpu-manager/gpu-manager.log
  max_size_mb: 10
  backup_count: 5

monitoring:
  interval_seconds: 5
  metrics_port: 9090

pcie:
  optimize: true
  settings:
    link_speed: auto
    pcie_gen: 4
    max_link_width: 16

gpu_settings:
  persistence_mode: enabled
  accounting_mode: enabled
  compute_mode: default
  auto_boost: enabled

alerts:
  temperature_threshold_celsius: 85
  memory_threshold_percent: 90
  power_threshold_percent: 95
  alert_destination_email: admin@example.com

api:
  enabled: true
  port: 8000
  auth_required: true
  allowed_ips:
    - 127.0.0.1
    - 0.0.0.0/0
EOF
    fi
    
    # Set up log rotation
    cat > /etc/logrotate.d/gpu-manager << 'EOF'
/var/log/gpu-manager/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 640 gpu-manager gpu-manager
    sharedscripts
    postrotate
        systemctl reload gpu-manager.service 2>/dev/null || true
    endscript
}
EOF

    success "GPU Manager infrastructure set up"
}

setup_python_environment() {
    log "Setting up Python environment..."
    
    VENV_PATH="/opt/gpu-manager-venv"
    
    # Create virtual environment
    python3 -m venv $VENV_PATH
    
    # Activate and upgrade pip
    $VENV_PATH/bin/pip install --upgrade pip
    
    # Install PyTorch with CUDA support and other required packages
    $VENV_PATH/bin/pip install torch nvidia-ml-py==12.535.133 psutil==5.9.8 pyyaml==6.0.1
    
    # Verify installation
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
    
    success "Python environment set up at $VENV_PATH"
}

install_filebrowser() {
    log "Installing Filebrowser..."
    
    # Download and install filebrowser
    curl -fsSL https://raw.githubusercontent.com/filebrowser/get/master/get.sh | bash
    
    # Create service
    cat > /etc/systemd/system/filebrowser.service << 'EOF'
[Unit]
Description=Filebrowser
After=network.target

[Service]
Type=simple
User=gpu-manager
Group=gpu-manager
ExecStart=/usr/local/bin/filebrowser -a 0.0.0.0 -p 8080 -r /
Restart=on-failure
RestartSec=5
StartLimitInterval=60s
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
EOF

    # Initialize filebrowser
    filebrowser config init --address 0.0.0.0 --port 8080 --baseurl "" --log /var/log/gpu-manager/filebrowser.log --root /
    filebrowser users add admin password --perm.admin
    
    # Start and enable service
    systemctl daemon-reload
    systemctl enable filebrowser.service
    systemctl start filebrowser.service
    
    success "Filebrowser installed and configured on port 8080"
}

copy_utility_scripts() {
    log "Copying utility scripts..."
    
    # Copy scripts if available, otherwise create them
    SCRIPT_DIR="$(dirname "$0")"
    
    # check-gpu-status.sh
    if [ -f "$SCRIPT_DIR/check-gpu-status.sh" ]; then
        cp "$SCRIPT_DIR/check-gpu-status.sh" /usr/local/bin/gpu-manager/
    else
        cat > /usr/local/bin/gpu-manager/check-gpu-status.sh << 'EOF'
#!/bin/bash
# Simple script to check NVIDIA GPU status

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
EOF
    fi
    
    # optimize-gpu-performance.sh
    if [ -f "$SCRIPT_DIR/optimize-gpu-performance.sh" ]; then
        cp "$SCRIPT_DIR/optimize-gpu-performance.sh" /usr/local/bin/gpu-manager/
    else
        cat > /usr/local/bin/gpu-manager/optimize-gpu-performance.sh << 'EOF'
#!/bin/bash
# GPU Performance Optimization Script

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
EOF
    fi
    
    # Make scripts executable
    chmod +x /usr/local/bin/gpu-manager/*.sh
    
    success "Utility scripts copied and made executable"
}

optimize_system() {
    log "Optimizing system for GPU workloads..."
    
    # Increase system limits for GPU workloads
    cat > /etc/security/limits.d/gpu-manager.conf << 'EOF'
# Increase limits for GPU workloads
*               soft    memlock         unlimited
*               hard    memlock         unlimited
*               soft    stack           unlimited
*               hard    stack           unlimited
*               soft    nofile          65535
*               hard    nofile          65535
EOF

    # Set up swappiness for better performance
    echo "vm.swappiness=10" > /etc/sysctl.d/90-gpu-swappiness.conf
    sysctl -p /etc/sysctl.d/90-gpu-swappiness.conf
    
    # Run the optimization script if the driver is installed
    if command -v nvidia-smi &> /dev/null; then
        /usr/local/bin/gpu-manager/optimize-gpu-performance.sh
    else
        log "NVIDIA driver not yet loaded, skipping GPU optimization. Run optimize-gpu-performance.sh after reboot."
    fi
    
    success "System optimized for GPU workloads"
}

verify_installation() {
    log "Verifying installation..."
    
    # Check if nvidia driver is loaded
    if ! lsmod | grep nvidia > /dev/null; then
        warning "NVIDIA driver not loaded yet. A reboot may be required."
        echo "Run the following command after reboot to verify driver installation:"
        echo "  nvidia-smi"
    else
        # Display driver information
        nvidia-smi
    fi
    
    # Check if directories and configs exist
    if [ -d "/etc/gpu-manager" ] && [ -f "/etc/gpu-manager/config.yaml" ]; then
        success "GPU Manager configuration verified"
    else
        warning "GPU Manager configuration is incomplete"
    fi
    
    # Check Python environment
    if [ -d "/opt/gpu-manager-venv" ]; then
        success "Python environment verified"
    else
        warning "Python environment installation issue detected"
    fi
    
    # Check filebrowser
    if systemctl is-active --quiet filebrowser.service; then
        success "Filebrowser service is running"
    else
        warning "Filebrowser service is not running"
    fi
    
    success "Installation verification complete"
    log "You may need to reboot your system to complete the installation"
}

main() {
    log "Starting NVIDIA GPU Stack installation..."
    
    check_requirements
    install_dependencies
    install_nvidia_drivers
    install_cuda
    setup_gpu_manager
    setup_python_environment
    install_filebrowser
    copy_utility_scripts
    optimize_system
    verify_installation
    
    success "NVIDIA GPU Stack installation complete!"
    log "Please reboot your system and then run 'nvidia-smi' to verify that everything is working"
    log "After reboot, you can access Filebrowser at http://your-server-ip:8080"
}

# Run main function
main
