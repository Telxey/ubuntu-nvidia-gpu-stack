#!/bin/bash
# Set up GPU management infrastructure
# Author: Ubuntu NVIDIA GPU Stack Team
# Date: April 12, 2025

set -e  # Exit on error

# Configuration
CHROOT_DIR="$(pwd)/build/chroot"
VENV_PATH="/opt/gpu-manager-venv"
GPU_MANAGER_USER="gpu-manager"

# Check if chroot exists
if [ ! -d "$CHROOT_DIR" ]; then
    echo "Error: Chroot directory not found at $CHROOT_DIR"
    echo "Please run 01-prepare-build-env.sh first"
    exit 1
fi

# Function to execute commands in chroot
run_in_chroot() {
    sudo chroot "$CHROOT_DIR" /bin/bash -c "$1"
}

echo "Installing Python and virtual environment tools..."
run_in_chroot "apt-get update && apt-get install -y python3 python3-venv python3-pip"

echo "Creating GPU manager user..."
run_in_chroot "useradd -r -m -d /var/lib/gpu-manager -s /bin/false $GPU_MANAGER_USER || true"

echo "Creating directory structure..."
run_in_chroot "mkdir -p /usr/local/bin/gpu-manager /etc/gpu-manager /var/log/gpu-manager"
run_in_chroot "chown $GPU_MANAGER_USER:$GPU_MANAGER_USER /var/log/gpu-manager"

echo "Creating virtual environment for GPU management..."
run_in_chroot "python3 -m venv $VENV_PATH"
run_in_chroot "$VENV_PATH/bin/pip install --upgrade pip"
run_in_chroot "$VENV_PATH/bin/pip install torch==2.6.0 nvidia-ml-py==12.535.133 psutil==5.9.8 pyyaml==6.0.1"

echo "Setting up configuration file..."
sudo mkdir -p "$CHROOT_DIR/etc/gpu-manager"
cat << 'EOL' | sudo tee "$CHROOT_DIR/etc/gpu-manager/config.yaml"
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
    - 192.168.0.0/24
EOL

echo "Setting up log rotation..."
cat << 'EOL' | sudo tee "$CHROOT_DIR/etc/logrotate.d/gpu-manager"
/var/log/gpu-manager/*.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
    create 0640 gpu-manager gpu-manager
}
EOL

echo "Copying utility scripts..."
sudo cp "$(pwd)/scripts/check-gpu-status.sh" "$CHROOT_DIR/usr/local/bin/gpu-manager/"
sudo cp "$(pwd)/scripts/optimize-gpu-performance.sh" "$CHROOT_DIR/usr/local/bin/gpu-manager/"
sudo cp "$(pwd)/scripts/setup-gpu-python-env.sh" "$CHROOT_DIR/usr/local/bin/gpu-manager/"

run_in_chroot "chmod +x /usr/local/bin/gpu-manager/*.sh"
run_in_chroot "ln -sf /usr/local/bin/gpu-manager/check-gpu-status.sh /usr/local/bin/check-gpu-status"
run_in_chroot "ln -sf /usr/local/bin/gpu-manager/optimize-gpu-performance.sh /usr/local/bin/optimize-gpu-performance"
run_in_chroot "ln -sf /usr/local/bin/gpu-manager/setup-gpu-python-env.sh /usr/local/bin/setup-gpu-python-env"

echo "GPU management infrastructure setup complete!"
echo "You can now proceed with installing additional tools like Filebrowser."
