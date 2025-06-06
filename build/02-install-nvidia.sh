#!/bin/bash
# Install NVIDIA drivers and CUDA in the chroot environment
# Author: Ubuntu NVIDIA GPU Stack Team
# Date: April 12, 2025

set -e  # Exit on error

# Configuration
CHROOT_DIR="$(pwd)/build/chroot"
NVIDIA_DRIVER_VERSION="550-server"

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

echo "Installing necessary tools in chroot..."
run_in_chroot "apt-get update && apt-get install -y software-properties-common build-essential dkms"

echo "Adding graphics drivers PPA..."
run_in_chroot "add-apt-repository -y ppa:graphics-drivers/ppa && apt-get update"

echo "Installing NVIDIA driver..."
run_in_chroot "apt-get install -y nvidia-driver-$NVIDIA_DRIVER_VERSION"

echo "Installing CUDA Toolkit..."
run_in_chroot "apt-get install -y nvidia-cuda-toolkit"

echo "Configuring NVIDIA settings..."
run_in_chroot "nvidia-xconfig --enable-all-gpus || true"  # May fail in chroot, that's OK

echo "Setting up persistence mode for better performance..."
cat << 'EOL' | sudo tee "$CHROOT_DIR/etc/systemd/system/nvidia-persistenced.service"
[Unit]
Description=NVIDIA Persistence Daemon
Wants=syslog.target

[Service]
Type=forking
ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced
ExecStopPost=/bin/rm -rf /var/run/nvidia-persistenced

[Install]
WantedBy=multi-user.target
EOL

echo "Creating nvidia-persistenced user..."
run_in_chroot "useradd -r -U -u 143 -c 'NVIDIA Persistence Daemon' -d / -s /sbin/nologin nvidia-persistenced || true"

echo "Enabling NVIDIA persistence daemon..."
run_in_chroot "systemctl enable nvidia-persistenced"

echo "Adding modprobe configuration for NVIDIA..."
cat << 'EOL' | sudo tee "$CHROOT_DIR/etc/modprobe.d/nvidia-graphics-drivers.conf"
# This file was generated by 02-install-nvidia.sh
# Enable NVIDIA driver settings
options nvidia NVreg_PreserveVideoMemoryAllocations=1
options nvidia NVreg_TemporaryFilePath=/var/tmp
EOL

echo "Setting up NVIDIA driver config for Xorg..."
sudo mkdir -p "$CHROOT_DIR/etc/X11"
cat << 'EOL' | sudo tee "$CHROOT_DIR/etc/X11/xorg.conf.d/10-nvidia.conf"
Section "OutputClass"
    Identifier "nvidia"
    MatchDriver "nvidia-drm"
    Driver "nvidia"
    Option "AllowEmptyInitialConfiguration"
    Option "PrimaryGPU" "yes"
EndSection
EOL

echo "NVIDIA driver and CUDA installation complete!"
echo "You can now proceed with setting up the Python environment and other tools."
