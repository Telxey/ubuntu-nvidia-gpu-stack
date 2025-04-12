#!/bin/bash
# Install Filebrowser for web-based file management
# Author: Ubuntu NVIDIA GPU Stack Team
# Date: April 12, 2025

set -e  # Exit on error

# Configuration
CHROOT_DIR="$(pwd)/build/chroot"
FILEBROWSER_VERSION="v2.32.0"
FILEBROWSER_ARCH="amd64"  # Change to arm64 for ARM-based systems

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

echo "Installing curl for downloading Filebrowser..."
run_in_chroot "apt-get update && apt-get install -y curl"

echo "Downloading Filebrowser binary..."
run_in_chroot "curl -fsSL https://github.com/filebrowser/filebrowser/releases/download/${FILEBROWSER_VERSION}/linux-${FILEBROWSER_ARCH}-filebrowser.tar.gz -o /tmp/filebrowser.tar.gz"

echo "Extracting Filebrowser binary..."
run_in_chroot "mkdir -p /tmp/filebrowser"
run_in_chroot "tar -xzf /tmp/filebrowser.tar.gz -C /tmp/filebrowser"
run_in_chroot "mv /tmp/filebrowser/filebrowser /usr/local/bin/"
run_in_chroot "chmod +x /usr/local/bin/filebrowser"
run_in_chroot "rm -rf /tmp/filebrowser /tmp/filebrowser.tar.gz"

echo "Creating Filebrowser config directory..."
run_in_chroot "mkdir -p /etc/filebrowser"

echo "Initializing Filebrowser database..."
run_in_chroot "filebrowser config init --database /etc/filebrowser.db"
run_in_chroot "filebrowser config set --address 0.0.0.0 --database /etc/filebrowser.db --port 8080 --root /"
run_in_chroot "filebrowser users add admin password --database /etc/filebrowser.db --perm.admin"

echo "Creating Filebrowser systemd service..."
cat << 'EOL' | sudo tee "$CHROOT_DIR/etc/systemd/system/filebrowser.service"
[Unit]
Description=Filebrowser
After=network.target

[Service]
ExecStart=/usr/local/bin/filebrowser --database /etc/filebrowser.db
Restart=on-failure
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOL

echo "Enabling Filebrowser service..."
run_in_chroot "systemctl enable filebrowser.service"

echo "Filebrowser installation complete!"
echo "You can now proceed with finalizing the ISO build."
