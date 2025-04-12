#!/bin/bash
# Main script to build the custom Ubuntu ISO with NVIDIA GPU stack
# Author: Ubuntu NVIDIA GPU Stack Team
# Date: April 12, 2025

set -e  # Exit on error

SCRIPT_DIR="$(pwd)/build"

echo "==============================================="
echo "Custom Ubuntu ISO with NVIDIA GPU Stack Builder"
echo "==============================================="
echo 

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root (sudo)."
    echo "Please run: sudo $0"
    exit 1
fi

# Check for required tools
REQUIRED_TOOLS="rsync unsquashfs mksquashfs xorriso"
MISSING_TOOLS=""

for tool in $REQUIRED_TOOLS; do
    if ! command -v $tool &> /dev/null; then
        MISSING_TOOLS="$MISSING_TOOLS $tool"
    fi
done

if [ ! -z "$MISSING_TOOLS" ]; then
    echo "The following required tools are missing:$MISSING_TOOLS"
    echo "Please install them with:"
    echo "sudo apt-get install squashfs-tools xorriso rsync"
    exit 1
fi

echo "Step 1: Preparing build environment..."
"$SCRIPT_DIR/01-prepare-build-env.sh"
echo

echo "Step 2: Installing NVIDIA drivers and CUDA..."
"$SCRIPT_DIR/02-install-nvidia.sh"
echo

echo "Step 3: Setting up GPU management infrastructure..."
"$SCRIPT_DIR/03-setup-gpu-management.sh"
echo

echo "Step 4: Installing Filebrowser..."
"$SCRIPT_DIR/04-install-filebrowser.sh"
echo

echo "Step 5: Finalizing ISO build..."
"$SCRIPT_DIR/05-finalize-iso.sh"
echo

echo "Build process completed successfully!"
echo "Your custom Ubuntu NVIDIA GPU Stack ISO is available at: $(pwd)/iso/custom-ubuntu-24.04-nvidia.iso"
echo "MD5 checksum: $(cat "$(pwd)/iso/custom-ubuntu-24.04-nvidia.iso.md5")"
echo "Run the following to verify the MD5 checksum:"
echo "cd $(pwd)/iso && md5sum -c custom-ubuntu-24.04-nvidia.iso.md5"

