#!/bin/bash
# Prepare the build environment for custom Ubuntu ISO with NVIDIA GPU stack
# Author: Ubuntu NVIDIA GPU Stack Team
# Date: April 12, 2025

set -e  # Exit on error

# Configuration
SRC_ISO="$(pwd)/downloads/ubuntu-24.04.2-live-server-amd64.iso"
BUILD_DIR="$(pwd)/build"
ISO_DIR="$(pwd)/build/iso"
MNT_DIR="$(pwd)/build/mnt"
SQUASHFS_DIR="$(pwd)/build/squashfs"
CHROOT_DIR="$(pwd)/build/chroot"
OUTPUT_ISO="$(pwd)/iso/custom-ubuntu-24.04-nvidia.iso"

# Check if ISO exists
if [ ! -f "$SRC_ISO" ]; then
    echo "Error: Source ISO not found at $SRC_ISO"
    echo "Please download Ubuntu 24.04 Server ISO and place it in the downloads directory"
    exit 1
fi

# Create necessary directories
echo "Creating build directories..."
mkdir -p "$ISO_DIR" "$MNT_DIR" "$SQUASHFS_DIR" "$CHROOT_DIR"

# Mount ISO
echo "Mounting ISO image..."
if mountpoint -q "$MNT_DIR"; then
    echo "ISO already mounted, unmounting first..."
    sudo umount "$MNT_DIR"
fi
sudo mount -o loop "$SRC_ISO" "$MNT_DIR"

# Copy ISO contents
echo "Copying ISO contents to working directory..."
rsync -a --exclude=/casper/filesystem.squashfs "$MNT_DIR/" "$ISO_DIR/"

# Extract squashfs filesystem
echo "Extracting squashfs filesystem..."
sudo unsquashfs -f -d "$SQUASHFS_DIR" "$MNT_DIR/casper/filesystem.squashfs"

# Prepare chroot environment
echo "Preparing chroot environment..."
sudo rsync -a "$SQUASHFS_DIR/" "$CHROOT_DIR/"

# Ensure network connectivity in chroot
echo "Setting up network in chroot..."
sudo cp /etc/resolv.conf "$CHROOT_DIR/etc/resolv.conf"

# Mount necessary filesystems for chroot
echo "Mounting filesystems for chroot..."
sudo mount --bind /dev "$CHROOT_DIR/dev"
sudo mount --bind /dev/pts "$CHROOT_DIR/dev/pts"
sudo mount -t proc proc "$CHROOT_DIR/proc"
sudo mount -t sysfs sysfs "$CHROOT_DIR/sys"

echo "Build environment preparation complete!"
echo "You can now proceed with installing NVIDIA components."
