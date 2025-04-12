#!/bin/bash
# Finalize the ISO build
# Author: Ubuntu NVIDIA GPU Stack Team
# Date: April 12, 2025

set -e  # Exit on error

# Configuration
CHROOT_DIR="$(pwd)/build/chroot"
ISO_DIR="$(pwd)/build/iso"
MNT_DIR="$(pwd)/build/mnt"
SQUASHFS_DIR="$(pwd)/build/squashfs"
OUTPUT_ISO="$(pwd)/iso/custom-ubuntu-24.04-nvidia.iso"
ISO_LABEL="Ubuntu-NVIDIA-24.04"
ISO_NAME="Ubuntu NVIDIA GPU Stack 24.04"

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

echo "Updating initramfs..."
run_in_chroot "update-initramfs -u"

echo "Cleaning up chroot environment..."
run_in_chroot "apt-get clean"
sudo rm -f "$CHROOT_DIR/etc/resolv.conf"

echo "Unmounting filesystems..."
sudo umount "$CHROOT_DIR/sys"
sudo umount "$CHROOT_DIR/proc"
sudo umount "$CHROOT_DIR/dev/pts"
sudo umount "$CHROOT_DIR/dev"

if mountpoint -q "$MNT_DIR"; then
    echo "Unmounting source ISO..."
    sudo umount "$MNT_DIR"
fi

echo "Creating new squashfs filesystem..."
sudo mksquashfs "$CHROOT_DIR" "$ISO_DIR/casper/filesystem.squashfs" -comp xz -noappend

echo "Generating filesystem size file..."
sudo du -sx --block-size=1 "$CHROOT_DIR" | cut -f1 | sudo tee "$ISO_DIR/casper/filesystem.size" > /dev/null

echo "Updating ISO meta information..."
echo "$ISO_NAME" | sudo tee "$ISO_DIR/.disk/info" > /dev/null

echo "Generating MD5 checksums..."
cd "$ISO_DIR"
sudo rm -f md5sum.txt
find . -type f -not -path "./isolinux/*" -not -path "./md5sum.txt" -not -path "./boot/*" -exec sudo md5sum {} \; | sudo tee md5sum.txt > /dev/null
cd - > /dev/null

echo "Creating ISO image..."
sudo xorriso -as mkisofs -isohybrid-mbr /usr/lib/ISOLINUX/isohdpfx.bin \
    -c isolinux/boot.cat -b isolinux/isolinux.bin -no-emul-boot -boot-load-size 4 \
    -boot-info-table -eltorito-alt-boot -e boot/grub/efi.img -no-emul-boot \
    -isohybrid-gpt-basdat -volid "$ISO_LABEL" -o "$OUTPUT_ISO" "$ISO_DIR"

echo "Generating MD5 checksum for the ISO..."
md5sum "$OUTPUT_ISO" | cut -d' ' -f1 > "$OUTPUT_ISO.md5"

echo "ISO build complete!"
echo "Your custom Ubuntu NVIDIA GPU Stack ISO is available at: $OUTPUT_ISO"
echo "MD5 checksum: $(cat "$OUTPUT_ISO.md5")"
