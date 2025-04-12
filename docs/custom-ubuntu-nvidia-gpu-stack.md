# Custom Ubuntu ISO with NVIDIA GPU Stack: Complete Reference

## ISO Information

```
Location: ~/custom-ubuntu-24.04-nvidia.iso
Size: 11G
MD5 checksum: 20d6ba71e82506cbbfaddaa68230a528
Creation Date: April 12, 2025
```

## Contents and Customizations

This custom Ubuntu 24.04 Server ISO has been tailored with the following NVIDIA GPU management components:

### 1. NVIDIA Drivers and CUDA Components

- **NVIDIA Driver**: 550-server (enterprise-grade)
- **CUDA Toolkit**: 12.0
- **NVIDIA configuration**: Configured with `nvidia-xconfig --enable-all-gpus`
- **NVIDIA Persistence Mode**: Enabled for improved performance

### 2. Python Environment and ML Libraries

- **Virtual Environment**: `/opt/gpu-manager-venv`
- **Python Packages**:
  - torch==2.6.0
  - nvidia-ml-py==12.535.133
  - psutil==5.9.8
  - pyyaml==6.0.1
  - CUDA dependencies (automatically installed with PyTorch)

### 3. GPU Management Infrastructure

- **System User/Group**: gpu-manager
- **Directory Structure**:
  - `/usr/local/bin/gpu-manager` - Executable scripts
  - `/etc/gpu-manager` - Configuration files
  - `/var/log/gpu-manager` - Log files
- **Configuration**: PCIe optimization settings in `/etc/gpu-manager/config.yaml`
- **Log Rotation**: Set up for GPU manager logs

### 4. Filebrowser for File Management

- **Binary**: Filebrowser v2.32.0 installed at `/usr/local/bin/filebrowser`
- **Service**: SystemD service configured to start at boot
- **Port**: Running on port 8080
- **Database**: Located at `/etc/filebrowser.db`

## Verification Steps

### Boot Testing

```bash
# Write to USB drive (replace X with your device letter)
sudo dd if=~/custom-ubuntu-24.04-nvidia.iso of=/dev/sdX bs=4M status=progress
```

### Virtual Machine Testing

```bash
# With QEMU/KVM
virt-install --name ubuntu-nvidia-test --memory 4096 --vcpus 2 --disk size=40 \
  --cdrom ~/custom-ubuntu-24.04-nvidia.iso --os-variant ubuntu24.04
   
# Or with VirtualBox (command line)
VBoxManage createvm --name ubuntu-nvidia-test --ostype Ubuntu_64 --register
VBoxManage modifyvm ubuntu-nvidia-test --memory 4096 --cpus 2
VBoxManage createmedium disk --filename ~/VirtualBox\ VMs/ubuntu-nvidia-test/disk.vdi --size 40000
VBoxManage storagectl ubuntu-nvidia-test --name "SATA Controller" --add sata --controller IntelAHCI
VBoxManage storageattach ubuntu-nvidia-test --storagectl "SATA Controller" --port 0 --device 0 --type hdd --medium ~/VirtualBox\ VMs/ubuntu-nvidia-test/disk.vdi
VBoxManage storageattach ubuntu-nvidia-test --storagectl "SATA Controller" --port 1 --device 0 --type dvddrive --medium ~/custom-ubuntu-24.04-nvidia.iso
VBoxManage startvm ubuntu-nvidia-test
```

### Post-Installation Verification

```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA installation
nvcc --version

# Test Python environment and GPU access
/opt/gpu-manager-venv/bin/python -c "import torch; print(torch.cuda.is_available())"

# Verify Filebrowser service
systemctl status filebrowser.service

# Access Filebrowser
# Open in browser: http://your-server-ip:8080

# Check GPU management configuration
cat /etc/gpu-manager/config.yaml
```

## GPU Manager Configuration

The system includes a comprehensive GPU management configuration file at `/etc/gpu-manager/config.yaml`:

```yaml
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
```

## Deployment Options

### PXE Boot Setup

For network-based deployments:

1. Configure TFTP server with ISO contents
2. Set up DHCP to point to the PXE boot server
3. Extract kernel and initrd from ISO:
   ```bash
   mkdir -p /tftpboot/ubuntu-nvidia
   cp ~/custom_iso_build/iso/casper/{vmlinuz,initrd} /tftpboot/ubuntu-nvidia/
   ```

### Automated Installation

For unattended installations, create a preseed file:

```bash
# Create preseed file
cat > preseed.cfg << 'END'
d-i debian-installer/locale string en_US.UTF-8
d-i keyboard-configuration/xkb-keymap select us
d-i netcfg/choose_interface select auto
d-i netcfg/get_hostname string ubuntu-gpu-node
d-i netcfg/get_domain string local
d-i passwd/root-login boolean false
d-i passwd/user-fullname string Ubuntu User
d-i passwd/username string ubuntu
d-i passwd/user-password password insecure
d-i passwd/user-password-again password insecure
d-i partman-auto/method string lvm
d-i partman-lvm/device_remove_lvm boolean true
d-i partman-md/device_remove_md boolean true
d-i partman-lvm/confirm boolean true
d-i partman-lvm/confirm_nooverwrite boolean true
d-i partman-auto/choose_recipe select atomic
d-i partman-partitioning/confirm_write_new_label boolean true
d-i partman/choose_partition select finish
d-i partman/confirm boolean true
d-i partman/confirm_nooverwrite boolean true
d-i grub-installer/only_debian boolean true
d-i grub-installer/with_other_os boolean true
END
```

## Build Process Summary

This ISO was created by:

1. Extracting the original Ubuntu 24.04.2 Server ISO
2. Installing essential build tools
3. Adding NVIDIA drivers and CUDA toolkit
4. Setting up Python environment with ML libraries
5. Creating GPU management infrastructure 
6. Integrating Filebrowser
7. Configuring system for optimal GPU performance
8. Repackaging into a bootable ISO

## Maintenance Notes

- **Driver Updates**: To update NVIDIA drivers, reinstall with newer versions from the graphics-drivers PPA
- **CUDA Updates**: Check compatibility between CUDA versions and installed drivers
- **Python Package Updates**: Use `/opt/gpu-manager-venv/bin/pip install --upgrade package_name`
- **Log Management**: Logs are rotated daily with 14-day retention

## Support Information

For issues with this custom image:

1. Check driver compatibility with your GPU models
2. Verify PCIe settings in `/etc/gpu-manager/config.yaml` match your hardware
3. Ensure adequate cooling for high-performance GPU workloads
4. When upgrading, back up `/etc/gpu-manager/` directory

---

Created: April 12, 2025  
MD5: 20d6ba71e82506cbbfaddaa68230a528
