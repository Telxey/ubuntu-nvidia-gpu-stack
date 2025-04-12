# Build Process for Custom Ubuntu with NVIDIA GPU Stack

This directory contains the scripts and instructions needed to build the custom Ubuntu ISO with NVIDIA GPU stack.

## Prerequisites
- Ubuntu 24.04 host system (recommended)
- At least 20GB of free disk space
- The base Ubuntu ISO and NVIDIA components (see `/downloads` directory)
- Root or sudo access

## Build Structure
The build process creates the following structure:
- `chroot`: Chroot environment for customizing the ISO
- `iso`: Extracted contents of the base ISO
- `mnt`: Temporary mount point
- `squashfs`: Extracted squashfs filesystem

## Build Steps
1. Extract the original Ubuntu 24.04.2 Server ISO
2. Install essential build tools
3. Add NVIDIA drivers and CUDA toolkit
4. Set up Python environment with ML libraries
5. Create GPU management infrastructure 
6. Integrate Filebrowser
7. Configure system for optimal GPU performance
8. Repackage into a bootable ISO

## Usage
See the build scripts in this directory for detailed build instructions.
