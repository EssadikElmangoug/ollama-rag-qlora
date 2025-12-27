#!/bin/bash
# Script to install NVIDIA Container Toolkit - REQUIRED for GPU support
# This must be run to enable GPU access in Docker containers

set -e

echo "=========================================="
echo "NVIDIA Container Toolkit Installation"
echo "=========================================="
echo ""
echo "This script will install NVIDIA Container Toolkit"
echo "which is REQUIRED for Docker to access your NVIDIA GPUs."
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Check if nvidia-smi works
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✓ NVIDIA drivers detected"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRIBUTION=$ID$VERSION_ID
else
    echo "ERROR: Cannot detect Linux distribution"
    exit 1
fi

echo "Detected distribution: $DISTRIBUTION"
echo ""

# Add NVIDIA GPG key
echo "Step 1/5: Adding NVIDIA GPG key..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA repository
echo "Step 2/5: Adding NVIDIA repository..."
curl -s -L https://nvidia.github.io/libnvidia-container/$DISTRIBUTION/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
echo "Step 3/5: Updating package list..."
apt-get update -qq

# Install NVIDIA Container Toolkit
echo "Step 4/5: Installing NVIDIA Container Toolkit..."
apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
echo "Step 5/5: Configuring Docker..."
nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
echo ""
echo "Restarting Docker daemon..."
systemctl restart docker

# Wait for Docker to restart
echo "Waiting for Docker to restart..."
sleep 3

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "✓ Docker is configured with NVIDIA runtime"
else
    echo "⚠ Warning: Docker may not be fully configured"
fi

echo ""
echo "Testing GPU access in Docker container..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! GPU support is now enabled."
    echo "=========================================="
    echo ""
    echo "You can now run:"
    echo "  docker-compose up -d"
    echo ""
    echo "Your containers will have access to GPU 1 (NVIDIA)."
else
    echo ""
    echo "=========================================="
    echo "⚠ WARNING: Installation completed but GPU test failed"
    echo "=========================================="
    echo ""
    echo "Please check:"
    echo "  1. Docker daemon is running: systemctl status docker"
    echo "  2. NVIDIA drivers are installed: nvidia-smi"
    echo "  3. Docker has NVIDIA runtime: docker info | grep nvidia"
    echo ""
    echo "Try restarting Docker: sudo systemctl restart docker"
fi



