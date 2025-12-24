#!/bin/bash
# Windows/WSL2 GPU Setup Script for NVIDIA Container Toolkit
# Run this script in your WSL2 Ubuntu terminal

set -e

echo "=========================================="
echo "NVIDIA Container Toolkit Setup for WSL2"
echo "=========================================="
echo ""

# Check if running in WSL2
if [ -z "$WSL_DISTRO_NAME" ]; then
    echo "⚠️  Warning: This script is designed for WSL2."
    echo "   If you're on Linux, use install-gpu-support.sh instead."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if nvidia-smi works
echo "Step 1: Checking NVIDIA drivers..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA drivers detected"
else
    echo "⚠️  nvidia-smi not found. Make sure:"
    echo "   1. NVIDIA drivers are installed in Windows"
    echo "   2. WSL2 is properly configured"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 2: Adding NVIDIA package repositories..."

# Detect distribution
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

# Add GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "✅ Repository added"

echo ""
echo "Step 3: Installing NVIDIA Container Toolkit..."
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "✅ NVIDIA Container Toolkit installed"

echo ""
echo "Step 4: Configuring Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

echo "✅ Docker runtime configured"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: You must restart Docker Desktop now!"
echo ""
echo "1. Close Docker Desktop completely"
echo "2. Reopen Docker Desktop"
echo "3. Wait for it to fully start"
echo ""
echo "Then test GPU access with:"
echo "  docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
echo ""
echo "If that works, you can start your application:"
echo "  docker-compose up -d"
echo ""

