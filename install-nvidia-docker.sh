#!/bin/bash
# Script to install NVIDIA Container Toolkit for Docker GPU support

set -e

echo "=== Installing NVIDIA Container Toolkit ==="

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRIBUTION=$ID$VERSION_ID
else
    echo "Error: Cannot detect Linux distribution"
    exit 1
fi

echo "Detected distribution: $DISTRIBUTION"

# Add NVIDIA GPG key
echo "Adding NVIDIA GPG key..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add NVIDIA repository
echo "Adding NVIDIA repository..."
curl -s -L https://nvidia.github.io/libnvidia-container/$DISTRIBUTION/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
echo "Configuring Docker..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
echo "Restarting Docker daemon..."
sudo systemctl restart docker

# Wait a moment for Docker to restart
sleep 2

# Verify installation
echo ""
echo "=== Verifying Installation ==="
echo "Testing GPU access in Docker..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi; then
    echo ""
    echo "✅ SUCCESS! NVIDIA Container Toolkit is installed and working."
    echo "You can now use docker-compose with GPU support."
else
    echo ""
    echo "⚠️  WARNING: Installation completed but GPU test failed."
    echo "Please check your NVIDIA drivers and Docker configuration."
fi

echo ""
echo "=== Installation Complete ==="
echo "To use GPU support, run: docker-compose up -d"
echo "To use CPU-only mode, run: docker-compose -f docker-compose.no-gpu.yml up -d"



