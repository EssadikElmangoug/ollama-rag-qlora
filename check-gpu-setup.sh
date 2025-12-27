#!/bin/bash
# Script to check if GPU support is properly configured for Docker

echo "=========================================="
echo "GPU Support Check for Docker"
echo "=========================================="
echo ""

# Check 1: NVIDIA drivers
echo "1. Checking NVIDIA drivers..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ nvidia-smi found"
    echo "   GPU Info:"
    nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | sed 's/^/      /'
else
    echo "   ✗ nvidia-smi not found - NVIDIA drivers not installed"
    exit 1
fi

echo ""

# Check 2: NVIDIA Container Toolkit
echo "2. Checking NVIDIA Container Toolkit..."
if command -v nvidia-ctk &> /dev/null; then
    echo "   ✓ nvidia-ctk found"
    nvidia-ctk --version | head -1 | sed 's/^/      /'
else
    echo "   ✗ nvidia-ctk not found"
    echo "   → Run: sudo ./install-gpu-support.sh"
    exit 1
fi

echo ""

# Check 3: Docker NVIDIA runtime
echo "3. Checking Docker NVIDIA runtime..."
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "   ✓ Docker has NVIDIA runtime configured"
    docker info 2>/dev/null | grep -i nvidia | sed 's/^/      /'
else
    echo "   ✗ Docker NVIDIA runtime not configured"
    echo "   → Run: sudo nvidia-ctk runtime configure --runtime=docker"
    echo "   → Then: sudo systemctl restart docker"
    exit 1
fi

echo ""

# Check 4: Test GPU in container
echo "4. Testing GPU access in Docker container..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "   ✓ GPU access works in Docker containers"
    echo ""
    echo "   Testing with nvidia-smi:"
    docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^/      /'
else
    echo "   ✗ GPU access test failed"
    echo "   → Check Docker daemon: sudo systemctl status docker"
    echo "   → Restart Docker: sudo systemctl restart docker"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ All checks passed! GPU support is ready."
echo "=========================================="
echo ""
echo "You can now run: docker-compose up -d"



