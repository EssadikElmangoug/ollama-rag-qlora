#!/bin/bash
# Script to check GPU visibility in Docker container

echo "=== Checking GPU on HOST ==="
nvidia-smi

echo ""
echo "=== Checking GPU in Docker Container ==="
docker exec engabusami-backend-dev nvidia-smi 2>/dev/null || docker exec engabusami-backend nvidia-smi 2>/dev/null

echo ""
echo "=== Checking Environment Variables in Container ==="
docker exec engabusami-backend-dev env | grep NVIDIA 2>/dev/null || docker exec engabusami-backend env | grep NVIDIA 2>/dev/null

echo ""
echo "=== Checking PyTorch GPU Detection ==="
docker exec engabusami-backend-dev python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || docker exec engabusami-backend python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null


