# NVIDIA Container Toolkit Setup Guide

## The Problem
Error: `could not select device driver "nvidia" with capabilities: [[gpu]]`

This means Docker doesn't have access to your NVIDIA GPUs, even though `nvidia-smi` works on the host.

## Solution: Install NVIDIA Container Toolkit

### For Linux / WSL2:

1. **Add NVIDIA package repositories:**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

2. **Install NVIDIA Container Toolkit:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

3. **Configure Docker to use NVIDIA runtime:**
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

4. **Restart Docker daemon:**
   ```bash
   sudo systemctl restart docker
   ```

5. **Verify installation:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

### For Windows (Docker Desktop):

If you're using Docker Desktop on Windows:

1. **Enable WSL2 backend** (if not already enabled)
2. **Install NVIDIA Container Toolkit in WSL2** (follow Linux instructions above)
3. **Or use Docker Desktop with WSL2 integration**

### Alternative: Use Runtime Method

If the deploy method doesn't work, the docker-compose files have a fallback using `runtime: nvidia`. 
Uncomment that line and comment out the deploy section.

## Verify GPU Access

After installation, test GPU access:

```bash
# Test GPU in container
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Or test with your compose file
docker-compose up -d
docker exec engabusami-backend nvidia-smi
```

## Troubleshooting

1. **Check if nvidia runtime is available:**
   ```bash
   docker info | grep -i nvidia
   ```

2. **Check Docker daemon configuration:**
   ```bash
   cat /etc/docker/daemon.json
   ```
   Should contain:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

3. **If using WSL2, ensure NVIDIA drivers are installed in WSL2:**
   ```bash
   nvidia-smi
   ```



