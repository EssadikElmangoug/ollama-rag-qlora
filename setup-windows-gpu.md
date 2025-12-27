# Windows GPU Setup Guide for Docker

This guide will help you set up GPU support for Docker on Windows using WSL2.

## Prerequisites

1. **Windows 10/11** with WSL2 installed
2. **NVIDIA GPU** with latest drivers installed in Windows
3. **Docker Desktop** with WSL2 backend enabled
4. **WSL2** with Ubuntu (or another Linux distribution)

## Step 1: Verify Windows NVIDIA Drivers

1. Open PowerShell as Administrator
2. Run: `nvidia-smi`
3. You should see your GPU information. If not, download and install NVIDIA drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)

## Step 2: Install WSL2 (if not already installed)

1. Open PowerShell as Administrator
2. Run:
   ```powershell
   wsl --install
   ```
3. Restart your computer if prompted

## Step 3: Install NVIDIA Container Toolkit in WSL2

1. **Open WSL2 terminal** (Ubuntu)
2. **Add NVIDIA package repositories:**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```

3. **Install NVIDIA Container Toolkit:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   ```

4. **Configure Docker to use NVIDIA runtime:**
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```

5. **Restart Docker Desktop** (close and reopen Docker Desktop application)

## Step 4: Verify GPU Access in WSL2

In WSL2 terminal, run:
```bash
nvidia-smi
```

You should see your GPU information. If this works, Docker can access your GPU.

## Step 5: Test Docker GPU Access

In WSL2 terminal, run:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see GPU information from inside the container. If this works, you're all set!

## Step 6: Run Your Application

Now you can run your application with GPU support:

```bash
docker-compose up -d
```

Or if you're in PowerShell/CMD:
```powershell
docker-compose up -d
```

## Troubleshooting

### Issue: "could not select device driver nvidia"

**Solution:** NVIDIA Container Toolkit is not installed or Docker Desktop hasn't been restarted.

1. Make sure you completed Step 3 above
2. Restart Docker Desktop completely (quit and reopen)
3. Try the test command from Step 5 again

### Issue: "nvidia-smi" not found in WSL2

**Solution:** You need to install NVIDIA drivers in WSL2. The drivers should be automatically available if:
- You have NVIDIA drivers installed in Windows
- WSL2 is properly configured

Try:
```bash
# In WSL2
sudo apt-get update
sudo apt-get install -y nvidia-driver-535  # or latest version
```

### Issue: Docker Desktop not using WSL2

**Solution:**
1. Open Docker Desktop
2. Go to Settings → General
3. Check "Use the WSL 2 based engine"
4. Apply & Restart

### Issue: CUDA not available in container

**Solution:** Make sure:
1. Your GPU supports CUDA (check NVIDIA website)
2. You're using the correct CUDA version in Dockerfile (12.1.0)
3. NVIDIA Container Toolkit is properly installed

## Alternative: Use CPU Mode

If you can't get GPU working, you can run in CPU mode:

```bash
docker-compose -f docker-compose.no-gpu.yml up -d
```

Note: This will be much slower for ML operations.

## Verify GPU in Your Application

After starting the containers, check the logs:

```bash
docker logs engabusami-backend
```

You should see:
```
[GPU Check] CUDA is available. Found X GPU(s)
```

If you see "CUDA is not available", GPU is not properly configured.



