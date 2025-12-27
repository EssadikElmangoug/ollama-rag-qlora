# GPU Setup - REQUIRED for This Project

This project **REQUIRES** GPU support to function properly. Follow these steps to enable GPU access in Docker.

## Quick Installation

```bash
# Make scripts executable
chmod +x install-gpu-support.sh check-gpu-setup.sh

# Install NVIDIA Container Toolkit (requires sudo)
sudo ./install-gpu-support.sh

# Verify installation
./check-gpu-setup.sh

# Start containers with GPU support
docker-compose up -d
```

## Manual Installation

If the script doesn't work, install manually:

### 1. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 2. Configure Docker

```bash
# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

### 3. Verify Installation

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this command shows your GPU, installation is successful!

## Troubleshooting

### Error: "nvidia-container-runtime: executable file not found"

**Solution:** Install NVIDIA Container Toolkit (see above)

### Error: "could not select device driver nvidia"

**Solution:** 
1. Verify installation: `./check-gpu-setup.sh`
2. Restart Docker: `sudo systemctl restart docker`
3. Check Docker config: `docker info | grep nvidia`

### Error: "nvidia-smi not found"

**Solution:** Install NVIDIA drivers first:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y nvidia-driver-XXX  # Replace XXX with your driver version

# Or use driver manager
sudo ubuntu-drivers autoinstall
```

### Docker daemon won't start after installation

**Solution:**
```bash
# Check Docker status
sudo systemctl status docker

# Check Docker logs
sudo journalctl -u docker

# Restart Docker
sudo systemctl restart docker
```

## Verification

After installation, verify everything works:

```bash
# Check GPU setup
./check-gpu-setup.sh

# Start containers
docker-compose up -d

# Check GPU in container
docker exec engabusami-backend nvidia-smi
```

## Important Notes

- **GPU 1 (NVIDIA) is used** - Intel GPU (GPU 0) is excluded
- **NVIDIA Container Toolkit is REQUIRED** - Without it, containers cannot start
- **Docker must be restarted** after installation
- **NVIDIA drivers must be installed** on the host system

## Support

If you continue to have issues:
1. Run `./check-gpu-setup.sh` and share the output
2. Check `docker info | grep nvidia`
3. Verify `nvidia-smi` works on the host
4. Check Docker logs: `sudo journalctl -u docker`



