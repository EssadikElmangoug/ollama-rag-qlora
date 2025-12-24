# Docker Setup Guide

This guide explains how to build and run the Engabusami project using Docker.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- (Optional) NVIDIA Docker for GPU support
- (Optional) Ollama installed locally or in Docker

## Quick Start

### Production Build

1. **Build and start all services:**
   ```bash
   docker-compose up -d --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000
   - API Health: http://localhost:5000/api/health

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Stop services:**
   ```bash
   docker-compose down
   ```

### Development Build

For development with hot reload:

```bash
docker-compose -f docker-compose.dev.yml up --build
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Backend API URL (for frontend)
NEXT_PUBLIC_BACKEND_URL=http://localhost:5000

# Ollama Base URL (for backend)
OLLAMA_BASE_URL=http://localhost:11434
# Or if using Ollama in Docker:
# OLLAMA_BASE_URL=http://ollama:11434
```

### GPU Support

To enable GPU support for QLoRA training:

1. **Install NVIDIA Docker:**
   ```bash
   # Follow instructions at: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```

2. **Uncomment GPU sections in `docker-compose.yml`:**
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

3. **Rebuild and restart:**
   ```bash
   docker-compose up -d --build
   ```

### Running Ollama in Docker

To run Ollama as a Docker service:

1. **Uncomment the Ollama service in `docker-compose.yml`**

2. **Update `OLLAMA_BASE_URL` in `.env`:**
   ```env
   OLLAMA_BASE_URL=http://ollama:11434
   ```

3. **Start services:**
   ```bash
   docker-compose up -d
   ```

4. **Pull models in Ollama container:**
   ```bash
   docker exec -it engabusami-ollama ollama pull llama3.2:3b
   ```

## Services

### Backend (Flask API)
- **Port:** 5000
- **Health Check:** `/api/health`
- **Volumes:**
  - `./backend/uploads` - Uploaded documents
  - `./backend/qlora_models` - Fine-tuned models
  - `./backend/faiss_index` - Vector store index

### Frontend (Next.js)
- **Port:** 3000
- **Environment:** Production (standalone build)
- **Dependencies:** Backend service

### Ollama (Optional)
- **Port:** 11434
- **Volume:** `ollama-data` (persists models)

## Building Individual Services

### Backend Only
```bash
cd backend
docker build -t engabusami-backend .
docker run -p 5000:5000 engabusami-backend
```

### Frontend Only
```bash
docker build -t engabusami-frontend .
docker run -p 3000:3000 engabusami-frontend
```

## Troubleshooting

### Backend Issues

1. **Check backend logs:**
   ```bash
   docker-compose logs backend
   ```

2. **Verify GPU access (if using GPU):**
   ```bash
   docker exec engabusami-backend nvidia-smi
   ```

3. **Check Ollama connection:**
   ```bash
   docker exec engabusami-backend curl http://ollama:11434/api/tags
   ```

### Frontend Issues

1. **Check frontend logs:**
   ```bash
   docker-compose logs frontend
   ```

2. **Verify backend connection:**
   ```bash
   docker exec engabusami-frontend wget -O- http://backend:5000/api/health
   ```

### Common Issues

**Port already in use:**
- Change ports in `docker-compose.yml` or stop conflicting services

**Permission denied:**
- Ensure Docker has proper permissions
- On Linux: `sudo usermod -aG docker $USER`

**Out of memory:**
- Reduce batch sizes in training configuration
- Use CPU-only build (remove GPU requirements)

**Model download fails:**
- Check internet connection
- Increase Docker memory limit
- Use volume mounts for model persistence

## Data Persistence

All important data is persisted via Docker volumes:
- **Uploads:** `./backend/uploads`
- **Models:** `./backend/qlora_models`
- **Vector Store:** `./backend/faiss_index`
- **Ollama Models:** `ollama-data` (if using Ollama service)

To backup:
```bash
# Backup all data
tar -czf backup.tar.gz backend/uploads backend/qlora_models backend/faiss_index
```

## Production Deployment

For production:

1. **Use production compose file:**
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

2. **Set proper environment variables**

3. **Use reverse proxy (nginx/traefik) for HTTPS**

4. **Enable resource limits in docker-compose.yml**

5. **Set up log rotation**

6. **Configure backups for volumes**

## Clean Up

**Remove all containers and volumes:**
```bash
docker-compose down -v
```

**Remove images:**
```bash
docker rmi engabusami-backend engabusami-frontend
```

**Full cleanup (careful!):**
```bash
docker system prune -a --volumes
```







