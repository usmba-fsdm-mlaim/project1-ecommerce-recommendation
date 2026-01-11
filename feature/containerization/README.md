# Docker Containerization

## Branch: feature/containerization

This branch containerizes the entire application stack with Docker and Docker Compose, including MLflow, Prometheus, and Grafana.

## Features

- ✅ Multi-stage Docker build for optimization
- ✅ Complete docker-compose stack
- ✅ MLflow tracking server
- ✅ Prometheus metrics collection
- ✅ Grafana visualization
- ✅ Nginx web server
- ✅ Volume persistence
- ✅ Health checks
- ✅ Auto-restart policies

## Files

```
feature/containerization/
├── Dockerfile              # Multi-stage build
├── docker-compose.yml      # Complete stack
├── .dockerignore          # Build exclusions
├── nginx.conf             # Nginx configuration
└── README.md              # This file
```

## Quick Start

### Build and Run

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Verify Services

```bash
# Check running containers
docker-compose ps

# Check logs for specific service
docker-compose logs api
docker-compose logs mlflow
```

## Services

### 1. Recommendation API

**Port**: 8000

Main application service serving recommendations.

```bash
# Access API
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

**Environment Variables**:
- `MODEL_PATH=/app/models/recommendation_model.pkl`
- `DATA_PATH=/app/data/cleaned_data.csv`
- `LOG_LEVEL=INFO`

**Volumes**:
- `./models:/app/models` - Model files
- `./data:/app/data` - Data files
- `./logs:/app/logs` - Application logs

### 2. Web Interface

**Port**: 80

Nginx server hosting the web interface.

```bash
# Access web interface
open http://localhost

# View logs
docker-compose logs -f web
```

**Volumes**:
- `./static:/usr/share/nginx/html` - Static files
- `./nginx.conf:/etc/nginx/nginx.conf` - Configuration

### 3. MLflow Tracking Server

**Port**: 5000

Experiment tracking and model registry.

```bash
# Access MLflow UI
open http://localhost:5000

# View logs
docker-compose logs -f mlflow
```

**Features**:
- Experiment tracking
- Model versioning
- Artifact storage
- Model comparison

**Volumes**:
- `./mlruns:/mlflow/mlruns` - Experiment data
- `./mlartifacts:/mlflow/mlartifacts` - Model artifacts

### 4. Prometheus

**Port**: 9090

Metrics collection and monitoring.

```bash
# Access Prometheus UI
open http://localhost:9090

# Check targets
open http://localhost:9090/targets

# View logs
docker-compose logs -f prometheus
```

**Configuration**: `prometheus.yml`

**Volumes**:
- `./prometheus.yml:/etc/prometheus/prometheus.yml` - Config
- `prometheus-data:/prometheus` - Time-series data

### 5. Grafana

**Port**: 3000
**Credentials**: admin/admin

Data visualization and dashboards.

```bash
# Access Grafana
open http://localhost:3000

# View logs
docker-compose logs -f grafana
```

**Features**:
- Pre-configured Prometheus datasource
- Custom dashboards
- Alert management

**Volumes**:
- `grafana-data:/var/lib/grafana` - Dashboard data
- `./grafana/dashboards:/etc/grafana/provisioning/dashboards`
- `./grafana/datasources:/etc/grafana/provisioning/datasources`

## Docker Image

### Multi-Stage Build

The Dockerfile uses multi-stage builds for optimization:

**Stage 1: Builder**
- Installs build dependencies
- Compiles Python packages
- Creates wheel files

**Stage 2: Runtime**
- Minimal Python slim image
- Copies only necessary files
- Smaller final image size

### Build Optimization

```dockerfile
# Size comparison
FROM python:3.10          # ~900MB
FROM python:3.10-slim     # ~150MB (used)
```

### Build Manually

```bash
# Build image
docker build -t recommendation-api:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  recommendation-api:latest
```

## Docker Compose Architecture

```
┌─────────────────────────────────────────┐
│            Load Balancer (Nginx)        │
│              Port: 80                   │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴───────────┐
    │                        │
┌───▼────┐            ┌──────▼─────┐
│  Web   │            │    API     │
│  :80   │            │   :8000    │
└────────┘            └──────┬─────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
       ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
       │ MLflow  │    │Prometheus │   │  Grafana  │
       │  :5000  │    │   :9090   │   │   :3000   │
       └─────────┘    └───────────┘   └───────────┘
```

## Volume Management

### Named Volumes

Created and managed by Docker:
- `prometheus-data`: Metrics time-series data
- `grafana-data`: Dashboard configurations

### Bind Mounts

Mapped to host filesystem:
- `./models`: Model files
- `./data`: Training data
- `./logs`: Application logs
- `./mlruns`: MLflow experiments
- `./mlartifacts`: Model artifacts

### Backup Volumes

```bash
# Backup Prometheus data
docker run --rm \
  -v prometheus-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/prometheus-backup.tar.gz /data

# Backup Grafana data
docker run --rm \
  -v grafana-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/grafana-backup.tar.gz /data
```

### Restore Volumes

```bash
# Restore Prometheus data
docker run --rm \
  -v prometheus-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/prometheus-backup.tar.gz -C /

# Restore Grafana data
docker run --rm \
  -v grafana-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/grafana-backup.tar.gz -C /
```

## Networking

### Network Configuration

Docker Compose creates a bridge network `app-network` for all services.

```yaml
networks:
  app-network:
    driver: bridge
```

### Service Communication

Services communicate using service names:
- API accessible at: `http://api:8000`
- MLflow at: `http://mlflow:5000`
- Prometheus at: `http://prometheus:9090`

### DNS Resolution

Docker provides automatic DNS resolution:
```python
# In API code
MLFLOW_TRACKING_URI = "http://mlflow:5000"
```

## Health Checks

### API Health Check

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Check Health Status

```bash
# All services
docker-compose ps

# Specific service
docker inspect --format='{{.State.Health.Status}}' recommendation-api
```

## Environment Configuration

### Development

```bash
# .env.dev
LOG_LEVEL=DEBUG
WORKERS=2
```

```bash
docker-compose --env-file .env.dev up
```

### Production

```bash
# .env.prod
LOG_LEVEL=INFO
WORKERS=4
```

```bash
docker-compose --env-file .env.prod up
```

## Scaling

### Scale API Instances

```bash
# Scale to 3 replicas
docker-compose up -d --scale api=3

# Verify
docker-compose ps
```

### Load Balancing

Add load balancer in docker-compose.yml:

```yaml
nginx:
  image: nginx:alpine
  depends_on:
    - api
  volumes:
    - ./nginx-lb.conf:/etc/nginx/nginx.conf
```

## Monitoring

### Container Metrics

```bash
# Resource usage
docker stats

# Specific container
docker stats recommendation-api
```

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api

# Since timestamp
docker-compose logs --since=2024-01-01T00:00:00 api
```

## Troubleshooting

### Issue: Container Won't Start

```bash
# Check logs
docker-compose logs api

# Check container status
docker-compose ps

# Rebuild
docker-compose build --no-cache api
docker-compose up -d
```

### Issue: Volume Permission Errors

```bash
# Fix permissions
chmod -R 755 models/ data/ logs/

# Or run as root
docker-compose run --user root api bash
```

### Issue: Port Already in Use

```bash
# Check port usage
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

### Issue: Out of Disk Space

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything
docker system prune -a --volumes
```

## Performance Tuning

### Optimize Build Time

```dockerfile
# Order layers from least to most frequently changing
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

### Reduce Image Size

```dockerfile
# Use slim base image
FROM python:3.10-slim

# Multi-stage build
FROM builder AS runtime

# Remove build dependencies
RUN apt-get purge -y gcc g++
```

### Memory Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

## CI/CD Integration

### Build in CI

```bash
# Build
docker build -t ghcr.io/your-org/recommendation-api:$VERSION .

# Push
docker push ghcr.io/your-org/recommendation-api:$VERSION

# Deploy
docker-compose pull
docker-compose up -d
```

## Next Steps

After containerization:
1. Move to `feature/ci-cd-pipeline` branch
2. Set up GitHub Actions
3. Automate builds and deployments

## Best Practices

✅ Use multi-stage builds
✅ Pin dependency versions
✅ Implement health checks
✅ Use named volumes for persistence
✅ Set resource limits
✅ Enable auto-restart
✅ Use secrets for sensitive data
✅ Regular security updates

## Security

### Scan for Vulnerabilities

```bash
# Install trivy
brew install trivy

# Scan image
trivy image recommendation-api:latest
```

### Non-Root User

```dockerfile
# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser
```

### Secrets Management

```yaml
secrets:
  db_password:
    file: ./secrets/db_password.txt

services:
  api:
    secrets:
      - db_password
```

## Dependencies

See `requirements.txt` and `Dockerfile`