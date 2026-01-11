# Kubernetes Deployment with Monitoring and Auto-Scaling

This directory contains all Kubernetes manifests for deploying the E-commerce Recommendation API with comprehensive monitoring and auto-scaling capabilities.

## 📁 Directory Structure

```
k8s/
├── namespace.yaml              # Kubernetes namespace
├── configmap.yaml              # Application configuration
├── secret.yaml                 # Secrets (API keys, etc.)
├── deployment.yaml             # Main application deployment
├── service.yaml                # Kubernetes service
├── hpa.yaml                    # Horizontal Pod Autoscaler
├── ingress.yaml                # Ingress configuration
├── kustomization.yaml          # Kustomize configuration
├── deploy.sh                   # Deployment script
├── undeploy.sh                 # Undeployment script
└── monitoring/
    ├── prometheus-deployment.yaml  # Prometheus monitoring
    ├── grafana-deployment.yaml     # Grafana dashboards
    ├── service-monitor.yaml        # Prometheus ServiceMonitor
    └── prometheus-rules.yaml       # Alerting rules
```

## 🚀 Quick Start

### Prerequisites

1. **Kubernetes Cluster**: A running Kubernetes cluster (minikube, kind, GKE, EKS, AKS, etc.)
2. **kubectl**: Kubernetes command-line tool installed and configured
3. **Docker Image**: The application Docker image should be available in your cluster's registry

### Build and Push Docker Image

```bash
# Build the Docker image
cd api
docker build -t recommendation-api:latest .

# Tag for your registry (replace with your registry)
docker tag recommendation-api:latest your-registry/recommendation-api:latest

# Push to registry
docker push your-registry/recommendation-api:latest
```

### Deploy to Kubernetes

#### Option 1: Using the deployment script

```bash
# Make script executable
chmod +x deploy.sh

# Deploy everything
./deploy.sh
```

#### Option 2: Manual deployment

```bash
# Create namespace
kubectl apply -f namespace.yaml

# Apply configurations
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

# Deploy application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml

# Deploy monitoring
kubectl apply -f monitoring/prometheus-deployment.yaml
kubectl apply -f monitoring/grafana-deployment.yaml
kubectl apply -f monitoring/service-monitor.yaml
kubectl apply -f monitoring/prometheus-rules.yaml

# Deploy ingress (optional)
kubectl apply -f ingress.yaml
```

#### Option 3: Using Kustomize

```bash
kubectl apply -k .
```

## 📊 Monitoring Setup

### Prometheus

Prometheus is configured to scrape metrics from:
- Kubernetes API server
- Kubernetes nodes
- Recommendation API pods
- All pods with `prometheus.io/scrape: "true"` annotation

**Access Prometheus:**
```bash
kubectl port-forward -n ecommerce-recommendation svc/prometheus-service 9090:9090
```
Then open http://localhost:9090 in your browser.

### Grafana

Grafana comes pre-configured with:
- Prometheus datasource
- Recommendation API dashboard

**Access Grafana:**
```bash
kubectl port-forward -n ecommerce-recommendation svc/grafana-service 3000:3000
```
Then open http://localhost:3000
- Username: `admin`
- Password: `admin` (change in production!)

### Metrics Exposed

**Note**: The current setup monitors Kubernetes-level metrics (CPU, memory, pod status, etc.) but does not expose custom application metrics. Prometheus will collect:

- Pod CPU and memory usage
- Pod status and health
- Kubernetes node metrics
- Container resource utilization

If you want to add application-level metrics (request rate, response time, etc.), you can add a Prometheus client library to your FastAPI application and expose a `/metrics` endpoint.

## 🔄 Auto-Scaling

The Horizontal Pod Autoscaler (HPA) is configured with:

- **Min Replicas**: 3
- **Max Replicas**: 10
- **CPU Threshold**: 70% utilization
- **Memory Threshold**: 80% utilization
- **Note**: Custom metrics (like HTTP requests per second) require application-level metrics to be exposed. Currently, HPA scales based on CPU and memory only.

**Check HPA status:**
```bash
kubectl get hpa -n ecommerce-recommendation
kubectl describe hpa recommendation-api-hpa -n ecommerce-recommendation
```

## 🌐 Ingress Configuration

The ingress is configured for:
- Host: `api.recommendation.example.com` (update in production)
- TLS/SSL support
- Rate limiting: 100 requests per second

**Note**: Update the hostname in `ingress.yaml` and ensure you have:
- NGINX Ingress Controller installed
- cert-manager for TLS certificates (optional)

## 🔍 Health Checks

The deployment includes three types of probes:

1. **Startup Probe**: Ensures the container has started
2. **Liveness Probe**: Restarts the container if it becomes unresponsive
3. **Readiness Probe**: Determines if the pod is ready to receive traffic

All probes use the `/health` endpoint.

## 📈 Resource Limits

Each pod has:
- **Requests**: 256Mi memory, 250m CPU
- **Limits**: 512Mi memory, 500m CPU

Adjust these based on your application's needs.

## 🚨 Alerts

Prometheus is configured with the following alert rules:

- **HighErrorRate**: Triggers when error rate > 5%
- **HighResponseTime**: Triggers when p95 response time > 1s
- **PodCrashLooping**: Triggers when pod restarts frequently
- **HighCPUUsage**: Triggers when CPU usage > 80%
- **HighMemoryUsage**: Triggers when memory usage > 90%
- **PodNotReady**: Triggers when pod is not ready for > 5 minutes

## 🔧 Configuration

### Environment Variables

Edit `configmap.yaml` to modify:
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `MODEL_VERSION`: Model version identifier
- `WORKERS`: Number of uvicorn workers

### Secrets

Add sensitive data to `secret.yaml`:
- API keys
- Database URLs
- Other credentials

## 🧪 Testing the Deployment

```bash
# Check all pods are running
kubectl get pods -n ecommerce-recommendation

# Check service endpoints
kubectl get endpoints -n ecommerce-recommendation

# Test the API
kubectl port-forward -n ecommerce-recommendation svc/recommendation-api-service 8000:80
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "viewed_products": [1, 2, 3]}'

# View logs
kubectl logs -f deployment/recommendation-api -n ecommerce-recommendation
```

## 🗑️ Undeploy

To remove all resources:

```bash
# Using script
./undeploy.sh

# Or manually
kubectl delete -f monitoring/prometheus-rules.yaml
kubectl delete -f monitoring/service-monitor.yaml
kubectl delete -f monitoring/grafana-deployment.yaml
kubectl delete -f monitoring/prometheus-deployment.yaml
kubectl delete -f ingress.yaml
kubectl delete -f hpa.yaml
kubectl delete -f service.yaml
kubectl delete -f deployment.yaml
kubectl delete -f secret.yaml
kubectl delete -f configmap.yaml
kubectl delete -f namespace.yaml
```

## 📝 Notes

1. **Image Pull Policy**: Currently set to `IfNotPresent`. For production, use a proper image registry and set to `Always`.

2. **Storage**: Prometheus and Grafana use `emptyDir` volumes. For production, use PersistentVolumes.

3. **Security**: 
   - Update default Grafana credentials
   - Use proper secrets management (e.g., Sealed Secrets, External Secrets Operator)
   - Enable RBAC properly
   - Use network policies

4. **Monitoring**: For production, consider using:
   - Prometheus Operator
   - Grafana Operator
   - AlertManager for alert routing

5. **Service Mesh**: Consider adding Istio or Linkerd for advanced traffic management and observability.

## 🔗 Useful Commands

```bash
# Scale deployment manually
kubectl scale deployment recommendation-api --replicas=5 -n ecommerce-recommendation

# Update deployment
kubectl set image deployment/recommendation-api recommendation-api=recommendation-api:v2 -n ecommerce-recommendation

# Rollback deployment
kubectl rollout undo deployment/recommendation-api -n ecommerce-recommendation

# View rollout status
kubectl rollout status deployment/recommendation-api -n ecommerce-recommendation

# Get events
kubectl get events -n ecommerce-recommendation --sort-by='.lastTimestamp'
```

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

