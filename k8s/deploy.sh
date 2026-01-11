#!/bin/bash

# Deployment script for Kubernetes
# Usage: ./deploy.sh [environment]

set -e

ENVIRONMENT=${1:-production}
NAMESPACE="ecommerce-recommendation"

echo "🚀 Deploying to Kubernetes - Environment: $ENVIRONMENT"
echo "=================================================="

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if kustomize is installed (optional, kubectl has built-in support)
if ! command -v kustomize &> /dev/null; then
    echo "⚠️  kustomize not found, using kubectl's built-in kustomize"
fi

# Create namespace if it doesn't exist
echo "📦 Creating namespace..."
kubectl apply -f namespace.yaml

# Apply base configurations
echo "⚙️  Applying ConfigMap and Secrets..."
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

# Apply application deployment
echo "🚢 Deploying application..."
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml

# Apply ingress (optional, comment out if not using ingress)
echo "🌐 Applying Ingress..."
kubectl apply -f ingress.yaml

# Apply monitoring stack
echo "📊 Deploying monitoring stack..."
kubectl apply -f monitoring/prometheus-deployment.yaml
kubectl apply -f monitoring/grafana-deployment.yaml
kubectl apply -f monitoring/service-monitor.yaml
kubectl apply -f monitoring/prometheus-rules.yaml

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/recommendation-api -n $NAMESPACE || true
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n $NAMESPACE || true
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n $NAMESPACE || true

# Show deployment status
echo ""
echo "✅ Deployment completed!"
echo "=================================================="
echo "📊 Deployment Status:"
kubectl get deployments -n $NAMESPACE
echo ""
echo "🔍 Pods Status:"
kubectl get pods -n $NAMESPACE
echo ""
echo "🌐 Services:"
kubectl get services -n $NAMESPACE
echo ""
echo "📈 HPA Status:"
kubectl get hpa -n $NAMESPACE
echo ""
echo "🔗 Access Points:"
echo "  - API: kubectl port-forward -n $NAMESPACE svc/recommendation-api-service 8000:80"
echo "  - Prometheus: kubectl port-forward -n $NAMESPACE svc/prometheus-service 9090:9090"
echo "  - Grafana: kubectl port-forward -n $NAMESPACE svc/grafana-service 3000:3000"
echo ""
echo "📝 To view logs:"
echo "  kubectl logs -f deployment/recommendation-api -n $NAMESPACE"

