#!/bin/bash

# Undeployment script for Kubernetes
# Usage: ./undeploy.sh

set -e

NAMESPACE="ecommerce-recommendation"

echo "🗑️  Undeploying from Kubernetes"
echo "=================================================="

# Delete resources
echo "🗑️  Deleting resources..."
kubectl delete -f monitoring/prometheus-rules.yaml --ignore-not-found=true
kubectl delete -f monitoring/service-monitor.yaml --ignore-not-found=true
kubectl delete -f monitoring/grafana-deployment.yaml --ignore-not-found=true
kubectl delete -f monitoring/prometheus-deployment.yaml --ignore-not-found=true
kubectl delete -f ingress.yaml --ignore-not-found=true
kubectl delete -f hpa.yaml --ignore-not-found=true
kubectl delete -f service.yaml --ignore-not-found=true
kubectl delete -f deployment.yaml --ignore-not-found=true
kubectl delete -f secret.yaml --ignore-not-found=true
kubectl delete -f configmap.yaml --ignore-not-found=true

# Delete namespace (optional, uncomment if you want to delete the entire namespace)
# echo "🗑️  Deleting namespace..."
# kubectl delete namespace $NAMESPACE --ignore-not-found=true

echo "✅ Undeployment completed!"

