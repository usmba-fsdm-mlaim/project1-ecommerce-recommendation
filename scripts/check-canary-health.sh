#!/bin/bash

# Check canary deployment health
# Returns 0 if canary is healthy, 1 otherwise

set -e

NAMESPACE=${NAMESPACE:-"production"}
CANARY_DEPLOYMENT="recommendation-api-canary"
THRESHOLD_ERROR_RATE=0.05  # 5%
THRESHOLD_LATENCY_P95=500  # 500ms

echo "🔍 Checking canary deployment health..."
echo "======================================="

# Check if canary deployment exists
if ! kubectl get deployment "$CANARY_DEPLOYMENT" -n "$NAMESPACE" &> /dev/null; then
    echo "❌ Canary deployment not found"
    exit 1
fi

# Check pod status
echo "1. Checking pod status..."
READY_PODS=$(kubectl get deployment "$CANARY_DEPLOYMENT" -n "$NAMESPACE" \
    -o jsonpath='{.status.readyReplicas}')
DESIRED_PODS=$(kubectl get deployment "$CANARY_DEPLOYMENT" -n "$NAMESPACE" \
    -o jsonpath='{.spec.replicas}')

if [ "$READY_PODS" != "$DESIRED_PODS" ]; then
    echo "❌ Not all pods ready ($READY_PODS/$DESIRED_PODS)"
    exit 1
fi
echo "✅ All pods ready ($READY_PODS/$DESIRED_PODS)"

# Check error rate from Prometheus
echo "2. Checking error rate..."
PROMETHEUS_URL=${PROMETHEUS_URL:-"http://prometheus:9090"}

# Query error rate for canary
ERROR_RATE=$(curl -s "$PROMETHEUS_URL/api/v1/query" \
    --data-urlencode "query=rate(api_requests_total{version=\"canary\",status=~\"5..\"}[5m]) / rate(api_requests_total{version=\"canary\"}[5m])" \
    | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

# Convert to float comparison
if [ "$(echo "$ERROR_RATE > $THRESHOLD_ERROR_RATE" | bc -l)" -eq 1 ]; then
    echo "❌ Error rate too high: $ERROR_RATE (threshold: $THRESHOLD_ERROR_RATE)"
    exit 1
fi
echo "✅ Error rate acceptable: $ERROR_RATE"

# Check latency from Prometheus
echo "3. Checking latency..."
LATENCY_P95=$(curl -s "$PROMETHEUS_URL/api/v1/query" \
    --data-urlencode "query=histogram_quantile(0.95, rate(api_request_duration_seconds_bucket{version=\"canary\"}[5m]))" \
    | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

# Convert to milliseconds
LATENCY_MS=$(echo "$LATENCY_P95 * 1000" | bc -l)

if [ "$(echo "$LATENCY_MS > $THRESHOLD_LATENCY_P95" | bc -l)" -eq 1 ]; then
    echo "❌ Latency too high: ${LATENCY_MS}ms (threshold: ${THRESHOLD_LATENCY_P95}ms)"
    exit 1
fi
echo "✅ Latency acceptable: ${LATENCY_MS}ms"

# Check for restart loops
echo "4. Checking for restart loops..."
RESTART_COUNT=$(kubectl get pods -n "$NAMESPACE" \
    -l app=recommendation-api,version=canary \
    -o jsonpath='{.items[*].status.containerStatuses[0].restartCount}' \
    | awk '{sum+=$1} END {print sum}')

if [ "$RESTART_COUNT" -gt 3 ]; then
    echo "❌ Too many restarts: $RESTART_COUNT"
    exit 1
fi
echo "✅ Restart count acceptable: $RESTART_COUNT"

# Check resource usage
echo "5. Checking resource usage..."
CPU_USAGE=$(kubectl top pods -n "$NAMESPACE" \
    -l app=recommendation-api,version=canary \
    | awk 'NR>1 {gsub(/m/,"",$2); print $2}' \
    | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

MEMORY_USAGE=$(kubectl top pods -n "$NAMESPACE" \
    -l app=recommendation-api,version=canary \
    | awk 'NR>1 {gsub(/Mi/,"",$3); print $3}' \
    | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')

echo "   CPU: ${CPU_USAGE}m"
echo "   Memory: ${MEMORY_USAGE}Mi"

if [ "$(echo "$CPU_USAGE > 800" | bc -l)" -eq 1 ]; then
    echo "⚠️  High CPU usage"
fi

if [ "$(echo "$MEMORY_USAGE > 900" | bc -l)" -eq 1 ]; then
    echo "⚠️  High memory usage"
fi

# Run smoke tests on canary
echo "6. Running smoke tests..."
CANARY_SERVICE_URL=$(kubectl get service recommendation-api-canary-service -n "$NAMESPACE" \
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "canary-service")

if [ -f "./scripts/smoke-test.sh" ]; then
    if ./scripts/smoke-test.sh "http://$CANARY_SERVICE_URL"; then
        echo "✅ Smoke tests passed"
    else
        echo "❌ Smoke tests failed"
        exit 1
    fi
else
    echo "⚠️  Smoke test script not found, skipping"
fi

echo "======================================="
echo "🎉 Canary deployment is healthy!"
exit 0