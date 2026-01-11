#!/bin/bash

# Integration tests for deployed API
# Usage: ./integration-test.sh <base_url>

set -e

BASE_URL=${1:-"http://localhost:8000"}
FAILED=0

echo "🔬 Running integration tests against $BASE_URL"
echo "==========================================="

# Test 1: Full recommendation flow
echo "Test 1: Full recommendation flow..."
RESPONSE=$(curl -s -X POST "$BASE_URL/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id": 0, "n": 10}')

if echo "$RESPONSE" | grep -q "user_id"; then
    echo "✅ Full recommendation flow passed"
else
    echo "❌ Full recommendation flow failed"
    echo "Response: $RESPONSE"
    FAILED=$((FAILED + 1))
fi

# Test 2: Response structure validation
echo "Test 2: Response structure validation..."
if echo "$RESPONSE" | grep -q "recommendations" && \
   echo "$RESPONSE" | grep -q "generated_at"; then
    echo "✅ Response structure validation passed"
else
    echo "❌ Response structure validation failed"
    FAILED=$((FAILED + 1))
fi

# Test 3: Recommendation quality
echo "Test 3: Recommendation quality check..."
NUM_RECS=$(echo "$RESPONSE" | grep -o "product_id" | wc -l)
if [ "$NUM_RECS" -gt 0 ]; then
    echo "✅ Recommendation quality check passed ($NUM_RECS recommendations)"
else
    echo "❌ Recommendation quality check failed (0 recommendations)"
    FAILED=$((FAILED + 1))
fi

# Test 4: Different user IDs
echo "Test 4: Different user recommendations..."
RESPONSE1=$(curl -s -X POST "$BASE_URL/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id": 0, "n": 5}')

RESPONSE2=$(curl -s -X POST "$BASE_URL/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id": 1, "n": 5}')

if [ "$RESPONSE1" != "$RESPONSE2" ]; then
    echo "✅ Different user recommendations passed"
else
    echo "⚠️  Different user recommendations identical (may be cold start)"
fi

# Test 5: Varying recommendation count
echo "Test 5: Varying recommendation count..."
for n in 1 5 10 20; do
    RESPONSE=$(curl -s -X POST "$BASE_URL/recommend" \
        -H "Content-Type: application/json" \
        -d "{\"user_id\": 0, \"n\": $n}")
    
    COUNT=$(echo "$RESPONSE" | grep -o "product_id" | wc -l)
    if [ "$COUNT" -le "$n" ]; then
        echo "✅ n=$n: returned $COUNT recommendations"
    else
        echo "❌ n=$n: returned $COUNT recommendations (expected <=$n)"
        FAILED=$((FAILED + 1))
    fi
done

# Test 6: Concurrent requests
echo "Test 6: Concurrent request handling..."
PIDS=()
for i in {1..10}; do
    curl -s -X POST "$BASE_URL/recommend" \
        -H "Content-Type: application/json" \
        -d '{"user_id": 0, "n": 5}' > /dev/null &
    PIDS+=($!)
done

# Wait for all requests
ALL_SUCCESS=true
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        ALL_SUCCESS=false
    fi
done

if $ALL_SUCCESS; then
    echo "✅ Concurrent request handling passed"
else
    echo "❌ Concurrent request handling failed"
    FAILED=$((FAILED + 1))
fi

# Test 7: API performance
echo "Test 7: API performance check..."
START=$(date +%s%N)
curl -s -X POST "$BASE_URL/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id": 0, "n": 10}' > /dev/null
END=$(date +%s%N)
DURATION=$((($END - $START) / 1000000))  # Convert to milliseconds

if [ "$DURATION" -lt 5000 ]; then
    echo "✅ API performance check passed (${DURATION}ms)"
else
    echo "⚠️  API performance slow (${DURATION}ms)"
fi

# Test 8: Health check stability
echo "Test 8: Health check stability..."
SUCCESS_COUNT=0
for i in {1..10}; do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
    if [ "$RESPONSE" -eq 200 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
    sleep 0.1
done

if [ "$SUCCESS_COUNT" -eq 10 ]; then
    echo "✅ Health check stability passed (10/10)"
else
    echo "❌ Health check stability failed ($SUCCESS_COUNT/10)"
    FAILED=$((FAILED + 1))
fi

echo "==========================================="
if [ $FAILED -eq 0 ]; then
    echo "🎉 All integration tests passed!"
    exit 0
else
    echo "❌ $FAILED test(s) failed"
    exit 1
fi