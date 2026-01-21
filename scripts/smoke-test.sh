#!/bin/bash
set -e

BASE_URL=\
echo "?? Running smoke tests against \"

# Health check
curl -f "\/health"
echo "? Health check passed"

# Root endpoint
curl -f "\/"
echo "? Root endpoint passed"

# Recommendation endpoint
curl -f -X POST "\/recommend" \
  -H "Content-Type: application/json" \
  -d '{\"user_id\": 0, \"n\": 5}'
echo "? Recommendation endpoint passed"

echo "?? All smoke tests passed!"
