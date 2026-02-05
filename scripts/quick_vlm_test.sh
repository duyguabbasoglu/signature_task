#!/bin/bash
# Quick VLM test script
# Run from project root: ./scripts/quick_vlm_test.sh

set -e

# Load .env if exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Loaded .env"
fi

# Check API key
if [ -z "$BBOX_LLM_API_KEY" ]; then
    echo "❌ BBOX_LLM_API_KEY not set"
    echo "Set it with: export BBOX_LLM_API_KEY='your-key'"
    exit 1
fi

ENDPOINT="${BBOX_LLM_ENDPOINT:-https://common-inference-apis.turkcelltech.ai/gpt-oss-120b/v1}"

echo "Testing VLM endpoint: $ENDPOINT"
echo "Testing with simple prompt..."
echo ""

curl --location "${ENDPOINT}/chat/completions" \
  --connect-timeout 10 \
  --max-time 30 \
  --header 'accept: application/json' \
  --header "Authorization: Bearer ${BBOX_LLM_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data '{
    "messages": [
      {"role": "system", "content": "Reply with one word only."},
      {"role": "user", "content": "Say OK"}
    ],
    "model": "gpt-oss-120b",
    "temperature": 0.1
  }' 2>/dev/null

echo ""
echo ""
echo "✓ VLM test complete"
