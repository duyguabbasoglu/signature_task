#!/bin/bash
# Usage: ./scripts/test_llm.sh [message]

set -e

# config 
LLM_ENDPOINT="${BBOX_LLM_ENDPOINT:-https://common-inference-apis.test-turkcelltech.ai/gpt-oss-120b/v1}"
LLM_MODEL="${BBOX_LLM_MODEL:-gpt-oss-120b}"
API_KEY="${BBOX_LLM_API_KEY:-uavPCHhER6/EZnQFc2JafwjyqkcPE0oL6sowlCWsLGw=}"

# mesg
MESSAGE="${1:-Hello, this is a test message.}"

echo "LLM Endpoint Test"
echo "Endpoint: $LLM_ENDPOINT"
echo "Model: $LLM_MODEL"
echo "Message: $MESSAGE"


response=$(curl -s -k --connect-timeout 15 --max-time 60 \
    --location "${LLM_ENDPOINT}/chat/completions" \
    --header 'accept: application/json' \
    --header "Authorization: Bearer ${API_KEY}" \
    --header 'Content-Type: application/json' \
    --data "{
        \"messages\": [
            {\"role\": \"system\", \"content\": \"You are a helpful assistant.\"},
            {\"role\": \"user\", \"content\": \"${MESSAGE}\"}
        ],
        \"model\": \"${LLM_MODEL}\"
    }" 2>&1)

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "❌ Connection failed (exit code: $exit_code)"
    exit $exit_code
fi

echo "Response:"
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
echo ""
