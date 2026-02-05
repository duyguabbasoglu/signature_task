#!/usr/bin/env python3
"""Test LLM API bağlantısını - basit text mesaj ile"""

import requests
import json

# Verilen parametreler (test: curl örneğinde production host kullanılıyor)
ENDPOINT = "https://common-inference-apis.turkcelltech.ai/gpt-oss-120b/v1"
API_KEY = "uavPCHhER6/EZnQFc2JafwjyqkcPE0oL6sowlCWsLGw="
MODEL = "gpt-oss-120b"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Test 1: Basit text mesaj (image olmadan)
print("=" * 80)
print("TEST 1: Basit text mesaj ile")
print("=" * 80)

payload = {
    "messages": [
        {
            "role": "system",
            "content": "You are helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    "model": MODEL,
    "temperature": 0.1,
    "max_tokens": 50
}

try:
    response = requests.post(
        f"{ENDPOINT}/chat/completions",
        headers=headers,
        json=payload,
        verify=False,
        timeout=30
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:500]}")
    if response.status_code == 200:
        data = response.json()
        print(f"SUCCESS: {data['choices'][0]['message']['content']}")
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: Imza vs noktalama sınıflandırması için
print("\n" + "=" * 80)
print("TEST 2: İmza sınıflandırması prompt ile")
print("=" * 80)

payload2 = {
    "messages": [
        {
            "role": "system",
            "content": "You are an image classification expert. Classify as SIGNATURE or PUNCTUATION."
        },
        {
            "role": "user",
            "content": "Is this a handwritten signature or simple punctuation mark? Answer only: SIGNATURE or PUNCTUATION"
        }
    ],
    "model": MODEL,
    "temperature": 0.1,
    "max_tokens": 10
}

try:
    response = requests.post(
        f"{ENDPOINT}/chat/completions",
        headers=headers,
        json=payload2,
        verify=False,
        timeout=30
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"SUCCESS: {data['choices'][0]['message']['content']}")
    else:
        print(f"FAILED: {response.text}")
except Exception as e:
    print(f"FAILED: {e}")
