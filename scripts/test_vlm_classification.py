#!/usr/bin/env python3
"""
VLM Test Script for Signature Detection

This script tests the VLM integration by classifying sample images.

Environment Variables Required:
- BBOX_LLM_API_KEY: Your API key for the VLM endpoint
- BBOX_LLM_ENDPOINT: (optional) VLM endpoint URL

Usage:
    # Set your API key
    export BBOX_LLM_API_KEY="your-api-key"
    
    # Run the test
    python scripts/test_vlm_classification.py
    
    # Or test specific images
    python scripts/test_vlm_classification.py data/B-S-95-F-02.tif data/punct_dot.png
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bbox_detector import is_signature, VLMClient, VLMResult
from bbox_detector.vlm_client import VLMClassification
import cv2


def test_vlm_connection() -> bool:
    """Test if VLM endpoint is accessible."""
    print("=" * 50)
    print("Testing VLM Connection")
    print("=" * 50)
    
    api_key = os.getenv("BBOX_LLM_API_KEY")
    if not api_key:
        print("❌ BBOX_LLM_API_KEY not set!")
        print("\nSet it with: export BBOX_LLM_API_KEY='your-key'")
        return False
    
    print(f"✓ API Key found: {api_key[:8]}...")
    
    endpoint = os.getenv("BBOX_LLM_ENDPOINT", "https://common-inference-apis.turkcelltech.ai/gpt-oss-120b/v1")
    print(f"✓ Endpoint: {endpoint}")
    
    # Create client and test
    client = VLMClient()
    print(f"✓ VLM Client initialized")
    print(f"  Model: {client.model}")
    print(f"  Enabled: {client.enabled}")
    
    return True


def classify_image(image_path: str, use_vlm: bool = True) -> dict:
    """Classify an image and return detailed results."""
    result = {
        "file": Path(image_path).name,
        "is_signature": None,
        "vlm_result": None,
        "error": None
    }
    
    try:
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            result["error"] = "Failed to load image"
            return result
        
        # Classify with unified API
        is_sig = is_signature(image_path, use_vlm=use_vlm)
        result["is_signature"] = is_sig
        
        # Also get VLM details if enabled
        if use_vlm:
            client = VLMClient()
            vlm_result = client.classify(img)
            result["vlm_result"] = {
                "classification": vlm_result.result.value,
                "confidence": vlm_result.confidence,
                "used_vlm": vlm_result.used_vlm,
                "reasoning": vlm_result.reasoning
            }
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        return result


def test_sample_images():
    """Test VLM on sample images from data folder."""
    print("\n" + "=" * 50)
    print("Testing Sample Images")
    print("=" * 50)
    
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Test a few sample images
    samples = [
        # Signatures
        ("B-S-95-F-02.tif", True),
        ("H-S-1-F-02.tif", True),
        # Punctuation
        ("punct_dot.png", False),
        ("punct_x.png", False),
        # Empty
        ("empty_white.png", False),
    ]
    
    print(f"\nTesting {len(samples)} sample images...\n")
    
    for filename, expected in samples:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"⏭️  {filename}: not found, skipping")
            continue
        
        result = classify_image(str(filepath), use_vlm=True)
        
        if result["error"]:
            print(f"❌ {filename}: ERROR - {result['error']}")
            continue
        
        is_sig = result["is_signature"]
        correct = is_sig == expected
        
        symbol = "✅" if correct else "❌"
        sig_str = "SIGNATURE" if is_sig else "NOT SIGNATURE"
        exp_str = "SIGNATURE" if expected else "NOT SIGNATURE"
        
        print(f"{symbol} {filename}")
        print(f"   Result: {sig_str}, Expected: {exp_str}")
        
        if result["vlm_result"]:
            vlm = result["vlm_result"]
            used = "VLM" if vlm["used_vlm"] else "Fallback"
            print(f"   VLM: {vlm['classification']} ({used}, conf={vlm['confidence']:.2f})")


def test_single_image(image_path: str):
    """Test a single image."""
    print(f"\nTesting: {image_path}")
    print("-" * 40)
    
    result = classify_image(image_path, use_vlm=True)
    
    if result["error"]:
        print(f"❌ Error: {result['error']}")
        return
    
    is_sig = result["is_signature"]
    print(f"Result: {'SIGNATURE' if is_sig else 'NOT SIGNATURE'}")
    
    if result["vlm_result"]:
        vlm = result["vlm_result"]
        print(f"\nVLM Details:")
        print(f"  Classification: {vlm['classification']}")
        print(f"  Confidence: {vlm['confidence']:.2f}")
        print(f"  Used VLM: {vlm['used_vlm']}")
        if vlm["reasoning"]:
            print(f"  Reasoning: {vlm['reasoning'][:100]}...")


def main():
    print("\n🔍 VLM Signature Classification Test\n")
    
    # Check connection first
    if not test_vlm_connection():
        print("\n⚠️  VLM not available, tests will use fallback")
    
    # If specific images provided, test those
    if len(sys.argv) > 1:
        for image_path in sys.argv[1:]:
            test_single_image(image_path)
    else:
        # Test samples
        test_sample_images()
    
    print("\n" + "=" * 50)
    print("Test Complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
