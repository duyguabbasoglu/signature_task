#!/usr/bin/env python3
"""
Local LLM Classification Test
Test resimlerini LLM ile sınıflandır
"""

import sys
import os
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from src.bbox_detector.vlm_client import VLMClient, VLMResult

# Test dosyaları
TEST_FILES = {
    # EMPTY - boş/gürültü
    "empty_white.png": "EMPTY",
    "empty_black.png": "EMPTY", 
    "empty_noise.png": "EMPTY",
    
    # PUNCT - noktalama/işaret
    "punct_dot.png": "PUNCTUATION",
    "punct_x.png": "PUNCTUATION",
    "punct_circle.png": "PUNCTUATION",
    "punct_line.png": "PUNCTUATION",
    "punct_check.png": "PUNCTUATION",
    "punct_square.png": "PUNCTUATION",
    
    # İmza (IMG_*_converted.png)
    "IMG_1807_converted.png": "SIGNATURE",
    "IMG_1808_converted.png": "SIGNATURE",
    "IMG_1809_converted.png": "SIGNATURE",  # Actually PUNCT but let's see what LLM says
}

def classify_with_llm(image_path: str, llm_client: VLMClient) -> dict:
    """Görüntüyü LLM ile sınıflandır."""
    try:
        # Görüntüyü oku
        img = cv2.imread(str(image_path))
        if img is None:
            return {"result": "ERROR", "reason": "Could not read image"}
        
        # LLM ile sınıflandır
        classification = llm_client.classify(img)
        
        return {
            "result": classification.result.value,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "used_vlm": classification.used_vlm
        }
    except Exception as e:
        return {"result": "ERROR", "reason": str(e)}

def main():
    print("=" * 80)
    print("LOCAL LLM CLASSIFICATION TEST")
    print("=" * 80)
    print()
    
    # API key kontrol et
    api_key = os.getenv("BBOX_LLM_API_KEY")
    if not api_key:
        print("❌ HATA: BBOX_LLM_API_KEY environment variable ayarlanmamış!")
        print("   Lütfen ayarla:")
        print('   $env:BBOX_LLM_API_KEY = "your-api-key"')
        return
    
    # LLM client oluştur
    try:
        print("🔌 Connecting to LLM API...")
        llm = VLMClient(api_key=api_key)
        print("✅ Connected to LLM API")
    except Exception as e:
        print(f"❌ LLM Connection Failed: {e}")
        return
    
    print()
    print("-" * 80)
    print(f"{'File':<30} | {'Expected':<13} | {'LLM Result':<13} | {'Conf':<6} | Status")
    print("-" * 80)
    
    correct = 0
    total = 0
    
    for filename, expected in TEST_FILES.items():
        file_path = Path("data") / filename
        
        if not file_path.exists():
            print(f"{filename:<30} | {expected:<13} | SKIP            | -      | FILE NOT FOUND")
            continue
        
        result = classify_with_llm(str(file_path), llm)
        total += 1
        
        if result.get("result") == "ERROR":
            print(f"{filename:<30} | {expected:<13} | ERROR           | -      | {result.get('reason')}")
        else:
            llm_result = result["result"].upper()
            
            # Sonuç karşılaştırması
            match = "✅" if llm_result == expected else "❌"
            if llm_result == expected:
                correct += 1
            
            conf = f"{result.get('confidence', 0):.1%}"
            print(f"{filename:<30} | {expected:<13} | {llm_result:<13} | {conf:<6} | {match}")
    
    print("-" * 80)
    if total > 0:
        accuracy = correct / total * 100
        print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
    else:
        print("No files tested!")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
