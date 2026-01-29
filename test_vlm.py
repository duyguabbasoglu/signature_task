#!/usr/bin/env python3
# VLM Test Script - Virtual PC'de calistir

import sys
import os

# Path ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import poc
from pathlib import Path


def test_vlm():
    # VLM'i ac
    poc.enable_vlm(True)
    
    print("=" * 70)
    print("VLM TEST")
    print("=" * 70)
    
    # Test dosyalari
    test_files = [
        # Punct olmasi gerekenler
        "data/test/punct_dot.png",
        "data/test/punct_x.png", 
        "data/test/punct_check.png",
        "data/test/punct_square.png",
        "data/test/punct_circle.png",
        "data/test/punct_line.png",
        # IMG dosyalari
        "data/IMG_1807_converted.png",
        "data/IMG_1808_converted.png", 
        "data/IMG_1809_converted.png",
        # Signature ornekleri
        "data/B-S-1-F-01.tif",
        "data/B-S-1-F-02.tif",
        "data/H-S-2-G-15.tif",
    ]
    
    print(f"{'FILE':<35} | {'RESULT':<8} | {'VLM':<5} | {'CONF':<6} | SHAPE")
    print("-" * 80)
    
    for f in test_files:
        if not Path(f).exists():
            print(f"{f:<35} | DOSYA YOK")
            continue
            
        try:
            result = poc.analyze(f, use_vlm=True)
            vlm_str = "✓" if result.used_vlm else "-"
            print(f"{Path(f).name:<35} | {result.result.value:<8} | {vlm_str:<5} | {result.confidence:>5.0%} | {result.shape_type}")
        except Exception as e:
            print(f"{Path(f).name:<35} | ERROR: {e}")
    
    print("-" * 80)
    print()
    print("VLM = ✓ ise VLM kullanildi, - ise shape analysis kullanildi")


def test_accuracy():
    # VLM kapatilmis halde accuracy testi
    poc.enable_vlm(False)
    
    print("=" * 70)
    print("ACCURACY TEST (VLM OFF)")
    print("=" * 70)
    
    # Signature dosyalari
    sign_files = list(Path("data").glob("B-S-*.tif")) + list(Path("data").glob("H-S-*.tif"))
    
    correct = 0
    wrong = []
    
    for f in sign_files:
        result = poc.analyze(str(f))
        if result.result == poc.DetectionResult.SIGN:
            correct += 1
        else:
            wrong.append((f.name, result.result.value, result.shape_type))
    
    accuracy = correct / len(sign_files) * 100 if sign_files else 0
    
    print(f"Signature Accuracy: {correct}/{len(sign_files)} = {accuracy:.2f}%")
    
    if wrong:
        print(f"\nYanlis siniflandirilan ({len(wrong)}):")
        for name, res, shape in wrong[:10]:
            print(f"  {name}: {res} ({shape})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--accuracy":
        test_accuracy()
    else:
        test_vlm()
        print()
        test_accuracy()
