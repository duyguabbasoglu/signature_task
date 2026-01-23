import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class DetectionResult(Enum):
    EMPTY = "EMPTY"  
    PUNCT = "PUNCT" 
    SIGN = "SIGN"


@dataclass
class AnalysisResult:
    result: DetectionResult
    confidence: float
    total_area: int
    component_count: int
    largest_component_area: int
    
    def to_dict(self) -> dict:
        return {
            "result": self.result.value,
            "confidence": round(self.confidence, 2),
            "total_area": self.total_area,
            "component_count": self.component_count,
            "largest_component_area": self.largest_component_area
        }


MIN_COMPONENT_AREA = 25  # piksel^2
CONTENT_THRESHOLD = 50  # piksel^2 (küçük lekeler altı empty)
SIGN_THRESHOLD = 500  # piksel^2 (punct vs sign ayrımı)
MORPH_KERNEL_SIZE = 2


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Görüntü yüklenemedi: {path}")
    
    # chanel sayisina gore grayscale 
    if len(img.shape) == 2:
        return img
    elif img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    raise ValueError(f"Desteklenmeyen görüntü formatı: {img.shape}")


def binarize(gray: np.ndarray) -> np.ndarray:
    # otsu
    _, bw_normal = cv2.threshold(
        gray, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, bw_inverse = cv2.threshold(
        gray, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
   
    if np.count_nonzero(bw_normal) < np.count_nonzero(bw_inverse):
        bw = bw_normal
    else:
        bw = bw_inverse
    
    #noise cleaning
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    
    return bw


def analyze_components(bw: np.ndarray) -> Tuple[int, int, int]:
   
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
    
    if num_labels <= 1:
        return 0, 0, 0
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_areas = areas[areas >= MIN_COMPONENT_AREA]
    
    if len(valid_areas) == 0:
        return 0, 0, 0
    
    total_area = int(valid_areas.sum())
    component_count = len(valid_areas)
    largest_area = int(valid_areas.max())
    
    return total_area, component_count, largest_area


def classify(total_area: int, component_count: int) -> Tuple[DetectionResult, float]:
    if total_area < CONTENT_THRESHOLD:
        # Alan 0'a ne kadar yakınsa güven o kadar yüksek
        ratio = total_area / CONTENT_THRESHOLD if CONTENT_THRESHOLD > 0 else 0
        confidence = 0.99 - (ratio * 0.1) # %89 - %99 arası
        return DetectionResult.EMPTY, confidence
    
    elif total_area < SIGN_THRESHOLD:
        # Punctuation bölgesi
        # Merkeze (threshold'ların ortasına) yakınlık güveni belirler
        mid = (CONTENT_THRESHOLD + SIGN_THRESHOLD) / 2
        dist_from_mid = abs(total_area - mid)
        max_dist = mid - CONTENT_THRESHOLD
        confidence = 0.95 - (dist_from_mid / max_dist * 0.15) # %80 - %95 arası
        return DetectionResult.PUNCT, confidence
        
    else:
        # Signature bölgesi
        if total_area > SIGN_THRESHOLD * 5:
            confidence = 0.99
        elif total_area > SIGN_THRESHOLD * 2:
            confidence = 0.97
        else:
            confidence = 0.90
        
        return DetectionResult.SIGN, confidence


def analyze(image_path: str) -> AnalysisResult:
    gray = load_image(image_path)
    bw = binarize(gray)
    total_area, component_count, largest_area = analyze_components(bw)
    result, confidence = classify(total_area, component_count)
    
    return AnalysisResult(
        result=result,
        confidence=confidence,
        total_area=total_area,
        component_count=component_count,
        largest_component_area=largest_area
    )


def analyze_bytes(image_bytes: bytes) -> AnalysisResult:
    nparr = np.frombuffer(image_bytes, np.uint8)
    gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if gray is None:
        raise ValueError("Görüntü decode edilemedi")
    
    bw = binarize(gray)
    total_area, component_count, largest_area = analyze_components(bw)
    result, confidence = classify(total_area, component_count)
    
    return AnalysisResult(
        result=result,
        confidence=confidence,
        total_area=total_area,
        component_count=component_count,
        largest_component_area=largest_area
    )


def process_folder(data_dir: str = "data") -> list:
    data_path = Path(data_dir)
    results = []
    
    # Desteklenen formatlar
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(data_path.glob(ext)))
    
    # Alfabetik sirala
    image_paths.sort(key=lambda x: x.name)
    
    for img_path in image_paths:
        try:
            result = analyze(str(img_path))
            results.append({
                "name": img_path.name,
                "analysis": result
            })
        except Exception as e:
            print(f"  ! {img_path.name}: Hata - {e}")
            
    return results



def main():
    import sys
    
    if len(sys.argv) > 1:
        for img_path in sys.argv[1:]:
            try:
                result = analyze(img_path)
                print(f"{img_path}:")
                print(f"  Sonuç: {result.result.value}")
                print(f"  Güven: {result.confidence:.0%}")
                print(f"  Toplam Alan: {result.total_area} px²")
                print(f"  Bileşen Sayısı: {result.component_count}")
                print()
            except Exception as e:
                print(f"{img_path}: HATA - {e}")
    else:
        print("=" * 60)
        print("SIGNATURE PRESENCE DETECTION - DATA ANALYSIS")
        print("=" * 60)
        print(f"\nKonfigürasyon:")
        print(f"  MIN_COMPONENT_AREA: {MIN_COMPONENT_AREA} px²")
        print(f"  CONTENT_THRESHOLD: {CONTENT_THRESHOLD} px²")
        print()
        
        results = process_folder()
        
        if not results:
            print("Test verisi bulunamadı. 'data/' klasörüne görüntü ekleyin.")
            return
        
        print(f"{'DOSYA ADI':<30} | {'SONUÇ':<10} | {'GÜVEN':<8} | {'ALAN'}")
        print("-" * 60)
        
        summary_counts = {
            DetectionResult.SIGN: 0,
            DetectionResult.PUNCT: 0,
            DetectionResult.EMPTY: 0
        }
        
        for item in results:
            res = item["analysis"]
            summary_counts[res.result] += 1
            
            # Icon belirleme
            if res.result == DetectionResult.SIGN:
                icon = "   [S]"
            elif res.result == DetectionResult.PUNCT:
                icon = "   [P]"
            else:
                icon = "   [E]"
                
            print(f"{item['name']:<30} | {res.result.value:<10} | {res.confidence:>6.0%} | {res.total_area:>5} px²{icon}")
        
        print("-" * 60)
        total_files = len(results)
        print(f"Toplam {total_files} dosya işlendi.")
        print("-" * 60)
        print(f"{'KATEGORİ':<15} | {'ADET':<10} | {'YÜZDE'}")
        print("-" * 60)
        
        for res_type, count in summary_counts.items():
            percentage = (count / total_files * 100) if total_files > 0 else 0
            print(f"{res_type.value:<15} | {count:<10} | {percentage:>5.1f}%")
        
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
