import cv2 
import numpy as np 
from pathlib import Path 
from dataclasses import dataclass
from typing import Tuple, Optional 
from enum import Enum 

class DetectionResult(Enum): 
    EMPTY = "EMPTY" 
    CONTENT = "CONTENT" 

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

MIN_COMPONENT_AREA = 25 # piksel^2 
CONTENT_THRESHOLD = 50 # piksel^2 
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
    _, bw_normal = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU ) 
    _, bw_inverse = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU ) 
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
        ratio = total_area / CONTENT_THRESHOLD if CONTENT_THRESHOLD > 0 else 0 
        confidence = 0.95 - (ratio * 0.3) # 0.65 - 0.95 arası 
        return DetectionResult.EMPTY, confidence 
    else: 
        if total_area > CONTENT_THRESHOLD * 10: 
            confidence = 0.98 
        elif total_area > CONTENT_THRESHOLD * 5: 
            confidence = 0.95 
        else: 
            confidence = 0.85 
        return DetectionResult.CONTENT, confidence 

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

def evaluate_dataset(data_dir: str = "data") -> dict: 
    data_path = Path(data_dir) 
    
    results = {
        "empty": {"correct": 0, "total": 0},
        "content": {"correct": 0, "total": 0}
    }
    
    # Desteklenen formatlar
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(data_path.glob(ext)))
    
    for img_path in image_paths:
        name = img_path.name.lower()
        
        # Ground truth belirleme (dosya adından)
        if "empty" in name:
            expected = DetectionResult.EMPTY
            cat = "empty"
        else:
            expected = DetectionResult.CONTENT
            cat = "content"
            
        try: 
            result = analyze(str(img_path)) 
            results[cat]["total"] += 1
            if result.result == expected: 
                results[cat]["correct"] += 1 
            else: 
                print(f" ✗ {img_path.name}: {result.result.value} " f"(beklenen: {expected.value}, alan: {result.total_area})") 
        except Exception as e: 
            print(f" ! {img_path.name}: Hata - {e}") 
            
    # Format results for the original main print logic
    final_results = {}
    for cat, stats in results.items():
        if stats["total"] > 0:
            final_results[cat] = {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": round(stats["correct"] / stats["total"] * 100, 1)
            }
    return final_results 

def main(): 
    import sys 
    if len(sys.argv) > 1: 
        for img_path in sys.argv[1:]: 
            try: 
                result = analyze(img_path) 
                print(f"{img_path}:") 
                print(f" Sonuç: {result.result.value}") 
                print(f" Güven: {result.confidence:.0%}") 
                print(f" Toplam Alan: {result.total_area} px²") 
                print(f" Bileşen Sayısı: {result.component_count}") 
                print() 
            except Exception as e: 
                print(f"{img_path}: HATA - {e}") 
    else: 
        print("=" * 50) 
        print("SIGNATURE PRESENCE DETECTION - TEST SONUÇLARI") 
        print("=" * 50) 
        print(f"\nKonfigürasyon:") 
        print(f" MIN_COMPONENT_AREA: {MIN_COMPONENT_AREA} px²") 
        print(f" CONTENT_THRESHOLD: {CONTENT_THRESHOLD} px²") 
        print() 
        results = evaluate_dataset() 
        if not results: 
            print("Test verisi bulunamadı. 'data/' klasörüne görüntü ekleyin.") 
            print("\nKlasör yapısı:") 
            return 
        print("Kategori Sonuçları:") 
        print("-" * 40) 
        total_correct = 0 
        total_count = 0 
        for category, stats in results.items(): 
            status = "✓" if stats["accuracy"] >= 90 else "△" if stats["accuracy"] >= 70 else "✗" 
            print(f"{status} {category.upper():8} → {stats['correct']:3}/{stats['total']:3} " f"= {stats['accuracy']:5.1f}%") 
            total_correct += stats["correct"] 
            total_count += stats["total"] 
        print("-" * 40) 
        overall = (total_correct / total_count * 100) if total_count > 0 else 0 
        print(f" GENEL → {total_correct:3}/{total_count:3} = {overall:5.1f}%") 
        print() 

if __name__ == "__main__": 
    main()
