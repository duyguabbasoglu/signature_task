"""
Bounding Box Content Detection Module

Algoritma:
1. Bounding box boş mu dolu mu? → İlk kontrol
2. Doluysa: Noktalama işareti mi? → CNN ile tespit
3. Uygun response döndür

Detection Results:
- EMPTY: Bounding box boş
- FILLED_PUNCT: Dolu ve noktalama işareti
- FILLED_OTHER: Dolu ama noktalama değil (imza vs.)
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class DetectionResult(Enum):
    """Tespit sonucu kategorileri"""
    EMPTY = "EMPTY"              # Bounding box boş
    FILLED_PUNCT = "FILLED_PUNCT"  # Noktalama işareti var
    FILLED_OTHER = "FILLED_OTHER"  # Başka içerik var (imza vb.)


@dataclass
class AnalysisResult:
    """Analiz sonucu veri yapısı"""
    result: DetectionResult
    is_empty: bool
    is_punctuation: Optional[bool]
    confidence: float
    total_area: int
    component_count: int
    
    def to_dict(self) -> dict:
        return {
            "result": self.result.value,
            "is_empty": self.is_empty,
            "is_punctuation": self.is_punctuation,
            "confidence": round(self.confidence, 2),
            "total_area": self.total_area,
            "component_count": self.component_count
        }


# Konfigürasyon Sabitleri
MIN_COMPONENT_AREA = 25  # piksel^2 - bu altındakiler gürültü
EMPTY_THRESHOLD = 50     # piksel^2 - bu altı "boş" sayılır
PUNCT_MAX_AREA = 500     # piksel^2 - noktalama işareti maksimum alan
PUNCT_MAX_COMPONENTS = 3 # noktalama işaretleri genelde 1-3 parça
MORPH_KERNEL_SIZE = 2


class BoundingBoxAnalyzer:
    """
    Bounding box içerik analizi için ana sınıf.
    
    İki aşamalı tespit:
    1. Boş/Dolu kontrolü (rule-based)
    2. Noktalama tespiti (rule-based + CNN hazırlığı)
    """
    
    def __init__(self):
        self.cnn_model = None  # İleride CNN model yüklenecek
    
    def load_image(self, path: str) -> np.ndarray:
        """Görüntüyü grayscale olarak yükle"""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError(f"Görüntü yüklenemedi: {path}")
        
        if len(img.shape) == 2:
            return img
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        raise ValueError(f"Desteklenmeyen format: {img.shape}")
    
    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """Otsu binarization + noise cleaning"""
        _, bw_normal = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        _, bw_inverse = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Daha az beyaz piksel içeren versiyonu seç (ink = foreground)
        if np.count_nonzero(bw_normal) < np.count_nonzero(bw_inverse):
            bw = bw_normal
        else:
            bw = bw_inverse
        
        # Morfolojik noise temizleme
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        
        return bw
    
    def analyze_components(self, bw: np.ndarray) -> Tuple[int, int, int]:
        """Connected component analizi"""
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
    
    def is_empty(self, total_area: int) -> bool:
        """Bounding box boş mu?"""
        return total_area < EMPTY_THRESHOLD
    
    def is_punctuation(self, total_area: int, component_count: int, largest_area: int) -> bool:
        """
        Noktalama işareti mi?
        
        Noktalama özellikleri:
        - Küçük alan
        - Az sayıda bileşen (1-3)
        - Tek büyük parça yok
        """
        # Alan kontrolü
        if total_area > PUNCT_MAX_AREA:
            return False
        
        # Bileşen sayısı kontrolü
        if component_count > PUNCT_MAX_COMPONENTS:
            return False
        
        # İleride CNN ile daha doğru tespit yapılacak
        # Şimdilik rule-based devam
        return True
    
    def analyze(self, image_path: str) -> AnalysisResult:
        """
        Ana analiz fonksiyonu.
        
        Algoritma:
        1. Görüntü yükle ve binarize et
        2. Component analizi yap
        3. Boş mu kontrol et → EMPTY
        4. Doluysa noktalama mı kontrol et → FILLED_PUNCT / FILLED_OTHER
        """
        gray = self.load_image(image_path)
        bw = self.binarize(gray)
        total_area, component_count, largest_area = self.analyze_components(bw)
        
        # Aşama 1: Boş/Dolu kontrolü
        empty = self.is_empty(total_area)
        
        if empty:
            return AnalysisResult(
                result=DetectionResult.EMPTY,
                is_empty=True,
                is_punctuation=None,
                confidence=0.95,
                total_area=total_area,
                component_count=component_count
            )
        
        # Aşama 2: Noktalama kontrolü
        punct = self.is_punctuation(total_area, component_count, largest_area)
        
        if punct:
            # Küçük alanlarda yüksek güven
            confidence = 0.90 if total_area < PUNCT_MAX_AREA / 2 else 0.80
            return AnalysisResult(
                result=DetectionResult.FILLED_PUNCT,
                is_empty=False,
                is_punctuation=True,
                confidence=confidence,
                total_area=total_area,
                component_count=component_count
            )
        else:
            # Büyük alanlarda yüksek güven
            confidence = 0.95 if total_area > PUNCT_MAX_AREA * 2 else 0.85
            return AnalysisResult(
                result=DetectionResult.FILLED_OTHER,
                is_empty=False,
                is_punctuation=False,
                confidence=confidence,
                total_area=total_area,
                component_count=component_count
            )
    
    def analyze_bytes(self, image_bytes: bytes) -> AnalysisResult:
        """Byte array'den analiz"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if gray is None:
            raise ValueError("Görüntü decode edilemedi")
        
        bw = self.binarize(gray)
        total_area, component_count, largest_area = self.analyze_components(bw)
        
        empty = self.is_empty(total_area)
        
        if empty:
            return AnalysisResult(
                result=DetectionResult.EMPTY,
                is_empty=True,
                is_punctuation=None,
                confidence=0.95,
                total_area=total_area,
                component_count=component_count
            )
        
        punct = self.is_punctuation(total_area, component_count, largest_area)
        
        if punct:
            confidence = 0.90 if total_area < PUNCT_MAX_AREA / 2 else 0.80
            return AnalysisResult(
                result=DetectionResult.FILLED_PUNCT,
                is_empty=False,
                is_punctuation=True,
                confidence=confidence,
                total_area=total_area,
                component_count=component_count
            )
        else:
            confidence = 0.95 if total_area > PUNCT_MAX_AREA * 2 else 0.85
            return AnalysisResult(
                result=DetectionResult.FILLED_OTHER,
                is_empty=False,
                is_punctuation=False,
                confidence=confidence,
                total_area=total_area,
                component_count=component_count
            )


# Singleton analyzer instance
_analyzer = BoundingBoxAnalyzer()


def analyze(image_path: str) -> AnalysisResult:
    """Convenience function for single image analysis"""
    return _analyzer.analyze(image_path)


def analyze_bytes(image_bytes: bytes) -> AnalysisResult:
    """Convenience function for byte array analysis"""
    return _analyzer.analyze_bytes(image_bytes)


def process_folder(data_dir: str = "data") -> list:
    """Klasördeki tüm resimleri işle"""
    data_path = Path(data_dir)
    results = []
    
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG", "*.tif", "*.TIF")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(data_path.glob(ext)))
    
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
        # Tek dosya analizi
        for img_path in sys.argv[1:]:
            try:
                result = analyze(img_path)
                print(f"{img_path}:")
                print(f"  Sonuç: {result.result.value}")
                print(f"  Boş mu: {result.is_empty}")
                print(f"  Noktalama mı: {result.is_punctuation}")
                print(f"  Güven: {result.confidence:.0%}")
                print(f"  Toplam Alan: {result.total_area} px²")
                print()
            except Exception as e:
                print(f"{img_path}: HATA - {e}")
    else:
        # Klasör analizi
        print("=" * 65)
        print("BOUNDING BOX CONTENT DETECTION - DATA ANALYSIS")
        print("=" * 65)
        print(f"\nKonfigürasyon:")
        print(f"  EMPTY_THRESHOLD: {EMPTY_THRESHOLD} px²")
        print(f"  PUNCT_MAX_AREA: {PUNCT_MAX_AREA} px²")
        print()
        
        results = process_folder()
        
        if not results:
            print("Test verisi bulunamadı. 'data/' klasörüne görüntü ekleyin.")
            return
        
        print(f"{'DOSYA ADI':<30} | {'SONUÇ':<15} | {'BOŞ':<5} | {'PUNCT':<5} | {'ALAN'}")
        print("-" * 75)
        
        summary = {
            DetectionResult.EMPTY: 0,
            DetectionResult.FILLED_PUNCT: 0,
            DetectionResult.FILLED_OTHER: 0
        }
        
        for item in results:
            res = item["analysis"]
            summary[res.result] += 1
            
            punct_str = "N/A" if res.is_punctuation is None else ("Evet" if res.is_punctuation else "Hayır")
            empty_str = "Evet" if res.is_empty else "Hayır"
            
            print(f"{item['name']:<30} | {res.result.value:<15} | {empty_str:<5} | {punct_str:<5} | {res.total_area:>5} px²")
        
        print("-" * 75)
        total = len(results)
        print(f"\nÖzet ({total} dosya):")
        for result_type, count in summary.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {result_type.value:<15}: {count:>4} ({pct:5.1f}%)")
        print()


if __name__ == "__main__":
    main()
