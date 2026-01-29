# Signature Detection POC
# EMPTY, PUNCT (noktalama/geometrik), SIGN (imza) siniflandirmasi
# Oncelik: Yuksek imza accuracy (>95%)

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Thresholds
class ThresholdConfig:
    MIN_COMPONENT_AREA = 25   # Minimum component area
    CONTENT_THRESHOLD = 50    # Bunun altinda = EMPTY
    SIGN_THRESHOLD = 500      # Bunun altinda = PUNCT
    MIN_CONTOUR_AREA = 50     # Shape analizi icin min area
    MORPH_KERNEL_SIZE = 2
    
    # Shape thresholds (conservative)
    CIRCULARITY_HIGH = 0.75
    SOLIDITY_HIGH = 0.85
    RECTANGULARITY_HIGH = 0.85
    COMPLEXITY_LOW = 6


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
    shape_type: Optional[str] = None
    used_vlm: bool = False

    def to_dict(self) -> dict:
        return {
            "result": self.result.value,
            "confidence": round(self.confidence, 2),
            "total_area": self.total_area,
            "component_count": self.component_count,
            "largest_component_area": self.largest_component_area,
            "shape_type": self.shape_type,
            "used_vlm": self.used_vlm
        }


# VLM state
_vlm_client = None
_vlm_enabled = False


def enable_vlm(enabled: bool = True) -> None:
    """VLM'i ac/kapat."""
    global _vlm_enabled
    _vlm_enabled = enabled
    logger.info(f"VLM {'enabled' if enabled else 'disabled'}")


def get_vlm_client():
    """Lazy load VLM client."""
    global _vlm_client
    if _vlm_client is None:
        try:
            from src.bbox_detector.vlm_client import VLMClient
            _vlm_client = VLMClient()
        except ImportError:
            logger.warning("VLM client bulunamadi")
            return None
    return _vlm_client


# Image processing
def load_image(path: str) -> np.ndarray:
    """Image yukle, grayscale'e cevir."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    if len(img.shape) == 2:
        return img
    elif img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported format: {img.shape}")


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu ile binarize et."""
    _, bw_normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bw_inverse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Az beyaz olan = content
    bw = bw_normal if np.count_nonzero(bw_normal) < np.count_nonzero(bw_inverse) else bw_inverse

    # Noise temizle
    kernel = np.ones((ThresholdConfig.MORPH_KERNEL_SIZE,) * 2, np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return bw


def analyze_components(bw: np.ndarray) -> Tuple[int, int, int]:
    """Connected components analiz et."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)

    if num_labels <= 1:
        return 0, 0, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_areas = areas[areas >= ThresholdConfig.MIN_COMPONENT_AREA]

    if len(valid_areas) == 0:
        return 0, 0, 0

    return int(valid_areas.sum()), len(valid_areas), int(valid_areas.max())


# Shape analysis helpers
def _compute_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return min(circularity, 1.0)


def _compute_solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0.0


def _compute_rectangularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    return area / rect_area if rect_area > 0 else 0.0


def _compute_complexity(contour: np.ndarray) -> float:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return float(len(approx))


def analyze_shape_conservative(bw: np.ndarray) -> Tuple[bool, str, float]:
    """Conservative shape analizi - sadece cok belirgin geometrik sekilller."""
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= ThresholdConfig.MIN_CONTOUR_AREA]
    
    if not valid_contours:
        return False, "unknown", 0.5
    
    # Weighted average features
    areas = [cv2.contourArea(c) for c in valid_contours]
    total_area = sum(areas)
    weights = [a / total_area for a in areas] if total_area > 0 else [1/len(areas)] * len(areas)
    
    avg_circularity = sum(_compute_circularity(c) * w for c, w in zip(valid_contours, weights))
    avg_solidity = sum(_compute_solidity(c) * w for c, w in zip(valid_contours, weights))
    avg_rectangularity = sum(_compute_rectangularity(c) * w for c, w in zip(valid_contours, weights))
    avg_complexity = sum(_compute_complexity(c) * w for c, w in zip(valid_contours, weights))
    
    cfg = ThresholdConfig
    
    # Cok yuvarlak = daire
    if avg_circularity > cfg.CIRCULARITY_HIGH:
        return True, "circle", 0.95
    
    # Cok dikdortgen
    if avg_rectangularity > cfg.RECTANGULARITY_HIGH:
        return True, "rectangle", 0.93
    
    # Basit dolu shape
    if avg_complexity < cfg.COMPLEXITY_LOW and avg_solidity > cfg.SOLIDITY_HIGH:
        return True, "simple_filled", 0.90
    
    # Tek solid block
    if len(valid_contours) == 1 and avg_solidity > 0.9:
        return True, "single_block", 0.88
    
    # Cok basit (< 5 vertex)
    if avg_complexity < 5:
        return True, "very_simple", 0.85
    
    # Default: imza kabul et
    return False, "signature", 0.95


def classify(
    total_area: int,
    component_count: int,
    bw: np.ndarray,
    gray: np.ndarray = None,
    use_vlm: bool = False
) -> Tuple[DetectionResult, float, str, bool]:
    """
    Hybrid classification:
    1. EMPTY: area < CONTENT_THRESHOLD
    2. PUNCT: area < SIGN_THRESHOLD
    3. PUNCT: geometrik shape tespit edildi
    4. VLM: belirsiz durumlar icin
    5. Default: SIGN
    """
    cfg = ThresholdConfig
    
    # Step 1: Empty check
    if total_area < cfg.CONTENT_THRESHOLD:
        ratio = total_area / cfg.CONTENT_THRESHOLD if cfg.CONTENT_THRESHOLD > 0 else 0
        confidence = 0.99 - (ratio * 0.1)
        return DetectionResult.EMPTY, confidence, "empty", False

    # Step 2: Kucuk content = punct
    if total_area < cfg.SIGN_THRESHOLD:
        mid = (cfg.CONTENT_THRESHOLD + cfg.SIGN_THRESHOLD) / 2
        dist_from_mid = abs(total_area - mid)
        max_dist = mid - cfg.CONTENT_THRESHOLD
        confidence = 0.95 - (dist_from_mid / max_dist * 0.15)
        return DetectionResult.PUNCT, confidence, "punctuation", False

    # Step 3: Shape analizi
    is_geometric, shape_type, shape_confidence = analyze_shape_conservative(bw)
    
    if is_geometric:
        return DetectionResult.PUNCT, shape_confidence, shape_type, False

    # Step 4: VLM
    vlm_flag = use_vlm if use_vlm is not None else _vlm_enabled
    
    if vlm_flag and gray is not None:
        vlm_client = get_vlm_client()
        if vlm_client is not None:
            try:
                from src.bbox_detector.vlm_client import VLMResult
                vlm_result = vlm_client.classify(gray)
                
                if vlm_result.result == VLMResult.PUNCTUATION:
                    return DetectionResult.PUNCT, vlm_result.confidence, "vlm_punct", vlm_result.used_vlm
                elif vlm_result.result == VLMResult.SIGNATURE:
                    return DetectionResult.SIGN, vlm_result.confidence, "vlm_signature", vlm_result.used_vlm
            except Exception as e:
                logger.warning(f"VLM failed: {e}")

    # Step 5: Default = imza
    if total_area > cfg.SIGN_THRESHOLD * 5:
        confidence = 0.99
    elif total_area > cfg.SIGN_THRESHOLD * 2:
        confidence = 0.97
    else:
        confidence = 0.95

    return DetectionResult.SIGN, confidence, "signature", False


# Public API
def analyze(image_path: str, use_vlm: bool = None) -> AnalysisResult:
    """Image analiz et."""
    gray = load_image(image_path)
    bw = binarize(gray)
    total_area, component_count, largest_area = analyze_components(bw)
    
    vlm_flag = use_vlm if use_vlm is not None else _vlm_enabled
    result, confidence, shape_type, used_vlm = classify(total_area, component_count, bw, gray, vlm_flag)

    return AnalysisResult(
        result=result,
        confidence=confidence,
        total_area=total_area,
        component_count=component_count,
        largest_component_area=largest_area,
        shape_type=shape_type,
        used_vlm=used_vlm
    )


def analyze_bytes(image_bytes: bytes, use_vlm: bool = None) -> AnalysisResult:
    """Bytes'tan image analiz et."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise ValueError("Could not decode image")

    bw = binarize(gray)
    total_area, component_count, largest_area = analyze_components(bw)
    
    vlm_flag = use_vlm if use_vlm is not None else _vlm_enabled
    result, confidence, shape_type, used_vlm = classify(total_area, component_count, bw, gray, vlm_flag)

    return AnalysisResult(
        result=result,
        confidence=confidence,
        total_area=total_area,
        component_count=component_count,
        largest_component_area=largest_area,
        shape_type=shape_type,
        used_vlm=used_vlm
    )


def process_folder(data_dir: str = "data") -> list:
    """Klasordeki tum image'lari isle."""
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
            results.append({"name": img_path.name, "analysis": result})
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")

    return results


# CLI
def main():
    import sys

    if len(sys.argv) > 1:
        for img_path in sys.argv[1:]:
            try:
                result = analyze(img_path)
                print(f"{img_path}: {result.result.value} ({result.confidence:.0%}) - {result.shape_type}")
            except Exception as e:
                print(f"{img_path}: ERROR - {e}")
    else:
        print("=" * 60)
        print("SIGNATURE DETECTION")
        print("=" * 60)

        results = process_folder("data")
        if not results:
            print("No images in data/")
            return

        print(f"{'FILE':<30} | {'RESULT':<8} | {'CONF':<6} | {'AREA':<10} | SHAPE")
        print("-" * 70)

        summary = {DetectionResult.SIGN: 0, DetectionResult.PUNCT: 0, DetectionResult.EMPTY: 0}
        for item in results:
            res = item["analysis"]
            summary[res.result] += 1
            print(f"{item['name']:<30} | {res.result.value:<8} | {res.confidence:>5.0%} | {res.total_area:>8} | {res.shape_type or '-'}")

        print("-" * 70)
        print(f"Total: {len(results)} | SIGN: {summary[DetectionResult.SIGN]} | PUNCT: {summary[DetectionResult.PUNCT]} | EMPTY: {summary[DetectionResult.EMPTY]}")


if __name__ == "__main__":
    main()