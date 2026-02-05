import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DetectionResult(Enum):
    EMPTY = "EMPTY"
    FILLED_PUNCT = "FILLED_PUNCT"
    FILLED_OTHER = "FILLED_OTHER"


@dataclass
class AnalysisResult:
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


MIN_COMPONENT_AREA = 25
EMPTY_THRESHOLD = 200
PUNCT_MAX_AREA = 500
PUNCT_MAX_COMPONENTS = 3
MORPH_KERNEL_SIZE = 2


class BoundingBoxAnalyzer:

    def __init__(
        self,
        min_component_area: int = MIN_COMPONENT_AREA,
        empty_threshold: int = EMPTY_THRESHOLD,
        punct_max_area: int = PUNCT_MAX_AREA,
        punct_max_components: int = PUNCT_MAX_COMPONENTS,
    ):
        self.min_component_area = min_component_area
        self.empty_threshold = empty_threshold
        self.punct_max_area = punct_max_area
        self.punct_max_components = punct_max_components
        self.cnn_model = None

        logger.debug(
            f"Analyzer initialized: empty_threshold={empty_threshold}, "
            f"punct_max_area={punct_max_area}"
        )

    def load_image(self, path: str) -> np.ndarray:
        # grayscale
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Could not load image: {path}")

        if len(img.shape) == 2:
            return img
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        raise ValueError(f"Unsupported image format: {img.shape}")

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        # otsu + noise removal
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

        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

        return bw

    def analyze_components(self, bw: np.ndarray) -> Tuple[int, int, int]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)

        if num_labels <= 1:
            return 0, 0, 0

        areas = stats[1:, cv2.CC_STAT_AREA]
        valid_areas = areas[areas >= self.min_component_area]

        if len(valid_areas) == 0:
            return 0, 0, 0

        total_area = int(valid_areas.sum())
        component_count = len(valid_areas)
        largest_area = int(valid_areas.max())

        return total_area, component_count, largest_area

    def is_empty(self, total_area: int) -> bool:
        return total_area < self.empty_threshold

    def is_punctuation(
        self, total_area: int, component_count: int, largest_area: int
    ) -> bool:
        if total_area > self.punct_max_area:
            return False
        if component_count > self.punct_max_components:
            return False
        return True

    def analyze(self, image_path: str) -> AnalysisResult:

        logger.debug(f"Analyzing: {image_path}")

        gray = self.load_image(image_path)
        bw = self.binarize(gray)
        total_area, component_count, largest_area = self.analyze_components(bw)

        empty = self.is_empty(total_area)

        if empty:
            logger.debug(f"Result: EMPTY (area={total_area})")
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
            confidence = 0.90 if total_area < self.punct_max_area / 2 else 0.80
            logger.debug(f"Result: FILLED_PUNCT (area={total_area})")
            return AnalysisResult(
                result=DetectionResult.FILLED_PUNCT,
                is_empty=False,
                is_punctuation=True,
                confidence=confidence,
                total_area=total_area,
                component_count=component_count
            )
        else:
            confidence = 0.95 if total_area > self.punct_max_area * 2 else 0.85
            logger.debug(f"Result: FILLED_OTHER (area={total_area})")
            return AnalysisResult(
                result=DetectionResult.FILLED_OTHER,
                is_empty=False,
                is_punctuation=False,
                confidence=confidence,
                total_area=total_area,
                component_count=component_count
            )

    def analyze_bytes(self, image_bytes: bytes) -> AnalysisResult:
        nparr = np.frombuffer(image_bytes, np.uint8)
        gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            raise ValueError("Could not decode image")

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
            confidence = 0.90 if total_area < self.punct_max_area / 2 else 0.80
            return AnalysisResult(
                result=DetectionResult.FILLED_PUNCT,
                is_empty=False,
                is_punctuation=True,
                confidence=confidence,
                total_area=total_area,
                component_count=component_count
            )
        else:
            confidence = 0.95 if total_area > self.punct_max_area * 2 else 0.85
            return AnalysisResult(
                result=DetectionResult.FILLED_OTHER,
                is_empty=False,
                is_punctuation=False,
                confidence=confidence,
                total_area=total_area,
                component_count=component_count
            )


# Default analyzer instance
_analyzer = BoundingBoxAnalyzer()


def analyze(image_path: str) -> AnalysisResult:
    """Convenience function for single image analysis"""
    return _analyzer.analyze(image_path)


def analyze_bytes(image_bytes: bytes) -> AnalysisResult:
    """Convenience function for byte array analysis"""
    return _analyzer.analyze_bytes(image_bytes)


def process_folder(data_dir: str = "data") -> list:
    """Process all images in a folder"""
    data_path = Path(data_dir)
    results = []

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG",
                  "*.JPG", "*.JPEG", "*.tif", "*.TIF")

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
            logger.error(f"Error processing {img_path.name}: {e}")

    return results


# -----------------------------------------------------------------------------
# Unified Signature Detection API
# Returns: True = Signature, False = Empty or Punctuation
# -----------------------------------------------------------------------------

def is_signature(image_path: str, use_vlm: bool = True) -> bool:
    """
    Determine if image contains a signature.
    
    Args:
        image_path: Path to image file
        use_vlm: Whether to use VLM for classification (default True)
    
    Returns:
        True if signature detected, False if empty or punctuation
    """
    # Step 1: Check if empty
    result = analyze(image_path)
    
    if result.is_empty:
        logger.debug(f"is_signature({image_path}): False (empty)")
        return False
    
    # Step 2: If filled, determine signature vs punctuation
    if use_vlm:
        try:
            from bbox_detector.vlm_client import VLMClient, VLMResult
            gray = _analyzer.load_image(image_path)
            vlm_client = VLMClient()
            classification = vlm_client.classify(gray)
            
            if classification.result == VLMResult.SIGNATURE:
                logger.debug(f"is_signature({image_path}): True (VLM)")
                return True
            elif classification.result == VLMResult.PUNCTUATION:
                logger.debug(f"is_signature({image_path}): False (VLM punctuation)")
                return False
            else:
                # VLM unknown - use rule-based fallback
                is_sig = not result.is_punctuation
                logger.debug(f"is_signature({image_path}): {is_sig} (VLM unknown, fallback)")
                return is_sig
                
        except Exception as e:
            logger.warning(f"VLM failed, using rule-based: {e}")
            # Fallback to rule-based
            is_sig = not result.is_punctuation
            logger.debug(f"is_signature({image_path}): {is_sig} (fallback)")
            return is_sig
    else:
        # Rule-based only
        is_sig = not result.is_punctuation
        logger.debug(f"is_signature({image_path}): {is_sig} (rule-based)")
        return is_sig


def is_signature_bytes(image_bytes: bytes, use_vlm: bool = True) -> bool:
    """
    Determine if image bytes contain a signature.
    
    Args:
        image_bytes: Image as bytes
        use_vlm: Whether to use VLM for classification (default True)
    
    Returns:
        True if signature detected, False if empty or punctuation
    """
    # Step 1: Check if empty
    result = analyze_bytes(image_bytes)
    
    if result.is_empty:
        return False
    
    # Step 2: If filled, determine signature vs punctuation
    if use_vlm:
        try:
            from bbox_detector.vlm_client import VLMClient, VLMResult
            vlm_client = VLMClient()
            classification = vlm_client.classify_bytes(image_bytes)
            
            if classification.result == VLMResult.SIGNATURE:
                return True
            elif classification.result == VLMResult.PUNCTUATION:
                return False
            else:
                # VLM unknown - use rule-based fallback
                return not result.is_punctuation
                
        except Exception as e:
            logger.warning(f"VLM failed, using rule-based: {e}")
            return not result.is_punctuation
    else:
        # Rule-based only
        return not result.is_punctuation


def get_expected_result(filename: str) -> bool:
    """
    Get expected classification result based on filename pattern.
    
    Returns:
        True if expected to be signature, False if expected to be empty/punctuation
    """
    name_lower = filename.lower()
    
    # Empty files
    if name_lower.startswith("empty_"):
        return False
    
    # Punctuation files
    if name_lower.startswith("punct_"):
        return False
    
    # IMG converted files are punctuation
    if "img_" in name_lower and "_converted" in name_lower:
        return False
    
    # Everything else (B-S-*, H-S-*, etc.) is signature
    return True


def test_accuracy(data_dir: str = "data", use_vlm: bool = False) -> dict:
    """
    Test classification accuracy on data folder.
    
    Args:
        data_dir: Path to data folder
        use_vlm: Whether to use VLM (default False for local testing)
    
    Returns:
        Dict with accuracy metrics
    """
    data_path = Path(data_dir)
    
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG",
                  "*.JPG", "*.JPEG", "*.tif", "*.TIF")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(data_path.glob(ext)))
    
    image_paths.sort(key=lambda x: x.name)
    
    correct = 0
    total = 0
    errors = []
    
    # Track by category
    categories = {
        "empty": {"correct": 0, "total": 0, "errors": []},
        "punct": {"correct": 0, "total": 0, "errors": []},
        "signature": {"correct": 0, "total": 0, "errors": []}
    }
    
    for img_path in image_paths:
        filename = img_path.name
        expected = get_expected_result(filename)
        
        # Determine category
        name_lower = filename.lower()
        if name_lower.startswith("empty_"):
            category = "empty"
        elif name_lower.startswith("punct_") or ("img_" in name_lower and "_converted" in name_lower):
            category = "punct"
        else:
            category = "signature"
        
        try:
            actual = is_signature(str(img_path), use_vlm=use_vlm)
            
            total += 1
            categories[category]["total"] += 1
            
            if actual == expected:
                correct += 1
                categories[category]["correct"] += 1
            else:
                error_info = {"file": filename, "expected": expected, "actual": actual}
                errors.append(error_info)
                categories[category]["errors"].append(error_info)
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            errors.append({"file": filename, "error": str(e)})
    
    accuracy = correct / total if total > 0 else 0
    
    result = {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy * 100, 2),
        "errors": errors,
        "by_category": {}
    }
    
    for cat, data in categories.items():
        cat_acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        result["by_category"][cat] = {
            "total": data["total"],
            "correct": data["correct"],
            "accuracy": round(cat_acc * 100, 2),
            "errors": data["errors"]
        }
    
    return result

