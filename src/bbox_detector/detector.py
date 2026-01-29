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
EMPTY_THRESHOLD = 50
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
