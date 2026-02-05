"""
Bounding Box Focus Selection

When multiple bounding boxes appear on the same document/page,
this module helps select the most likely signature box.

Strategies:
1. Position-based: Prefer boxes in typical signature locations (bottom, right)
2. Area-based: Prefer boxes with signature-like dimensions
3. Content-based: Use empty/filled detection to filter candidates
4. Aspect ratio: Signatures tend to be wider than tall
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a bounding box with its coordinates and metadata."""
    x: int  # left
    y: int  # top
    width: int
    height: int
    confidence: float = 1.0
    label: str = ""
    
    @property
    def x2(self) -> int:
        """Right edge x-coordinate."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom edge y-coordinate."""
        return self.y + self.height
    
    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio. > 1 means wider than tall."""
        return self.width / self.height if self.height > 0 else 0
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class FocusResult:
    """Result of focus box selection."""
    selected_box: Optional[BoundingBox]
    all_boxes: List[BoundingBox]
    selection_reason: str
    scores: List[float]


class BoxFocusSelector:
    """
    Selects the most relevant bounding box when multiple boxes
    are present on a document page.
    """
    
    def __init__(
        self,
        min_area: int = 500,
        max_area: int = 500000,
        min_aspect_ratio: float = 0.3,  # Very tall boxes unlikely to be signatures
        max_aspect_ratio: float = 10.0,  # Very wide boxes unlikely to be signatures
        prefer_bottom: bool = True,      # Signatures often at bottom of documents
        prefer_right: bool = True,       # Signatures often on right side
        position_weight: float = 0.3,    # Weight for position scoring
        area_weight: float = 0.3,        # Weight for area scoring
        aspect_weight: float = 0.2,      # Weight for aspect ratio scoring
        content_weight: float = 0.2,     # Weight for content scoring
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.prefer_bottom = prefer_bottom
        self.prefer_right = prefer_right
        self.position_weight = position_weight
        self.area_weight = area_weight
        self.aspect_weight = aspect_weight
        self.content_weight = content_weight
        
        logger.debug(
            f"BoxFocusSelector initialized: min_area={min_area}, "
            f"prefer_bottom={prefer_bottom}, prefer_right={prefer_right}"
        )
    
    def filter_valid_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Filter out boxes that don't meet basic criteria."""
        valid = []
        for box in boxes:
            # Area check
            if box.area < self.min_area:
                logger.debug(f"Box filtered: area {box.area} < {self.min_area}")
                continue
            if box.area > self.max_area:
                logger.debug(f"Box filtered: area {box.area} > {self.max_area}")
                continue
            
            # Aspect ratio check
            if box.aspect_ratio < self.min_aspect_ratio:
                logger.debug(f"Box filtered: aspect {box.aspect_ratio:.2f} < {self.min_aspect_ratio}")
                continue
            if box.aspect_ratio > self.max_aspect_ratio:
                logger.debug(f"Box filtered: aspect {box.aspect_ratio:.2f} > {self.max_aspect_ratio}")
                continue
            
            valid.append(box)
        
        return valid
    
    def score_position(self, box: BoundingBox, image_height: int, image_width: int) -> float:
        """
        Score based on position. Higher scores for signature-typical positions.
        
        Typical signature locations:
        - Bottom of document
        - Right side of document (for single signature)
        - Bottom-right corner
        """
        cx, cy = box.center
        
        # Vertical position score (0-1, higher = more bottom)
        y_score = cy / image_height if image_height > 0 else 0.5
        
        # Horizontal position score (0-1, higher = more right)
        x_score = cx / image_width if image_width > 0 else 0.5
        
        # Combine scores based on preferences
        position_score = 0.5
        if self.prefer_bottom:
            position_score = y_score * 0.7 + position_score * 0.3
        if self.prefer_right:
            position_score = x_score * 0.3 + position_score * 0.7
        
        return position_score
    
    def score_area(self, box: BoundingBox) -> float:
        """
        Score based on area. Prefer medium-sized boxes for signatures.
        
        Very small boxes are likely noise/punctuation.
        Very large boxes might be full-page content.
        """
        # Ideal signature area range: 2000 - 50000 pixels
        ideal_min = 2000
        ideal_max = 50000
        
        if box.area < ideal_min:
            return box.area / ideal_min  # 0 to 1
        elif box.area > ideal_max:
            return max(0, 1 - (box.area - ideal_max) / ideal_max)  # Decreasing
        else:
            return 1.0  # In ideal range
    
    def score_aspect_ratio(self, box: BoundingBox) -> float:
        """
        Score based on aspect ratio.
        
        Signatures are typically wider than tall (1.5 - 5 ratio).
        """
        ideal_min = 1.5
        ideal_max = 5.0
        
        ar = box.aspect_ratio
        
        if ar < ideal_min:
            return max(0, ar / ideal_min)
        elif ar > ideal_max:
            return max(0, 1 - (ar - ideal_max) / ideal_max)
        else:
            return 1.0
    
    def score_content(self, box: BoundingBox, image: np.ndarray) -> float:
        """
        Score based on content analysis within the box.
        
        Uses the detector to check if content looks like a signature.
        """
        try:
            # Extract box region
            x1, y1 = max(0, box.x), max(0, box.y)
            x2, y2 = min(image.shape[1], box.x2), min(image.shape[0], box.y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            region = image[y1:y2, x1:x2]
            
            # Use our detector to analyze
            from bbox_detector.detector import BoundingBoxAnalyzer
            analyzer = BoundingBoxAnalyzer()
            
            # Convert region to bytes
            _, buffer = cv2.imencode('.png', region)
            result = analyzer.analyze_bytes(buffer.tobytes())
            
            # Empty boxes get 0, non-empty boxes get higher score
            if result.is_empty:
                return 0.0
            
            # Prefer larger content (more likely signature)
            if result.total_area > 1000:
                return 1.0
            elif result.total_area > 500:
                return 0.7
            else:
                return 0.4
                
        except Exception as e:
            logger.warning(f"Content scoring failed: {e}")
            return 0.5
    
    def calculate_scores(
        self, 
        boxes: List[BoundingBox], 
        image: np.ndarray
    ) -> List[float]:
        """Calculate composite scores for each box."""
        if not boxes:
            return []
        
        h, w = image.shape[:2]
        scores = []
        
        for box in boxes:
            pos_score = self.score_position(box, h, w)
            area_score = self.score_area(box)
            aspect_score = self.score_aspect_ratio(box)
            content_score = self.score_content(box, image)
            
            composite = (
                pos_score * self.position_weight +
                area_score * self.area_weight +
                aspect_score * self.aspect_weight +
                content_score * self.content_weight
            )
            
            logger.debug(
                f"Box ({box.x},{box.y}): pos={pos_score:.2f}, area={area_score:.2f}, "
                f"aspect={aspect_score:.2f}, content={content_score:.2f}, total={composite:.2f}"
            )
            
            scores.append(composite)
        
        return scores
    
    def select_focus_box(
        self, 
        boxes: List[BoundingBox], 
        image: np.ndarray
    ) -> FocusResult:
        """
        Select the most relevant bounding box for signature detection.
        
        Args:
            boxes: List of candidate bounding boxes
            image: The document image (grayscale or color)
        
        Returns:
            FocusResult with selected box and reasoning
        """
        if not boxes:
            return FocusResult(
                selected_box=None,
                all_boxes=[],
                selection_reason="No boxes provided",
                scores=[]
            )
        
        # Filter valid boxes
        valid_boxes = self.filter_valid_boxes(boxes)
        
        if not valid_boxes:
            return FocusResult(
                selected_box=None,
                all_boxes=boxes,
                selection_reason="No boxes passed filtering criteria",
                scores=[0.0] * len(boxes)
            )
        
        if len(valid_boxes) == 1:
            return FocusResult(
                selected_box=valid_boxes[0],
                all_boxes=valid_boxes,
                selection_reason="Only one valid box",
                scores=[1.0]
            )
        
        # Calculate scores for valid boxes
        scores = self.calculate_scores(valid_boxes, image)
        
        # Select highest scoring box
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        
        return FocusResult(
            selected_box=valid_boxes[best_idx],
            all_boxes=valid_boxes,
            selection_reason=f"Highest score ({scores[best_idx]:.2f})",
            scores=scores
        )


# Default selector instance
_selector = BoxFocusSelector()


def select_focus_box(
    boxes: List[Tuple[int, int, int, int]], 
    image: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convenience function to select the focus box.
    
    Args:
        boxes: List of (x, y, width, height) tuples
        image: Document image
    
    Returns:
        Selected box as (x, y, width, height) or None
    """
    bbox_list = [BoundingBox(x=b[0], y=b[1], width=b[2], height=b[3]) for b in boxes]
    result = _selector.select_focus_box(bbox_list, image)
    
    if result.selected_box:
        return result.selected_box.to_tuple()
    return None


def select_focus_box_detailed(
    boxes: List[Tuple[int, int, int, int]], 
    image: np.ndarray
) -> FocusResult:
    """
    Select focus box with detailed results.
    
    Args:
        boxes: List of (x, y, width, height) tuples
        image: Document image
    
    Returns:
        FocusResult with selected box, all boxes, and scores
    """
    bbox_list = [BoundingBox(x=b[0], y=b[1], width=b[2], height=b[3]) for b in boxes]
    return _selector.select_focus_box(bbox_list, image)
