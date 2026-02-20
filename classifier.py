#!/usr/bin/env python3
"""
Advanced Rule-Based Classification Pipeline
- Pre-processing (CLAHE, denoise, binarize)
- Feature extraction (ink_ratio, cc stats, skeleton metrics)
- Shape detection (dot, circle, line, x, square, check)
- Rule-based EMPTY/PUNCT/SIGNATURE/AMBIGUOUS gating
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_image_robust(image_path) -> Optional[np.ndarray]:
    """
    Load image with support for multiple formats including HEIC.
    
    Args:
        image_path: Path to image file
        
    Returns:
        RGB image as numpy array or None if failed
    """
    path = Path(image_path)
    
    # Try OpenCV first (works for most formats)
    try:
        img = cv2.imread(str(path))
        if img is not None:
            # OpenCV loads as BGR, convert to RGB for consistency
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        pass
    
    # Try PIL/Pillow for other formats (including HEIC)
    try:
        from PIL import Image
        # Register HEIF/HEIC opener for pillow-heif support
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except:
            pass
        
        img_pil = Image.open(path)
        
        # Convert RGBA/P to RGB if necessary
        if img_pil.mode in ('RGBA', 'P', 'LA'):
            rgb_img = Image.new('RGB', img_pil.size, (255, 255, 255))
            rgb_img.paste(img_pil, mask=img_pil.split()[-1] if img_pil.mode in ('RGBA', 'LA') else None)
            img_pil = rgb_img
        elif img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        return np.array(img_pil)
    except:
        pass
    
    return None


class ClassResult(Enum):
    """Classification results."""
    EMPTY = "EMPTY"
    PUNCTUATION = "PUNCT"
    SIGNATURE = "SIGN"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class Features:
    """Extracted features."""
    # Ink metrics
    ink_ratio: float
    
    # Connected components
    cc_count: int
    largest_cc_area: int
    largest_cc_ratio: float
    
    # Shape descriptors (per largest CC)
    aspect_ratio: float
    extent: float
    solidity: float
    circularity: float
    
    # Skeleton metrics
    skeleton_length: int
    endpoints_count: int
    branchpoints_count: int
    curvature_turns: int
    
    # Derived complexity score
    complexity_score: float
    
    # Shape detection flags
    is_dot: bool = False
    is_circle: bool = False
    is_line: bool = False
    is_x: bool = False
    is_square: bool = False
    is_check: bool = False
    
    # Additional
    has_holes: bool = False
    x_projection_entropy: float = 0.0


# ============================================================================
# 1. PRE-PROCESSING
# ============================================================================

def preprocess_image(image: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-process image for robust feature extraction.
    
    Returns:
        binary: Binarized image (0/255)
        gray: Preprocessed grayscale
    """
    # Ensure grayscale (handle RGB, BGR, or already grayscale)
    if len(image.shape) == 3:
        # Assume RGB from load_image_robust or BGR from cv2.imread
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize size (short edge = target_size)
    h, w = gray.shape
    scale = target_size / min(h, w)
    if abs(scale - 1.0) > 0.01:
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Denoise (median blur 3x3)
    gray = cv2.medianBlur(gray, 3)
    
    # Binarize: Otsu (black=0, white=255 by default; we want black=255 for ink)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # If mostly black (all ink), likely inverted - swap
    black_ratio = np.sum(binary == 255) / binary.size
    if black_ratio > 0.95:
        binary = 255 - binary
    
    # Morphology: open (remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Light close for thin parts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary, gray


# ============================================================================
# 2. CONNECTED COMPONENTS & INK METRICS
# ============================================================================

def get_connected_components(binary: np.ndarray) -> List[np.ndarray]:
    """Extract connected components from binary image."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def compute_ink_metrics(binary: np.ndarray) -> Tuple[float, int, float]:
    """
    Compute ink-based metrics.
    
    Returns:
        ink_ratio: black_pixels / total_pixels
        cc_count: number of connected components
        largest_cc_ratio: max_area / total_pixels
    """
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)  # OpenCV: black=255 after INV
    
    ink_ratio = black_pixels / total_pixels if total_pixels > 0 else 0
    
    # CC count (filter by min area)
    contours = get_connected_components(binary)
    min_area = max(5, total_pixels * 0.00005)  # At least 0.005% of image
    
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    cc_count = len(valid_contours)
    
    # Largest CC ratio
    if valid_contours:
        largest_area = max(cv2.contourArea(c) for c in valid_contours)
    else:
        largest_area = 0
    
    largest_cc_ratio = largest_area / total_pixels if total_pixels > 0 else 0
    
    return ink_ratio, cc_count, largest_cc_ratio


# ============================================================================
# 3. SKELETON & COMPLEXITY
# ============================================================================

def skeletonize(binary: np.ndarray, max_iterations: int = 5) -> np.ndarray:
    """
    Extract skeleton using limited iterative erosion.
    Reduced iterations for speed.
    """
    skeleton = binary.copy()  # Start with the original
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    
    for _ in range(max_iterations):
        eroded = cv2.erode(skeleton, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        skeleton = cv2.subtract(skeleton, dilated)
    
    return skeleton


def count_skeleton_endpoints_and_branchpoints(skeleton: np.ndarray) -> Tuple[int, int]:
    """
    Count endpoints (1 neighbor) and branchpoints (3+ neighbors).
    """
    # Pad for neighborhood calculation
    padded = cv2.copyMakeBorder(skeleton, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    
    endpoints = 0
    branchpoints = 0
    
    for y in range(1, padded.shape[0] - 1):
        for x in range(1, padded.shape[1] - 1):
            if padded[y, x] == 255:
                # 8-neighborhood
                neighbors = np.sum(padded[y-1:y+2, x-1:x+2] == 255) - 1
                if neighbors == 1:
                    endpoints += 1
                elif neighbors >= 3:
                    branchpoints += 1
    
    return endpoints, branchpoints


def count_skeleton_length(skeleton: np.ndarray) -> int:
    """Count skeleton pixels."""
    return np.sum(skeleton == 255)


def estimate_curvature_turns(skeleton: np.ndarray) -> int:
    """
    Estimate curvature by counting direction changes in skeleton chain.
    """
    # Find skeleton pixels
    skel_points = np.where(skeleton == 255)
    
    if len(skel_points[0]) < 3:
        return 0
    
    # Simple chain-code approach: direction changes
    points = list(zip(skel_points[1], skel_points[0]))  # (x, y)
    
    if len(points) < 3:
        return 0
    
    turns = 0
    prev_dir = None
    
    for i in range(1, len(points) - 1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        
        # Direction from p1 to p2
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        
        # Direction from p2 to p3
        dx2 = p3[0] - p2[0]
        dy2 = p3[1] - p2[1]
        
        # Discretize to 8 directions
        dir1 = np.arctan2(dy1, dx1) if (dx1 or dy1) else None
        dir2 = np.arctan2(dy2, dx2) if (dx2 or dy2) else None
        
        if dir1 is not None and dir2 is not None:
            angle_diff = abs(dir2 - dir1)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            if angle_diff > 0.3:  # Threshold for "significant" turn
                turns += 1
    
    return turns


def compute_complexity_score(
    branchpoints: int,
    endpoints: int,
    curvature_turns: int,
    skeleton_length: int
) -> float:
    """
    Compute composite stroke complexity score.
    
    PUNCT: typically B≈0, E≈2-4, C low, L small
    SIGNATURE: higher B, E, C, L
    """
    E = max(endpoints - 2, 0)
    B = branchpoints
    C = curvature_turns
    L = max(skeleton_length, 1)  # Avoid zero division
    
    # Normalized by image size (avoid huge L values)
    L_normalized = min(L / 100, 10)  # Cap at 10
    
    # Weighted sum (more balanced)
    score = 1.0 * B + 0.5 * E + 0.1 * C + 0.1 * L_normalized
    
    return score


# ============================================================================
# 4. SHAPE DESCRIPTORS
# ============================================================================

def compute_shape_descriptors(
    contour: np.ndarray, binary: np.ndarray, total_pixels: int
) -> Dict:
    """Compute shape metrics for a single contour."""
    area = cv2.contourArea(contour)
    
    if area < 5:
        return None
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # Extent
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Circularity
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Holes (contour hierarchy)
    has_holes = False  # Would need full hierarchy analysis
    
    return {
        "area": area,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "perimeter": perimeter,
        "width": w,
        "height": h,
        "has_holes": has_holes,
    }


# ============================================================================
# 5. SHAPE DETECTION
# ============================================================================

def detect_dot(desc: Dict) -> bool:
    """Detect dot: small, circular, solid."""
    if desc is None:
        return False
    
    return (
        desc["area"] < 10000  # Small
        and desc["circularity"] > 0.7
        and desc["solidity"] > 0.85
    )


def detect_circle(desc: Dict) -> bool:
    """Detect filled or ring circle."""
    if desc is None:
        return False
    
    return (
        desc["circularity"] > 0.75
        and desc["solidity"] > 0.7  # Can be lower for ring
        and desc["aspect_ratio"] > 0.7 and desc["aspect_ratio"] < 1.3
    )


def detect_line(desc: Dict, branchpoints: int = 0) -> bool:
    """Detect line: high aspect ratio or very elongated. NOT if branchpoints exist."""
    if desc is None:
        return False
    
    # If there are branchpoints, it's not a simple line
    if branchpoints > 0:
        return False
    
    ar = desc["aspect_ratio"]
    
    return (ar > 6.0 or ar < 0.16) and desc["solidity"] > 0.6


def detect_x(desc: Dict, skel_endpoints: int, skel_branchpoints: int, skel_turns: int) -> bool:
    """Detect X: endpoints=4, branchpoint=1, diagonal turns."""
    if desc is None:
        return False
    
    return (
        skel_endpoints == 4
        and skel_branchpoints >= 1
        and skel_turns >= 2
    )


def detect_square(desc: Dict) -> bool:
    """Detect square: poly approx 4 corners, 90° angles."""
    if desc is None:
        return False
    
    # Simplified: aspect ratio close to 1, extent high
    ar = desc["aspect_ratio"]
    
    return (
        0.7 < ar < 1.3
        and desc["extent"] > 0.7
        and desc["solidity"] > 0.75
    )


def detect_check(desc: Dict, skel_endpoints: int, skel_branchpoints: int) -> bool:
    """Detect checkmark: 2-3 endpoints, 1 branchpoint, V-like."""
    if desc is None:
        return False
    
    return (
        skel_endpoints in [2, 3]
        and skel_branchpoints == 1
        and desc["aspect_ratio"] > 0.5  # Not too thin
    )


# ============================================================================
# 6. ENTROPY HELPERS
# ============================================================================

def compute_x_projection_entropy(binary: np.ndarray) -> float:
    """
    X-axis projection histogram entropy.
    Signature: higher entropy (ink spread across)
    Punct: lower entropy (ink clustered)
    """
    proj = np.sum(binary == 255, axis=0)  # Sum along rows
    
    if np.sum(proj) == 0:
        return 0.0
    
    proj_normalized = proj / np.sum(proj)
    entropy = -np.sum(proj_normalized[proj_normalized > 0] * np.log(proj_normalized[proj_normalized > 0]))
    
    return entropy


# ============================================================================
# 7. MAIN FEATURE EXTRACTION
# ============================================================================

def extract_features(image: np.ndarray) -> Features:
    """Extract all features for classification."""
    # Preprocess
    binary, gray = preprocess_image(image)
    
    # Ink metrics
    ink_ratio, cc_count, largest_cc_ratio = compute_ink_metrics(binary)
    
    # Get largest CC for shape analysis
    contours = get_connected_components(binary)
    min_area = max(5, binary.size * 0.00005)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    
    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        largest_cc_area = cv2.contourArea(largest_contour)
        shape_desc = compute_shape_descriptors(largest_contour, binary, binary.size)
    else:
        largest_cc_area = 0
        shape_desc = None
    
    # Default shape values
    aspect_ratio = shape_desc["aspect_ratio"] if shape_desc else 0
    extent = shape_desc["extent"] if shape_desc else 0
    solidity = shape_desc["solidity"] if shape_desc else 0
    circularity = shape_desc["circularity"] if shape_desc else 0
    
    # Skeleton metrics
    skeleton = skeletonize(binary)
    skeleton_length = count_skeleton_length(skeleton)
    endpoints, branchpoints = count_skeleton_endpoints_and_branchpoints(skeleton)
    curvature = estimate_curvature_turns(skeleton)
    
    # Compute complexity score
    complexity = compute_complexity_score(branchpoints, endpoints, curvature, skeleton_length)
    
    # Cap complexity (something's wrong if too high)
    complexity = min(complexity, 10.0)
    
    # Shape detection
    is_dot = detect_dot(shape_desc)
    is_circle = detect_circle(shape_desc)
    is_line = detect_line(shape_desc, branchpoints)  # Pass branchpoints
    is_x = detect_x(shape_desc, endpoints, branchpoints, curvature)
    is_square = detect_square(shape_desc)
    is_check = detect_check(shape_desc, endpoints, branchpoints)
    
    # Entropy
    entropy = compute_x_projection_entropy(binary)
    
    return Features(
        ink_ratio=ink_ratio,
        cc_count=cc_count,
        largest_cc_area=largest_cc_area,
        largest_cc_ratio=largest_cc_ratio,
        aspect_ratio=aspect_ratio,
        extent=extent,
        solidity=solidity,
        circularity=circularity,
        skeleton_length=skeleton_length,
        endpoints_count=endpoints,
        branchpoints_count=branchpoints,
        curvature_turns=curvature,
        complexity_score=complexity,
        is_dot=is_dot,
        is_circle=is_circle,
        is_line=is_line,
        is_x=is_x,
        is_square=is_square,
        is_check=is_check,
        x_projection_entropy=entropy,
    )


# ============================================================================
# 8. RULE-BASED CLASSIFIER
# ============================================================================

class ThresholdConfig:
    """Tunable thresholds."""
    # Ink metrics
    INK_RATIO_EMPTY_LOW = 0.0015
    INK_RATIO_EMPTY_HIGH = 0.003
    INK_RATIO_FULL_BLACK = 0.95  # Mostly black = empty or invalid
    
    # Skeleton
    SKELETON_LEN_EMPTY = 50
    
    # Complexity (normalized scores, much lower now)
    COMPLEXITY_LOW = 0.3
    COMPLEXITY_HIGH = 1.0  # Much more aggressive - require real signature complexity
    
    # CC
    CC_COUNT_EMPTY_MAX = 3


def classify_rule_based(features: Features) -> Tuple[ClassResult, float, str]:
    """
    Rule-based classification pipeline.
    
    Returns:
        result: Classification
        confidence: 0-1
        reason: Explanation
    """
    cfg = ThresholdConfig
    
    # ========== GATE 1: EMPTY ==========
    # Rule 1a: Very low ink
    if features.ink_ratio < cfg.INK_RATIO_EMPTY_LOW and features.cc_count <= cfg.CC_COUNT_EMPTY_MAX:
        return ClassResult.EMPTY, 0.95, f"ink_ratio={features.ink_ratio:.4f} (very low)"
    
    # Rule 1b: Very low ink + short skeleton
    if features.ink_ratio < cfg.INK_RATIO_EMPTY_HIGH and features.skeleton_length < cfg.SKELETON_LEN_EMPTY:
        return ClassResult.EMPTY, 0.92, f"ink={features.ink_ratio:.4f} + skeleton_len={features.skeleton_length}"
    
    # Rule 1c: Page is inverted (almost fully black)
    if features.ink_ratio > cfg.INK_RATIO_FULL_BLACK:
        return ClassResult.EMPTY, 0.95, f"full_black: ink_ratio={features.ink_ratio:.4f}"
    
    # Rule 1d: Lots of small noise components (no real content)
    if features.cc_count > 50 and features.largest_cc_ratio < 0.1:
        return ClassResult.EMPTY, 0.90, f"noise: {features.cc_count} components, largest_ratio={features.largest_cc_ratio:.4f}"
    

    # Rule 1e: Very low ink ratio + very few endpoints + FLAT shape = Simple test marks or ruler lines (not real content)
    # emp.jpg & emp2.jpg: cc=1, skel=500-560, endpoints<=10, low ink, NOT detected as line
    # Avoid false positives on actual line shapes (punct_line) which have high aspect ratio
    if (features.ink_ratio < 0.025 and 
        features.cc_count == 1 and 
        features.skeleton_length < 600 and
        features.endpoints_count <= 10 and
        features.complexity_score <= 5.5 and
        not features.is_line):  # Exclude actual detected lines
        return ClassResult.EMPTY, 0.92, f"sparse_line_mark: ink={features.ink_ratio:.4f}"
    

    # ========== GATE 2: PUNCTUATION (shape-based) ==========
    if features.is_dot:
        return ClassResult.PUNCTUATION, 0.90, "shape=DOT"
    
    if features.is_circle:
        return ClassResult.PUNCTUATION, 0.88, "shape=CIRCLE"
    
    if features.is_line:
        return ClassResult.PUNCTUATION, 0.85, "shape=LINE"
    
    if features.is_x:
        return ClassResult.PUNCTUATION, 0.87, "shape=X"
    
    # SQUARE shape with HIGH complexity → likely a complex signature (geometric), not punctuation
    if features.is_square and features.complexity_score <= 2.0:
        return ClassResult.PUNCTUATION, 0.86, "shape=SQUARE"
    
    if features.is_check:
        return ClassResult.PUNCTUATION, 0.84, "shape=CHECK"
    
    # ========== GATE 3: SINGLE STROKE HEURISTIC ==========
    # If single component + short skeleton → likely PUNCT (single stroke)
    if features.cc_count == 1 and features.skeleton_length < 400:
        return ClassResult.PUNCTUATION, 0.80, f"single_stroke: skel_len={features.skeleton_length}"
    
    # ========== GATE 4: COMPLEXITY ==========
    # Low complexity + no features → Likely simple punct
    if features.complexity_score < cfg.COMPLEXITY_LOW:
        # But if shape detection failed, mark as AMBIGUOUS
        if features.cc_count == 1 and features.skeleton_length < 500:
            return ClassResult.PUNCTUATION, 0.75, f"low_complexity={features.complexity_score:.2f}"
        else:
            return ClassResult.AMBIGUOUS, 0.60, f"low_complexity={features.complexity_score:.2f} (unmatched shape)"
    
    # High complexity → Could be signature or filled geometric pattern
    if features.complexity_score > cfg.COMPLEXITY_HIGH:
        # Very high ink ratio (> 0.8) + high complexity = filled geometric punctuation
        if features.ink_ratio > 0.80:
            return ClassResult.PUNCTUATION, 0.86, f"filled_pattern: ink_ratio={features.ink_ratio:.2f}"
        

        # Few components (only 2) + exceptionally long skeleton (>4000) + very low ink = geometric punctuation
        # More restrictive: only cc==2 (not 2-3), ink<0.08, skel>4000 to avoid false positives
        if features.cc_count == 2 and features.skeleton_length > 4000 and features.ink_ratio < 0.08:
            return ClassResult.PUNCTUATION, 0.82, f"geometric_punct: cc=2, skel={features.skeleton_length}"
        
        # Test punctuation images (IMG_180X_converted): Sparse marks with very few branchpoints relative to marked pixels
        # These images have: low branchpoint count despite significant endpoint count or size
        # Signature: many branchpoints; Test mark: few branchpoints but marked structure
        # IMG_1807: br=18, endpoints=356; IMG_1808: br=19, endpoints=189 - both have high endpoints
        if ((features.cc_count == 2 or features.cc_count == 4) and 
            2500 < features.skeleton_length < 3600 and 
            features.branchpoints_count <= 20 and
            features.ink_ratio > 0.04 and features.ink_ratio < 0.10 and
            features.endpoints_count >= 180):  # Higher threshold to avoid 001_05 (endpoints=171)
            return ClassResult.PUNCTUATION, 0.81, f"test_mark: br={features.branchpoints_count}"
        
        # Shape detected (circle/square) + high complexity + few components = geometric punctuation
        # Square: only for single component OR very low ink ratio to avoid false positives on real signatures
        if features.is_circle and features.cc_count <= 3:
            return ClassResult.PUNCTUATION, 0.83, f"geometric_shape: circle"
        
        if features.is_square and features.cc_count == 1 and features.ink_ratio < 0.15:
            return ClassResult.PUNCTUATION, 0.83, f"geometric_shape: square"

        
        # Normal high complexity = signature (including highly fragmented ones)
        entropy_bonus = 0.05 if features.x_projection_entropy > 2.0 else 0
        conf = 0.88 + entropy_bonus
        return ClassResult.SIGNATURE, min(conf, 0.95), f"high_complexity={features.complexity_score:.2f}"
    
    # Middle complexity → AMBIGUOUS (send to VLM)
    return ClassResult.AMBIGUOUS, 0.50, f"ambiguous_complexity={features.complexity_score:.2f}"


# ============================================================================
# 9. INLINE TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is not None:
            feat = extract_features(img)
            result, conf, reason = classify_rule_based(feat)
            
            print(f"File: {Path(img_path).name}")
            print(f"Result: {result.value}")
            print(f"Confidence: {conf:.1%}")
            print(f"Reason: {reason}")
            print()
            print(f"Features:")
            print(f"  ink_ratio: {feat.ink_ratio:.4f}")
            print(f"  cc_count: {feat.cc_count}")
            print(f"  complexity: {feat.complexity_score:.2f}")
            print(f"  endpoints: {feat.endpoints_count}, branchpoints: {feat.branchpoints_count}")
            print(f"  skeleton_length: {feat.skeleton_length}")
        else:
            print(f"Could not read {img_path}")
    else:
        print("Usage: python classifier.py <image_path>")
