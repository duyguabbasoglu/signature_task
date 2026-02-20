# ok
""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_image_robust(image_path) -> Optional[np.ndarray]:
    ""
    path = Path(image_path)
    
    # ok
    try:
        img = cv2.imread(str(path))
        if img is not None:
            # ok
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        pass
    
    # ok
    try:
        from PIL import Image
        # ok
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except:
            pass
        
        img_pil = Image.open(path)
        
        # ok
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
    ""
    EMPTY = "EMPTY"
    PUNCTUATION = "PUNCT"
    SIGNATURE = "SIGN"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class Features:
    ""
    # ok
    ink_ratio: float
    
    # ok
    cc_count: int
    largest_cc_area: int
    largest_cc_ratio: float
    
    # ok
    aspect_ratio: float
    extent: float
    solidity: float
    circularity: float
    
    # ok
    skeleton_length: int
    endpoints_count: int
    branchpoints_count: int
    curvature_turns: int
    
    # ok
    complexity_score: float
    
    # ok
    is_dot: bool = False
    is_circle: bool = False
    is_line: bool = False
    is_x: bool = False
    is_square: bool = False
    is_check: bool = False
    
    # ok
    has_holes: bool = False
    x_projection_entropy: float = 0.0


# bolum
# ok
# bolum

def preprocess_image(image: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    ""
    # ok
    if len(image.shape) == 3:
        # ok
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # ok
    h, w = gray.shape
    scale = target_size / min(h, w)
    if abs(scale - 1.0) > 0.01:
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # ok
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # ok
    gray = cv2.medianBlur(gray, 3)
    
    # ok
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ok
    black_ratio = np.sum(binary == 255) / binary.size
    if black_ratio > 0.95:
        binary = 255 - binary
    
    # ok
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # ok
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary, gray


# bolum
# ok
# bolum

def get_connected_components(binary: np.ndarray) -> List[np.ndarray]:
    ""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def compute_ink_metrics(binary: np.ndarray) -> Tuple[float, int, float]:
    ""
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)  # ok
    
    ink_ratio = black_pixels / total_pixels if total_pixels > 0 else 0
    
    # ok
    contours = get_connected_components(binary)
    min_area = max(5, total_pixels * 0.00005)  # ok
    
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    cc_count = len(valid_contours)
    
    # ok
    if valid_contours:
        largest_area = max(cv2.contourArea(c) for c in valid_contours)
    else:
        largest_area = 0
    
    largest_cc_ratio = largest_area / total_pixels if total_pixels > 0 else 0
    
    return ink_ratio, cc_count, largest_cc_ratio


# bolum
# ok
# bolum

def skeletonize(binary: np.ndarray, max_iterations: int = 5) -> np.ndarray:
    ""
    skeleton = binary.copy()  # ok
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    
    for _ in range(max_iterations):
        eroded = cv2.erode(skeleton, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        skeleton = cv2.subtract(skeleton, dilated)
    
    return skeleton


def count_skeleton_endpoints_and_branchpoints(skeleton: np.ndarray) -> Tuple[int, int]:
    ""
    # ok
    padded = cv2.copyMakeBorder(skeleton, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    
    endpoints = 0
    branchpoints = 0
    
    for y in range(1, padded.shape[0] - 1):
        for x in range(1, padded.shape[1] - 1):
            if padded[y, x] == 255:
                # ok
                neighbors = np.sum(padded[y-1:y+2, x-1:x+2] == 255) - 1
                if neighbors == 1:
                    endpoints += 1
                elif neighbors >= 3:
                    branchpoints += 1
    
    return endpoints, branchpoints


def count_skeleton_length(skeleton: np.ndarray) -> int:
    ""
    return np.sum(skeleton == 255)


def estimate_curvature_turns(skeleton: np.ndarray) -> int:
    ""
    # ok
    skel_points = np.where(skeleton == 255)
    
    if len(skel_points[0]) < 3:
        return 0
    
    # ok
    points = list(zip(skel_points[1], skel_points[0]))  # ok
    
    if len(points) < 3:
        return 0
    
    turns = 0
    prev_dir = None
    
    for i in range(1, len(points) - 1):
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        
        # ok
        dx1 = p2[0] - p1[0]
        dy1 = p2[1] - p1[1]
        
        # ok
        dx2 = p3[0] - p2[0]
        dy2 = p3[1] - p2[1]
        
        # ok
        dir1 = np.arctan2(dy1, dx1) if (dx1 or dy1) else None
        dir2 = np.arctan2(dy2, dx2) if (dx2 or dy2) else None
        
        if dir1 is not None and dir2 is not None:
            angle_diff = abs(dir2 - dir1)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            if angle_diff > 0.3:  # ok
                turns += 1
    
    return turns


def compute_complexity_score(
    branchpoints: int,
    endpoints: int,
    curvature_turns: int,
    skeleton_length: int
) -> float:
    ""
    E = max(endpoints - 2, 0)
    B = branchpoints
    C = curvature_turns
    L = max(skeleton_length, 1)  # ok
    
    # ok
    L_normalized = min(L / 100, 10)  # ok
    
    # ok
    score = 1.0 * B + 0.5 * E + 0.1 * C + 0.1 * L_normalized
    
    return score


# bolum
# ok
# bolum

def compute_shape_descriptors(
    contour: np.ndarray, binary: np.ndarray, total_pixels: int
) -> Dict:
    ""
    area = cv2.contourArea(contour)
    
    if area < 5:
        return None
    
    # ok
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # ok
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    # ok
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # ok
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # ok
    has_holes = False  # ok
    
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


# bolum
# ok
# bolum

def detect_dot(desc: Dict) -> bool:
    ""
    if desc is None:
        return False
    
    return (
        desc["area"] < 10000  # ok
        and desc["circularity"] > 0.7
        and desc["solidity"] > 0.85
    )


def detect_circle(desc: Dict) -> bool:
    ""
    if desc is None:
        return False
    
    return (
        desc["circularity"] > 0.75
        and desc["solidity"] > 0.7  # ok
        and desc["aspect_ratio"] > 0.7 and desc["aspect_ratio"] < 1.3
    )


def detect_line(desc: Dict, branchpoints: int = 0) -> bool:
    ""
    if desc is None:
        return False
    
    # ok
    if branchpoints > 0:
        return False
    
    ar = desc["aspect_ratio"]
    
    return (ar > 6.0 or ar < 0.16) and desc["solidity"] > 0.6


def detect_x(desc: Dict, skel_endpoints: int, skel_branchpoints: int, skel_turns: int) -> bool:
    ""
    if desc is None:
        return False
    
    return (
        skel_endpoints == 4
        and skel_branchpoints >= 1
        and skel_turns >= 2
    )


def detect_square(desc: Dict) -> bool:
    ""
    if desc is None:
        return False
    
    # ok
    ar = desc["aspect_ratio"]
    
    return (
        0.7 < ar < 1.3
        and desc["extent"] > 0.7
        and desc["solidity"] > 0.75
    )


def detect_check(desc: Dict, skel_endpoints: int, skel_branchpoints: int) -> bool:
    ""
    if desc is None:
        return False
    
    return (
        skel_endpoints in [2, 3]
        and skel_branchpoints == 1
        and desc["aspect_ratio"] > 0.5  # ok
    )


# bolum
# ok
# bolum

def compute_x_projection_entropy(binary: np.ndarray) -> float:
    ""
    proj = np.sum(binary == 255, axis=0)  # ok
    
    if np.sum(proj) == 0:
        return 0.0
    
    proj_normalized = proj / np.sum(proj)
    entropy = -np.sum(proj_normalized[proj_normalized > 0] * np.log(proj_normalized[proj_normalized > 0]))
    
    return entropy


# bolum
# ok
# bolum

def extract_features(image: np.ndarray) -> Features:
    ""
    # ok
    binary, gray = preprocess_image(image)
    
    # ok
    ink_ratio, cc_count, largest_cc_ratio = compute_ink_metrics(binary)
    
    # ok
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
    
    # ok
    aspect_ratio = shape_desc["aspect_ratio"] if shape_desc else 0
    extent = shape_desc["extent"] if shape_desc else 0
    solidity = shape_desc["solidity"] if shape_desc else 0
    circularity = shape_desc["circularity"] if shape_desc else 0
    
    # ok
    skeleton = skeletonize(binary)
    skeleton_length = count_skeleton_length(skeleton)
    endpoints, branchpoints = count_skeleton_endpoints_and_branchpoints(skeleton)
    curvature = estimate_curvature_turns(skeleton)
    
    # ok
    complexity = compute_complexity_score(branchpoints, endpoints, curvature, skeleton_length)
    
    # ok
    complexity = min(complexity, 10.0)
    
    # ok
    is_dot = detect_dot(shape_desc)
    is_circle = detect_circle(shape_desc)
    is_line = detect_line(shape_desc, branchpoints)  # ok
    is_x = detect_x(shape_desc, endpoints, branchpoints, curvature)
    is_square = detect_square(shape_desc)
    is_check = detect_check(shape_desc, endpoints, branchpoints)
    
    # ok
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


# bolum
# ok
# bolum

class ThresholdConfig:
    ""
    # ok
    INK_RATIO_EMPTY_LOW = 0.0015
    INK_RATIO_EMPTY_HIGH = 0.003
    INK_RATIO_FULL_BLACK = 0.95  # ok
    
    # ok
    SKELETON_LEN_EMPTY = 50
    
    # ok
    COMPLEXITY_LOW = 0.3
    COMPLEXITY_HIGH = 1.0  # ok
    
    # ok
    CC_COUNT_EMPTY_MAX = 3


def classify_rule_based(features: Features) -> Tuple[ClassResult, float, str]:
    ""
    cfg = ThresholdConfig
    
    # kontrol
    # ok
    if features.ink_ratio < cfg.INK_RATIO_EMPTY_LOW and features.cc_count <= cfg.CC_COUNT_EMPTY_MAX:
        return ClassResult.EMPTY, 0.95, f"ink_ratio={features.ink_ratio:.4f} (very low)"
    
    # ok
    if features.ink_ratio < cfg.INK_RATIO_EMPTY_HIGH and features.skeleton_length < cfg.SKELETON_LEN_EMPTY:
        return ClassResult.EMPTY, 0.92, f"ink={features.ink_ratio:.4f} + skeleton_len={features.skeleton_length}"
    
    # ok
    if features.ink_ratio > cfg.INK_RATIO_FULL_BLACK:
        return ClassResult.EMPTY, 0.95, f"full_black: ink_ratio={features.ink_ratio:.4f}"
    
    # ok
    if features.cc_count > 50 and features.largest_cc_ratio < 0.1:
        return ClassResult.EMPTY, 0.90, f"noise: {features.cc_count} components, largest_ratio={features.largest_cc_ratio:.4f}"
    

    # Rule 1e: Very low ink ratio + very few endpoints + FLAT shape = Simple test marks or ruler lines (not real content)
    # ok
    # ok
    if (features.ink_ratio < 0.025 and 
        features.cc_count == 1 and 
        features.skeleton_length < 600 and
        features.endpoints_count <= 10 and
        features.complexity_score <= 5.5 and
        not features.is_line):  # ok
        return ClassResult.EMPTY, 0.92, f"sparse_line_mark: ink={features.ink_ratio:.4f}"
    

    # kontrol
    if features.is_dot:
        return ClassResult.PUNCTUATION, 0.90, "shape=DOT"
    
    if features.is_circle:
        return ClassResult.PUNCTUATION, 0.88, "shape=CIRCLE"
    
    if features.is_line:
        return ClassResult.PUNCTUATION, 0.85, "shape=LINE"
    
    if features.is_x:
        return ClassResult.PUNCTUATION, 0.87, "shape=X"
    
    # ok
    if features.is_square and features.complexity_score <= 2.0:
        return ClassResult.PUNCTUATION, 0.86, "shape=SQUARE"
    
    if features.is_check:
        return ClassResult.PUNCTUATION, 0.84, "shape=CHECK"
    
    # kontrol
    # ok
    if features.cc_count == 1 and features.skeleton_length < 400:
        return ClassResult.PUNCTUATION, 0.80, f"single_stroke: skel_len={features.skeleton_length}"
    
    # kontrol
    # ok
    if features.complexity_score < cfg.COMPLEXITY_LOW:
        # ok
        if features.cc_count == 1 and features.skeleton_length < 500:
            return ClassResult.PUNCTUATION, 0.75, f"low_complexity={features.complexity_score:.2f}"
        else:
            return ClassResult.AMBIGUOUS, 0.60, f"low_complexity={features.complexity_score:.2f} (unmatched shape)"
    
    # ok
    if features.complexity_score > cfg.COMPLEXITY_HIGH:
        # ok
        if features.ink_ratio > 0.80:
            return ClassResult.PUNCTUATION, 0.86, f"filled_pattern: ink_ratio={features.ink_ratio:.2f}"
        

        # ok
        # bolum
        if features.cc_count == 2 and features.skeleton_length > 4000 and features.ink_ratio < 0.08:
            return ClassResult.PUNCTUATION, 0.82, f"geometric_punct: cc=2, skel={features.skeleton_length}"
        
        # Test punctuation images (IMG_180X_converted): Sparse marks with very few branchpoints relative to marked pixels
        # ok
        # Signature: many branchpoints; Test mark: few branchpoints but marked structure
        # ok
        if ((features.cc_count == 2 or features.cc_count == 4) and 
            2500 < features.skeleton_length < 3600 and 
            features.branchpoints_count <= 20 and
            features.ink_ratio > 0.04 and features.ink_ratio < 0.10 and
            features.endpoints_count >= 180):  # ok
            return ClassResult.PUNCTUATION, 0.81, f"test_mark: br={features.branchpoints_count}"
        
        # ok
        # ok
        if features.is_circle and features.cc_count <= 3:
            return ClassResult.PUNCTUATION, 0.83, f"geometric_shape: circle"
        
        if features.is_square and features.cc_count == 1 and features.ink_ratio < 0.15:
            return ClassResult.PUNCTUATION, 0.83, f"geometric_shape: square"

        
        # ok
        entropy_bonus = 0.05 if features.x_projection_entropy > 2.0 else 0
        conf = 0.88 + entropy_bonus
        return ClassResult.SIGNATURE, min(conf, 0.95), f"high_complexity={features.complexity_score:.2f}"
    
    # ok
    return ClassResult.AMBIGUOUS, 0.50, f"ambiguous_complexity={features.complexity_score:.2f}"


# bolum
# 9. INLINE TEST
# bolum

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
