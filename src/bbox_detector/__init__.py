from bbox_detector.detector import (
    BoundingBoxAnalyzer,
    DetectionResult,
    AnalysisResult,
    analyze,
    analyze_bytes,
    is_signature,
    is_signature_bytes,
    get_expected_result,
    test_accuracy,
)

from bbox_detector.vlm_client import (
    VLMClient,
    VLMResult,
    VLMClassification,
    get_vlm_client,
    classify_with_vlm,
)

from bbox_detector.focus_selector import (
    BoundingBox,
    BoxFocusSelector,
    FocusResult,
    select_focus_box,
    select_focus_box_detailed,
)

__version__ = "1.0.0"
__all__ = [
    "BoundingBoxAnalyzer",
    "DetectionResult",
    "AnalysisResult",
    "analyze",
    "analyze_bytes",
    "is_signature",
    "is_signature_bytes",
    "get_expected_result",
    "test_accuracy",
    "VLMClient",
    "VLMResult",
    "VLMClassification",
    "get_vlm_client",
    "classify_with_vlm",
    "BoundingBox",
    "BoxFocusSelector",
    "FocusResult",
    "select_focus_box",
    "select_focus_box_detailed",
]


