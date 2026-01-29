import pytest
from bbox_detector import (
    BoundingBoxAnalyzer,
    DetectionResult,
    analyze_bytes,
)


class TestDetectionResult:
    def test_enum_values(self):
        # unit test
        assert DetectionResult.EMPTY.value == "EMPTY"
        assert DetectionResult.FILLED_PUNCT.value == "FILLED_PUNCT"
        assert DetectionResult.FILLED_OTHER.value == "FILLED_OTHER"


class TestBoundingBoxAnalyzer:
    # bound box analyzer
    def test_initialization(self):
        analyzer = BoundingBoxAnalyzer()
        assert analyzer.empty_threshold == 50
        assert analyzer.punct_max_area == 500

    def test_custom_thresholds(self):
        analyzer = BoundingBoxAnalyzer(
            empty_threshold=100,
            punct_max_area=1000
        )
        assert analyzer.empty_threshold == 100
        assert analyzer.punct_max_area == 1000

    def test_is_empty_true(self):
        analyzer = BoundingBoxAnalyzer()
        assert analyzer.is_empty(30) is True

    def test_is_empty_false(self):
        analyzer = BoundingBoxAnalyzer()
        assert analyzer.is_empty(100) is False


class TestAnalyzeBytes:
    def test_empty_image(self, empty_image):
        result = analyze_bytes(empty_image)
        assert result.is_empty is True
        assert result.result == DetectionResult.EMPTY
        assert result.is_punctuation is None

    def test_filled_image(self, filled_image):
        result = analyze_bytes(filled_image)
        assert result.is_empty is False
        assert result.result == DetectionResult.FILLED_OTHER
        assert result.is_punctuation is False

    def test_punct_image(self, punct_image):
        result = analyze_bytes(punct_image)
        assert result.is_empty is False
        assert result.result in [
            DetectionResult.FILLED_PUNCT, DetectionResult.FILLED_OTHER]

    def test_result_to_dict(self, empty_image):
        result = analyze_bytes(empty_image)
        d = result.to_dict()
        assert "result" in d
        assert "is_empty" in d
        assert "confidence" in d
        assert "total_area" in d


class TestAnalyzeFile:
    def test_analyze_file(self, sample_image_path):
        from bbox_detector import analyze
        result = analyze(str(sample_image_path))
        assert result.is_empty is False
        assert result.result == DetectionResult.FILLED_OTHER

    def test_file_not_found(self):
        from bbox_detector.detector import analyze
        with pytest.raises(ValueError):
            analyze("/nonexistent/path/image.png")


class TestConfidence:
    # confidence scores
    def test_empty_confidence(self, empty_image):
        result = analyze_bytes(empty_image)
        assert result.confidence >= 0.9

    def test_filled_confidence(self, filled_image):
        result = analyze_bytes(filled_image)
        assert result.confidence >= 0.8
