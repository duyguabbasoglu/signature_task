"""
Classification accuracy tests using data folder.
Tests that empty/punctuation detection works correctly.
"""
import pytest
from pathlib import Path


class TestGetExpectedResult:
    """Test the filename-based expected result function."""
    
    def test_empty_files(self):
        from bbox_detector import get_expected_result
        assert get_expected_result("empty_black.png") is False
        assert get_expected_result("empty_white.png") is False
        assert get_expected_result("empty_noise.png") is False
    
    def test_punct_files(self):
        from bbox_detector import get_expected_result
        assert get_expected_result("punct_dot.png") is False
        assert get_expected_result("punct_circle.png") is False
        assert get_expected_result("punct_x.png") is False
        assert get_expected_result("punct_line.png") is False
    
    def test_img_converted_files(self):
        from bbox_detector import get_expected_result
        assert get_expected_result("IMG_1807_converted.png") is False
        assert get_expected_result("IMG_1808_converted.png") is False
    
    def test_signature_files(self):
        from bbox_detector import get_expected_result
        assert get_expected_result("B-S-1-F-01.tif") is True
        assert get_expected_result("B-S-100-G-01.tif") is True
        assert get_expected_result("H-S-1-F-01.tif") is True


class TestIsSignatureEmpty:
    """Test that empty images return False."""
    
    def test_empty_white(self, empty_image):
        from bbox_detector import is_signature_bytes
        result = is_signature_bytes(empty_image, use_vlm=False)
        assert result is False
    
    def test_empty_detection_no_vlm(self, empty_image):
        """Empty detection should work without VLM."""
        from bbox_detector import is_signature_bytes
        result = is_signature_bytes(empty_image, use_vlm=False)
        assert result is False


class TestIsSignatureFilled:
    """Test that filled (signature) images return True."""
    
    def test_filled_image_no_vlm(self, filled_image):
        """Large filled content should be classified as signature."""
        from bbox_detector import is_signature_bytes
        # Without VLM, rule-based uses threshold
        result = is_signature_bytes(filled_image, use_vlm=False)
        assert result is True


class TestIsSignaturePunct:
    """Test that punctuation images return False."""
    
    def test_punct_image_no_vlm(self, punct_image):
        """Small punctuation should be classified as not signature."""
        from bbox_detector import is_signature_bytes
        result = is_signature_bytes(punct_image, use_vlm=False)
        assert result is False


class TestDataFolderAccuracy:
    """Test accuracy on actual data folder."""
    
    @pytest.fixture
    def data_dir(self):
        """Get path to data directory."""
        base = Path(__file__).parent.parent / "data"
        if base.exists():
            return str(base)
        return None
    
    def test_empty_detection_100_percent(self, data_dir):
        """Empty files must have 100% accuracy."""
        if data_dir is None:
            pytest.skip("Data directory not found")
        
        from bbox_detector import test_accuracy
        results = test_accuracy(data_dir, use_vlm=False)
        
        empty_results = results["by_category"]["empty"]
        assert empty_results["accuracy"] == 100.0, \
            f"Empty accuracy is {empty_results['accuracy']}%, errors: {empty_results['errors']}"
    
    def test_overall_accuracy_threshold(self, data_dir):
        """Overall accuracy should be reasonable with rule-based."""
        if data_dir is None:
            pytest.skip("Data directory not found")
        
        from bbox_detector import test_accuracy
        results = test_accuracy(data_dir, use_vlm=False)
        
        # Without VLM, we expect at least 90% accuracy
        assert results["accuracy"] >= 90.0, \
            f"Overall accuracy is {results['accuracy']}%"
