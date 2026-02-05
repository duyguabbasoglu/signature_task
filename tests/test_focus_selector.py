"""Tests for focus selector module."""
import pytest
import cv2
import numpy as np

from bbox_detector.focus_selector import (
    BoundingBox,
    BoxFocusSelector,
    FocusResult,
    select_focus_box,
    select_focus_box_detailed,
)


class TestBoundingBox:
    """Test BoundingBox dataclass."""
    
    def test_properties(self):
        box = BoundingBox(x=10, y=20, width=100, height=50)
        assert box.x2 == 110
        assert box.y2 == 70
        assert box.area == 5000
        assert box.aspect_ratio == 2.0
        assert box.center == (60, 45)
    
    def test_to_tuple(self):
        box = BoundingBox(x=10, y=20, width=100, height=50)
        assert box.to_tuple() == (10, 20, 100, 50)


class TestBoxFocusSelector:
    """Test BoxFocusSelector class."""
    
    @pytest.fixture
    def selector(self):
        return BoxFocusSelector()
    
    @pytest.fixture
    def sample_image(self):
        # Create a 500x500 grayscale image
        return np.ones((500, 500), dtype=np.uint8) * 255
    
    def test_initialization(self, selector):
        assert selector.min_area == 500
        assert selector.prefer_bottom is True
        assert selector.prefer_right is True
    
    def test_filter_valid_boxes(self, selector):
        boxes = [
            BoundingBox(x=0, y=0, width=10, height=10),  # Too small (100 < 500)
            BoundingBox(x=0, y=0, width=100, height=50),  # Valid (5000)
            BoundingBox(x=0, y=0, width=500, height=300),  # Valid (150000)
        ]
        valid = selector.filter_valid_boxes(boxes)
        assert len(valid) == 2
    
    def test_score_position(self, selector, sample_image):
        h, w = sample_image.shape[:2]
        
        # Bottom-right box should score higher
        bottom_right = BoundingBox(x=400, y=400, width=50, height=50)
        top_left = BoundingBox(x=10, y=10, width=50, height=50)
        
        score_br = selector.score_position(bottom_right, h, w)
        score_tl = selector.score_position(top_left, h, w)
        
        assert score_br > score_tl
    
    def test_score_area(self, selector):
        # Medium-sized box should score highest
        small = BoundingBox(x=0, y=0, width=10, height=10)  # 100 pixels
        medium = BoundingBox(x=0, y=0, width=100, height=100)  # 10000 pixels
        large = BoundingBox(x=0, y=0, width=500, height=500)  # 250000 pixels
        
        assert selector.score_area(medium) > selector.score_area(small)
        assert selector.score_area(medium) > selector.score_area(large)
    
    def test_score_aspect_ratio(self, selector):
        # Wider boxes should score higher (signatures are usually wider)
        wide = BoundingBox(x=0, y=0, width=200, height=50)  # ratio 4
        square = BoundingBox(x=0, y=0, width=100, height=100)  # ratio 1
        tall = BoundingBox(x=0, y=0, width=50, height=200)  # ratio 0.25
        
        assert selector.score_aspect_ratio(wide) > selector.score_aspect_ratio(square)
        assert selector.score_aspect_ratio(square) > selector.score_aspect_ratio(tall)
    
    def test_select_single_box(self, selector, sample_image):
        boxes = [BoundingBox(x=100, y=100, width=100, height=50)]
        result = selector.select_focus_box(boxes, sample_image)
        
        assert result.selected_box is not None
        assert result.selection_reason == "Only one valid box"
    
    def test_select_no_boxes(self, selector, sample_image):
        result = selector.select_focus_box([], sample_image)
        
        assert result.selected_box is None
        assert "No boxes" in result.selection_reason
    
    def test_select_multiple_boxes(self, selector, sample_image):
        # Create boxes at different positions
        boxes = [
            BoundingBox(x=10, y=10, width=100, height=50),   # Top-left
            BoundingBox(x=350, y=400, width=100, height=50),  # Bottom-right
        ]
        result = selector.select_focus_box(boxes, sample_image)
        
        # Bottom-right should be selected (prefer_bottom=True, prefer_right=True)
        assert result.selected_box is not None
        assert result.selected_box.x == 350


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def sample_image(self):
        return np.ones((500, 500), dtype=np.uint8) * 255
    
    def test_select_focus_box(self, sample_image):
        boxes = [(100, 100, 100, 50)]
        result = select_focus_box(boxes, sample_image)
        
        assert result == (100, 100, 100, 50)
    
    def test_select_focus_box_detailed(self, sample_image):
        boxes = [(100, 100, 100, 50)]
        result = select_focus_box_detailed(boxes, sample_image)
        
        assert isinstance(result, FocusResult)
        assert result.selected_box is not None
        assert len(result.all_boxes) == 1
