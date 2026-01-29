import numpy as np
import pytest
import cv2
from pathlib import Path


@pytest.fixture
def empty_image() -> bytes:
    # create a blank white image
    img = np.ones((100, 200), dtype=np.uint8) * 255
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()


@pytest.fixture
def filled_image() -> bytes:
    # create an image with filled content
    img = np.ones((100, 200), dtype=np.uint8) * 255

    cv2.rectangle(img, (20, 20), (180, 80), 0, -1)
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()


@pytest.fixture
def punct_image() -> bytes:
    # create an image with punctuation-like content
    img = np.ones((100, 200), dtype=np.uint8) * 255

    cv2.circle(img, (100, 50), 5, 0, -1)
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()


@pytest.fixture
def sample_image_path(tmp_path) -> Path:
   # create an image file with filled content
    img = np.ones((100, 200), dtype=np.uint8) * 255

    cv2.rectangle(img, (20, 20), (180, 80), 0, -1)
    path = tmp_path / "test_image.png"
    cv2.imwrite(str(path), img)
    return path
