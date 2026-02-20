#!/usr/bin/env python3
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class ClassificationConfig:
    COMPLEXITY_LOW: float = 0.3      # alt sinir
    COMPLEXITY_HIGH: float = 1.0     # ust sinir
    INK_RATIO_EMPTY_MAX: float = 0.0015      # bos kagit
    INK_RATIO_FILLED_MIN: float = 0.80       # dolu kagit
    CC_GEOMETRIC_MIN: int = 2         # geometrik min
    CC_GEOMETRIC_MAX: int = 5         # geometrik max
    CC_FRAGMENTED_MIN: int = 50       # parcalik
    SKELETON_LENGTH_GEOM_MIN: int = 2500  # iskelet min
    SKELETON_LENGTH_STROKE_MAX: int = 400 # iskelet max
    MAX_SKELETON_ITERATIONS: int = 5      # hiz limiti
    NOISE_COMPONENT_THRESHOLD: int = 50    # gurultu
    LARGEST_CC_RATIO_MIN: float = 0.1      # buyuk parca
    FULL_BLACK_THRESHOLD: float = 0.95     # siyah sinir
    CONFIDENCE_THRESHOLD_HIGH: float = 0.85   # guven yuksek
    CONFIDENCE_THRESHOLD_VLM: float = 0.60    # guven dusuk
    MAX_IMAGE_DIMENSION: int = 8192       # max cozunurluk
    MAX_IMAGE_SIZE_MB: float = 50         # max boyut
    PROCESSING_TIMEOUT_SEC: int = 30      # zaman asimi

@dataclass
class PreprocessingConfig:
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)
    BILATERAL_D: int = 9
    BILATERAL_SIGMA_COLOR: float = 75.0
    BILATERAL_SIGMA_SPACE: float = 75.0
    MORPH_KERNEL_SIZE: int = 3
    MORPH_ITERATIONS: int = 2
    OTSU_THRESHOLD_BUFFER: int = 5

@dataclass
class IOConfig:
    SUPPORTED_FORMATS: Tuple[str, ...] = ('png', 'jpg', 'jpeg', 'tif', 'tiff', 'heic')
    RESULTS_DIR: str = '.'
    CSV_FILENAME: str = 'vlm_full_results.csv'
    LOG_FILENAME: str = 'classification.log'
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

class Config:
    classification = ClassificationConfig()
    preprocessing = PreprocessingConfig()
    io = IOConfig()
    
    @classmethod
    def load_from_env(cls) -> None:
        try:
            if complexity_high := os.getenv('CLASSIFICATION_COMPLEXITY_HIGH'):
                cls.classification.COMPLEXITY_HIGH = float(complexity_high)
            if log_level := os.getenv('LOG_LEVEL'):
                cls.io.LOG_LEVEL = log_level
        except (ValueError, TypeError) as e:
            logger.warning(f"hata: {e}")
    
    @classmethod
    def validate(cls) -> bool:
        errors = []
        if cls.classification.COMPLEXITY_LOW >= cls.classification.COMPLEXITY_HIGH:
            errors.append("Low >= High")
        if cls.classification.INK_RATIO_EMPTY_MAX >= cls.classification.INK_RATIO_FILLED_MIN:
            errors.append("Empty >= Filled")
        if cls.classification.CC_GEOMETRIC_MIN >= cls.classification.CC_GEOMETRIC_MAX:
            errors.append("Min >= Max")
        
        if errors:
            logger.error(f"Gecersiz config")
            return False
        return True

Config.load_from_env()
if not Config.validate():
    raise ValueError("Config hatasi")
