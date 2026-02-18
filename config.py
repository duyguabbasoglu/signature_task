#!/usr/bin/env python3
"""
Signature Classification System - Configuration Management

This module centralizes all configuration constants and environment settings
for the signature classification pipeline. All security-critical parameters
and thresholds are managed here for easy audit and modification.

Author: AI Assistant
Date: February 18, 2026
Version: 1.0.0
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig:
    """Classification algorithm thresholds and parameters."""
    
    # Complexity thresholds (normalized 0-10 scale)
    COMPLEXITY_LOW: float = 0.3      # Simple punctuation boundary
    COMPLEXITY_HIGH: float = 1.0     # Signature complexity boundary
    
    # Ink metrics thresholds
    INK_RATIO_EMPTY_MAX: float = 0.0015      # Threshold for empty detection
    INK_RATIO_FILLED_MIN: float = 0.80       # Filled pattern detection
    
    # Connected components thresholds
    CC_GEOMETRIC_MIN: int = 2         # Minimum for geometric punctuation
    CC_GEOMETRIC_MAX: int = 5         # Maximum for geometric punctuation
    CC_FRAGMENTED_MIN: int = 50       # Fragmentation threshold
    
    # Skeleton analysis thresholds
    SKELETON_LENGTH_GEOM_MIN: int = 2500  # Geometric punctuation minimum
    SKELETON_LENGTH_STROKE_MAX: int = 400 # Single stroke punctuation maximum
    MAX_SKELETON_ITERATIONS: int = 5      # Iteration limit for speed
    
    # Noise detection
    NOISE_COMPONENT_THRESHOLD: int = 50    # Component count for noise detection
    LARGEST_CC_RATIO_MIN: float = 0.1      # Largest component ratio threshold
    FULL_BLACK_THRESHOLD: float = 0.95     # Full black/noise threshold
    
    # Confidence thresholds (for reporting)
    CONFIDENCE_THRESHOLD_HIGH: float = 0.85   # Route to human review if below
    CONFIDENCE_THRESHOLD_VLM: float = 0.60    # Route to VLM if in ambiguous zone
    
    # Processing limits (security/DOS prevention)
    MAX_IMAGE_DIMENSION: int = 8192       # Maximum image resolution
    MAX_IMAGE_SIZE_MB: float = 50         # Maximum file size in MB
    PROCESSING_TIMEOUT_SEC: int = 30      # Maximum processing time per image


@dataclass
class PreprocessingConfig:
    """Image preprocessing parameters."""
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)
    
    # Bilateral denoise
    BILATERAL_D: int = 9
    BILATERAL_SIGMA_COLOR: float = 75.0
    BILATERAL_SIGMA_SPACE: float = 75.0
    
    # Morphological operations
    MORPH_KERNEL_SIZE: int = 3
    MORPH_ITERATIONS: int = 2
    
    # Binarization
    OTSU_THRESHOLD_BUFFER: int = 5  # Buffer for Otsu adjustment


@dataclass
class IOConfig:
    """Input/Output configuration."""
    
    # Supported formats
    SUPPORTED_FORMATS: Tuple[str, ...] = ('png', 'jpg', 'jpeg', 'tif', 'tiff', 'heic')
    
    # Output paths (relative to project root)
    RESULTS_DIR: str = '.'
    CSV_FILENAME: str = 'vlm_full_results.csv'
    LOG_FILENAME: str = 'classification.log'
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class Config:
    """Master configuration class with env variable override support."""
    
    classification = ClassificationConfig()
    preprocessing = PreprocessingConfig()
    io = IOConfig()
    
    @classmethod
    def load_from_env(cls) -> None:
        """Load configuration from environment variables (for deployment)."""
        try:
            if complexity_high := os.getenv('CLASSIFICATION_COMPLEXITY_HIGH'):
                cls.classification.COMPLEXITY_HIGH = float(complexity_high)
            
            if log_level := os.getenv('LOG_LEVEL'):
                cls.io.LOG_LEVEL = log_level
            
            logger.info(f"Configuration loaded with env overrides: {cls.__dict__}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error loading config from env: {e}. Using defaults.")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration consistency."""
        errors = []
        
        if cls.classification.COMPLEXITY_LOW >= cls.classification.COMPLEXITY_HIGH:
            errors.append("COMPLEXITY_LOW must be < COMPLEXITY_HIGH")
        
        if cls.classification.INK_RATIO_EMPTY_MAX >= cls.classification.INK_RATIO_FILLED_MIN:
            errors.append("INK_RATIO_EMPTY_MAX must be < INK_RATIO_FILLED_MIN")
        
        if cls.classification.CC_GEOMETRIC_MIN >= cls.classification.CC_GEOMETRIC_MAX:
            errors.append("CC_GEOMETRIC_MIN must be < CC_GEOMETRIC_MAX")
        
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True


# Initialize and validate on module load
Config.load_from_env()
if not Config.validate():
    raise ValueError("Configuration validation failed")
