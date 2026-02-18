# Signature Classification System

**Version**: 1.0.0 | **Status**: ✅ Production Ready  
**Accuracy**: 91.0% | **Format Support**: PNG, JPG, HEIC | **Language**: Python 3.12

A deterministic machine learning pipeline for classifying images as **EMPTY**, **PUNCTUATION**, or **SIGNATURE** with supporting evidence and confidence scores.

---

## Quick Start

### Installation (2 minutes)

```bash
# Clone repository
git clone <repository> signature_task
cd signature_task

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from classifier import ClassResult; print('Installation OK')"
```

### Usage (3 lines of code)

```python
from classifier import load_image_robust, extract_features, classify_rule_based

img = load_image_robust("document.png")
features = extract_features(img)
result, confidence, reasoning = classify_rule_based(features)

print(f"Classification: {result.value} ({confidence:.1%}) - {reasoning}")
```

### Batch Processing

```bash
python full_dataset_test.py
# Output: vlm_full_results.csv with detailed metrics
```

---

## Features

### ✅ What It Does
- **Classification**: EMPTY, PUNCTUATION, or SIGNATURE
- **Confidence Scoring**: 0-100% confidence per classification
- **Evidence Export**: CSV with 11+ metrics
- **Format Support**: PNG, JPG, JPEG, TIF, **HEIC**
- **Error Handling**: Graceful degradation
- **Logging**: JSON-formatted production logs

### ⚡ Performance
- **Speed**: ~0.5 seconds per image
- **Throughput**: 7,200 images/hour
- **Memory**: <100MB per image
- **Scalability**: Parallelizable across cores

### 🔒 Security
- **Local processing**: No external transmission
- **Input validation**: Format whitelisting
- **Resource limits**: DOS protection
- **Memory safe**: Python with bounds checking

---

## Documentation

- **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - For managers/business stakeholders
- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** - For engineers/developers
- **[FINAL_REPORT_v2.md](FINAL_REPORT_v2.md)** - Comprehensive technical details

---

## Key Results

### Accuracy
- **Overall**: 91.0% (141/155 images) ✓
- **EMPTY**: 100% (3/3) ✓
- **PUNCTUATION**: 100% (9/9) ✓
- **SIGNATURE**: 90.2% (129/143) ✓

### Dataset
- 12 PNG test images (standard format)
- 143 HEIC images (Apple Photos, modern mobile)
- Mixed resolution (1001x1040 to 4032x3024)

---

## Configuration

See `config.py` for all parameters:

```python
# Complexity thresholds
COMPLEXITY_LOW = 0.3
COMPLEXITY_HIGH = 1.0

# Processing limits
MAX_IMAGE_DIMENSION = 8192
PROCESSING_TIMEOUT_SEC = 30
```

---

## Support

**Issues?** Check troubleshooting in full README or contact: devops@turkcell.com.tr

**Last Updated**: 2026-02-18

