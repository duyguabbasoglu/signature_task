# 🚀 Execution Commands - Signature Classification System

**Quick Reference**: All commands needed to run the system

---

## 📦 Setup & Dependencies

### Initial Setup (First Time)
```bash
# Clone repository
git clone <REPOSITORY_URL> signature_task
cd signature_task

# Create Python virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/Mac

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import cv2, numpy, PIL, pandas; print('✓ All dependencies OK')"
```

---

## 🎯 Core Commands

### 1. Classify Single Image
```python
# Python script
from classifier import load_image_robust, extract_features, classify_rule_based

img = load_image_robust("path/to/image.png")
features = extract_features(img)
result, confidence, reasoning = classify_rule_based(features)
print(f"Classification: {result.value}")
print(f"Confidence: {confidence:.1%}")
print(f"Reasoning: {reasoning}")
```

Or save as `classify_single.py`:
```bash
python classify_single.py
```

### 2. Batch Test (155 Images)
```bash
python full_dataset_test.py
# Output: vlm_full_results.csv
```

Output columns:
- filename, expected_class, result_class, confidence, reasoning
- ink_ratio, connected_components, complexity_score, skeleton_length, endpoints, branchpoints, correct

### 3. Quick API Test
```bash
python test_api_simple.py
# Tests: LLM API connectivity (200 OK expected)
```

### 4. Validate Dataset
```bash
python analyze_errors.py
# Shows: Classification accuracy, error analysis, metrics breakdown
```

### 5. Run All Unit Tests
```bash
pytest tests/ -v --tb=short
# Runs: classifier tests, detector tests, focus selector tests
```

### 6. Run Specific Test File
```bash
pytest tests/test_detector.py -v
pytest tests/test_classification_accuracy.py -v
```

---

## 🔧 Advanced Commands

### Run with Custom Configuration
```bash
# Override thresholds via environment variables
export CONFIG_COMPLEXITY_HIGH=1.2
export CONFIG_INK_RATIO_EMPTY_MAX=0.002
python full_dataset_test.py
```

### Start API Server
```bash
uvicorn bbox_detector.api.server:app --reload --host 0.0.0.0 --port 8000
# API available at: http://localhost:8000
# Docs at: http://localhost:8000/docs
```

### Debug Specific Image
```bash
# Check image properties
python -c "
from classifier import load_image_robust, extract_features
img = load_image_robust('data/image.heic')
features = extract_features(img)
print(f'Features: {features}')
"
```

### Export Results as CSV
```bash
python full_dataset_test.py
# Output: vlm_full_results.csv with all metrics
```

### Run Classification on Folder
```python
import os
from pathlib import Path
from classifier import load_image_robust, extract_features, classify_rule_based

data_dir = "data/"
for img_file in Path(data_dir).glob("*.heic"):
    img = load_image_robust(str(img_file))
    result = classify_rule_based(extract_features(img))
    print(f"{img_file.name}: {result[0].value}")
```

---

## 📊 Testing & Validation

### Full Test Suite
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow"
```

### Validate 155-Image Dataset
```bash
# Complete validation
python full_dataset_test.py

# Check results
python analyze_errors.py

# Show summary
python show_summary.py  # If available
```

---

## 🐛 Troubleshooting Commands

### Check Python Version
```bash
python --version
# Expected: Python 3.12.x
```

### Verify OpenCV Installation
```bash
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

### Test HEIC Support
```bash
python -c "
import pillow_heif
pillow_heif.register_heif_opener()
from PIL import Image
img = Image.open('test.heic')
print(f'HEIC loaded: {img.size}')
"
```

### Clear Cache & Rebuild
```bash
# Remove Python cache
Remove-Item -Recurse -Force __pycache__
Remove-Item -Recurse -Force .pytest_cache

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

---

## 📁 File Outputs

| Command | Output | Purpose |
|---------|--------|---------|
| `python full_dataset_test.py` | `vlm_full_results.csv` | Classification results for all images |
| `python analyze_errors.py` | Console output | Error analysis & metrics |
| API server | `logs/` (if enabled) | Production logs |

---

## ✅ Verification Checklist

After running commands, verify:

```bash
# 1. CSV exists and has data
ls -la vlm_full_results.csv
head -5 vlm_full_results.csv

# 2. Accuracy ≥ 91%
# Check line: "TOTAL | 141/155 = 91.0%"

# 3. All tests pass
pytest tests/ -q
# Expected: "X passed"

# 4. No errors in logs
# Check for ERROR entries
```

---

## 🔄 Continuous Monitoring

### Log Monitoring (if running server)
```bash
# Tail live logs (Linux/Mac)
tail -f logs/server.log

# Windows: Use VS Code terminal or PowerShell ISE
Get-Content -Path logs/server.log -Tail 20 -Wait
```

### Performance Monitoring
```bash
# Track processing time
python -c "
import time
from classifier import load_image_robust, classify_rule_based, extract_features

start = time.time()
img = load_image_robust('image.heic')
result = classify_rule_based(extract_features(img))
elapsed = time.time() - start
print(f'Processed in {elapsed:.2f}s')
"
```

---

## 📋 Common Workflows

### Complete Validation (5 min)
```bash
# 1. Run all tests
pytest tests/ -q

# 2. Test 155-image dataset
python full_dataset_test.py

# 3. Analyze results
python analyze_errors.py

# Done! Check vlm_full_results.csv
```

### Production Deployment
```bash
# 1. Activate environment
.venv\Scripts\activate

# 2. Start server
uvicorn bbox_detector.api.server:app --host 0.0.0.0 --port 8000

# 3. Monitor (in separate terminal)
# curl http://localhost:8000/health
```

### Debug Single Failure
```bash
# 1. Find failed image in vlm_full_results.csv
# 2. Run debug check
python -c "
from classifier import load_image_robust, extract_features, classify_rule_based

img = load_image_robust('data/IMG_FAILED.heic')
features = extract_features(img)
result = classify_rule_based(features)
print(f'Result: {result}')
print(f'Features: {features}')
"
```

---

## 🎓 Learning Resources

- **TECHNICAL_REPORT.md** - Architecture & algorithm details
- **EXECUTIVE_SUMMARY.md** - Business metrics & ROI
- **classifier.py** - Source code with inline documentation
- **config.py** - Configuration parameters and defaults

---

**Last Updated**: Feb 18, 2026 | **Status**: ✅ Production Ready
