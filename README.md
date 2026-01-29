# Bounding Box Detector

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bounding box içeriklerini tespit eden lightweight bir sistem. OCR pipeline'ları ile entegre çalışacak şekilde tasarlanmıştır.

## Kurulum

pip install -e .

# Klasör analizi

bbox-detect data/

### REST API

make serve

# → http://localhost:8000/docs

**Endpoints:**
| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/health` | Health check |
| POST | `/analyze` | File upload ile analiz |
| POST | `/analyze/base64` | Base64 ile analiz |

**Örnek:**
curl -X POST http://localhost:8000/analyze \
 -F "file=@image.png"

## Test

make test

## 📁 Proje Yapısı

bbox-detector/
├── src/
│ └── bbox_detector/
│ ├── **init**.py # Package exports
│ ├── detector.py # Core detection logic
│ ├── cli.py # CLI interface
│ ├── models/ # CNN models
│ └── api/ # FastAPI server
├── tests/ # Pytest tests
├── config/ # Configuration
├── scripts/ # Utility scripts
├── pyproject.toml # Dependencies
├── Makefile # Easy commands
└── data/ # Test images

## ⚙️ Makefile Komutları

```bash
make install      # Core bağımlılıkları yükle
make install-dev  # Dev bağımlılıkları dahil yükle
make test         # Testleri çalıştır
make serve        # API sunucusu başlat
make analyze      # data/ klasörünü analiz et
make clean        # Cache temizle
```

## LLM Endpoint Test

export BBOX_LLM_API_KEY="your-api-key"
./scripts/test_llm.sh "Test message"
