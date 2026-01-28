# Bounding Box Content Detection

Bu repository, bir bounding box içindeki içeriğin ne olduğunu tespit eden
bir pipeline içerir.

## Algoritma

```
┌─────────────────────────────────────────────────────────────┐
│                    Bounding Box Görüntüsü                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │    Aşama 1: Boş mu Dolu mu?   │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
        ┌───────────┐                  ┌─────────────┐
        │   EMPTY   │                  │   FILLED    │
        │  (Boş)    │                  │   (Dolu)    │
        └───────────┘                  └──────┬──────┘
                                              │
                                              ▼
                               ┌──────────────────────────┐
                               │ Aşama 2: Noktalama mı?   │
                               │    (Basit CNN / Rules)   │
                               └────────────┬─────────────┘
                                            │
                              ┌─────────────┴─────────────┐
                              │                           │
                              ▼                           ▼
                    ┌─────────────────┐         ┌─────────────────┐
                    │  FILLED_PUNCT   │         │  FILLED_OTHER   │
                    │ (Noktalama var) │         │ (Başka içerik)  │
                    └─────────────────┘         └─────────────────┘
```

## Tespit Sonuçları

| Sonuç | Açıklama |
|-------|----------|
| `EMPTY` | Bounding box boş |
| `FILLED_PUNCT` | Noktalama işareti mevcut (virgül, nokta, vs.) |
| `FILLED_OTHER` | Başka içerik mevcut (imza, yazı, vs.) |

## Kullanım

### Tek Dosya Analizi

```bash
python detector.py path/to/image.png
```

### Klasör Analizi

```bash
python detector.py
# data/ klasöründeki tüm görüntüleri analiz eder
```

### Python API

```python
from detector import analyze, analyze_bytes, DetectionResult

# Dosyadan analiz
result = analyze("image.png")
print(result.result)           # DetectionResult.FILLED_PUNCT
print(result.is_empty)         # False
print(result.is_punctuation)   # True
print(result.confidence)       # 0.90

# Byte array'den analiz
with open("image.png", "rb") as f:
    result = analyze_bytes(f.read())
```

## Teknik Detaylar

### Aşama 1: Boş/Dolu Kontrolü (Rule-based)

1. Grayscale dönüşüm
2. Otsu binarization
3. Connected component analizi
4. Toplam ink alanı hesaplama
5. Threshold karşılaştırma (50 px² altı = boş)

### Aşama 2: Noktalama Tespiti

Şu anda rule-based yaklaşım:
- Maksimum alan: 500 px²
- Maksimum bileşen sayısı: 3

İleride basit CNN modeli ile değiştirilecek.

## Yapı

```
signature_task/
├── detector.py          # Ana tespit modülü
├── models/
│   ├── __init__.py
│   └── punct_cnn.py     # Basit CNN modeli (placeholder)
├── data/                # Test görüntüleri
└── README.md
```

## Gereksinimler

```bash
pip install opencv-python numpy
```

## OCR Entegrasyonu

Bu modül OCR pipeline'ı ile entegre çalışacak şekilde tasarlanmıştır.
Bounding box koordinatları upstream OCR sisteminden gelecektir.
