# Signature Task 🖋️

Bu repo, imza/noktalama tespiti için oluşturduğum kodları ve veri setini içeriyor. Temel amacım, resim formatındaki belgelerde yer alan dolu alanların gerçekten bir imza mı yoksa sadece nokta/çizgi gibi karalamalar mı olduğunu tespit etmekti.

## Özellikler
- **Doğruluk:** Karmaşıklık sınırlarını (complexity thresholds) detaylıca kalibre ederek **%92 doğruluk (accuracy)** oranına ulaştım.
- **Format Desteği:** PNG, JPG ve özellikle telefondan gelen HEIC formatları destekleniyor.
- **Clean & Secure Code:** Pipelinelar temiz, okunaklı ve hata toleranslı şekilde yazıldı.

## Kurulum

Repoyu bilgisayarınıza indirip sanal ortamda çalıştırabilirsiniz:

```bash
git clone https://github.com/duyguabbasoglu/signature_task.git
cd signature_task

python3 -m venv .venv
source .venv/bin/activate  # Windows için: .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Nasıl Çalıştırılır?

Bütün veri setini hızlıca test etmek ve sonuçları görmek isterseniz:
```bash
python full_dataset_test.py
```
*(Bu komut çalıştığında elde edilen tahminleri `vlm_full_results.csv` isimli bir dosyaya kaydeder.)*

Bir dosyayı doğrudan kod içinden kendiniz test etmek isterseniz örnek kullanım şu şekildedir:
```python
from classifier import load_image_robust, extract_features, classify_rule_based

img = load_image_robust("ornek_imza.png")
features = extract_features(img)
result, confidence, reasoning = classify_rule_based(features)

print(f"Sonuç: {result.value} (Güven Skoru: %{confidence*100:.1f})")
```

Eğer `make` aracı sizde yüklüyse, otomatik komutlarımı da kullanabilirsiniz:
- `make test` : Testleri çalıştırır.
- `make serve` : API sunucusunu (FastAPI) ayağa kaldırır.
- `make clean` : Gereksiz önbellek dosyalarını temizler.
