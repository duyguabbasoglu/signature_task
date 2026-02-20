Bu repo, imza/noktalama tespiti için oluşturduğum kodları ve veri setini içeriyor. Temel amacım, resim formatındaki belgelerde imza için ayrılan bound boxların tespiti ve malum alanların gerçekten bir imza mı yoksa sadece nokta/çizgi gibi karalamalar mı olduğunu tespit etmektedir, doğruluk oranı %97.5'tir.

#kurulum
bash
git clone https://github.com/duyguabbasoglu/signature_task.git
cd signature_task
python3 -m venv .venv
source .venv/bin/activate
#.\.venv\Scripts\Activate.ps1 # windows için
pip install -r requirements.txt

#tüm veri seti üzerinden test
python full_dataset_test.py # vlm_full_test_results.csv

#tekli test
python
from classifier import load_image_robust, extract_features, classify_rule_based
img = load_image_robust("test_images/ornek_imza.png") # test edilecek görselin yolu
features = extract_features(img)
result, confidence, reasoning = classify_rule_based(features)
print(f"Sınıflandırma: {result.value} (Güven Skoru: %{confidence*100:.1f})")
print(f"Gerekçe: {reasoning}")

#Alternatif: Makefile ile Hızlı Kullanım Eğer sisteminizde make kuruluysa, test komutları için hazırladığım kısayolları da kullanabilirsiniz
Testleri çalıştırmak için: make test
API sunucusunu ayağa kaldırmak için: make serve
Önbellek/gereksiz dosyaları temizlemek için: make clean

