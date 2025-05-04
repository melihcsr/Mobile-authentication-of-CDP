# CDP QR Kod Test Kılavuzu

Bu belge, eğitilmiş CDP (Copy Detection Pattern) modelini kullanarak QR kod görüntülerini test etme sürecini açıklar.

## Test Araçları

Sistem üç farklı test aracı sunar:

1. **Standart Model Testi (`test_model.py`)**: Tüm veri kümesi üzerinde kapsamlı bir test yapar
2. **Tek Görsel Testi (`test_single_image.py`)**: Tek bir QR kod görselini test eder ve sonuçları görselleştirir
3. **Toplu Görsel Testi (`batch_test_images.py`)**: Bir klasördeki tüm QR kod görsellerini test eder

## Kurulum Gereksinimleri

Testleri çalıştırmadan önce şunların hazır olduğundan emin olun:

- Eğitilmiş model checkpoint'leri (eğitim tamamlandığında oluşturulur)
- Test edilecek QR kod görselleri
- Python bağımlılıkları (numpy, matplotlib, scikit-image, tensorflow vb.)

## Tek Bir QR Kod Görselini Test Etme

Bir QR kod görselini test etmek için:

```bash
python test_single_image.py --image_path /path/to/your/qrcode.png
```

### Parametreler:

- `--image_path`: Test edilecek QR kod görselinin yolu (**zorunlu**)
- `--checkpoint_path`: Belirli bir checkpoint kullanmak için (belirtilmezse en son checkpoint kullanılır)
- `--image_type`: Görüntü tipi (`rgb` veya `gray`)
- `--show_image`: Sonuçları görsel olarak göster (varsayılan: `True`)
- `--save_result`: Sonuçları görsel olarak kaydet (varsayılan: `True`)

### Örnek:

```bash
python test_single_image.py --image_path qr_samples/test1.png --image_type rgb
```

## Bir Klasördeki Tüm QR Kod Görsellerini Test Etme

Bir klasördeki tüm QR kod görsellerini test etmek için:

```bash
python batch_test_images.py --images_dir /path/to/qrcode/folder
```

### Parametreler:

- `--images_dir`: Test edilecek QR kod görsellerinin bulunduğu klasör (**zorunlu**)
- `--checkpoint_path`: Belirli bir checkpoint kullanmak için (belirtilmezse en son checkpoint kullanılır)
- `--image_type`: Görüntü tipi (`rgb` veya `gray`)
- `--save_results`: Sonuçları JSON dosyasına kaydet (varsayılan: `True`)
- `--results_file`: Sonuçların kaydedileceği dosya adı (varsayılan: `batch_test_results.json`)
- `--save_visualizations`: Her görüntü için görselleştirme kaydet (varsayılan: `True`)

### Örnek:

```bash
python batch_test_images.py --images_dir qr_samples/ --image_type rgb
```

## Standart Model Testi

Eğitilmiş modeli tüm test veri kümesi üzerinde değerlendirmek için:

```bash
python test_model.py --test_size 100
```

### Parametreler:

- `--test_size`: Her sınıftan test edilecek görüntü sayısı (varsayılan: `100`)
- `--checkpoint_path`: Belirli bir checkpoint kullanmak için (belirtilmezse en son checkpoint kullanılır)
- `--image_type`: Görüntü tipi (`rgb` veya `gray`)
- `--results_file`: Sonuçların kaydedileceği dosya adı (varsayılan: `test_results.json`)

### Örnek:

```bash
python test_model.py --test_size 50 --image_type rgb
```

## QR Kod Test Sonuçlarını Anlama

Test sonuçları şu bilgileri içerir:

### Tek Görsel Testi İçin:

- Sınıflandırma sonucu (Orijinal veya Sahte)
- Güven skoru (yüzde olarak)
- Her sınıf için olasılık değerleri
- Görselleştirme (orijinal görüntü ve sınıf olasılıkları)

### Toplu Test İçin:

- Her görüntü için tahmin edilen sınıf ve güven skoru
- Test edilen tüm görsellerin özet istatistikleri
- Sınıf dağılımı (her sınıfa ait kaç görüntü olduğu)
- Görselleştirmeler ve JSON formatında detaylı sonuçlar

## Hata Durumlarında Yapılacaklar

Eğer test sırasında hatalarla karşılaşırsanız:

1. Doğru checkpoint dosyasını kullandığınızdan emin olun
2. QR kod görsellerinin desteklenen formatta olduğunu kontrol edin (PNG, JPG, JPEG)
3. Görüntü boyutunun model için uygun olduğunu kontrol edin
4. Gerektiğinde model sınıf sayısını (`--n_classes`) doğru ayarlayın

## Görselleştirmeleri Anlama

Görselleştirme sonuçları iki bölümden oluşur:

1. **Sol taraf**: Test edilen orijinal QR kod görseli
2. **Sağ taraf**: Tahmin sonuçları çubuk grafik olarak gösterilir
   - Yeşil çubuk: "Orijinal" sınıfı için olasılık
   - Kırmızı çubuk: "Sahte" sınıfı için olasılık
   - En yüksek olasılıklı sınıf ve güven skoru başlıkta gösterilir 