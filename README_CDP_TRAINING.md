# CDP Eğitim Sistemi

Bu belge, CDP (Copy Detection Pattern) görüntü sınıflandırma modelinin eğitilmesi için oluşturulan sistem hakkında bilgi verir.

## Genel Bakış

Bu sistem, CDP görüntülerinin orijinal mi yoksa sahte mi olduğunu sınıflandırmak için bir makine öğrenimi modeli eğitmektedir. Düşük GPU kapasiteli sistemlerde bile çalışabilmesi için görüntüleri tek tek işleyerek eğitim yapar.

## Dosyalar

- `train_one_by_one.py`: Görüntüleri tek tek işleyerek GPU yükünü minimize eden eğitim scripti
- `monitor_training.py`: Eğitim ilerlemesini gerçek zamanlı olarak izlemeye yarayan araç
- `test_model.py`: Eğitilen modeli test etmeye yarayan script
- `training_one_by_one_status.json`: Eğitim durumunu kaydeden JSON dosyası

## Eğitim Sistemi Özellikleri

- **Düşük GPU Kullanımı**: Görüntüler tek tek işlenerek GPU belleği minimum seviyede tutulur
- **Kaldığı Yerden Devam Edebilme**: Eğitim herhangi bir noktada durdurulup tekrar başlatılabilir
- **Ayrıntılı İlerleme İzleme**: Gerçek zamanlı ilerleme izleme ve tahmini tamamlanma süresi
- **Otomatik Checkpoint Kaydetme**: Düzenli aralıklarla model ağırlıkları kaydedilir
- **Güvenli Bellek Yönetimi**: Her görüntü işleminden sonra bellek temizlenir
- **Tüm Veri Yollarını Kullanma**: Sistem tüm veri klasörlerinden görüntüleri okur ve eğitimde kullanır

## Kullanım

### Eğitimi Başlatma

```bash
python train_one_by_one.py --epochs 5 --n_classes 2
```

### Eğitim İlerlemesini İzleme

```bash
python monitor_training.py
```

### Model Testi

```bash
python test_model.py --test_size 100
```

## Parametre Açıklamaları

### train_one_by_one.py

- `--epochs`: Eğitilecek epoch sayısı
- `--n_classes`: Sınıf sayısı (2: orijinal veya sahte, 5: orijinal ve 4 çeşit sahte)
- `--data_paths`: Veri yolları
- `--image_type`: Görüntü tipi (rgb veya gray)
- `--no_augmentation`: Veri artırmayı devre dışı bırakma (1: kapalı, 0: açık)
- `--save_interval`: Kaydetme aralığı (kaç görüntüde bir checkpoint kaydedileceği)

### monitor_training.py

- `--status_file`: İzlenecek durum dosyası
- `--refresh_interval`: Yenileme aralığı (saniye)

### test_model.py

- `--checkpoint_path`: Test edilecek checkpoint dosyası (belirtilmezse en son oluşturulan kullanılır)
- `--test_size`: Her sınıftan test edilecek görüntü sayısı
- `--results_file`: Test sonuçlarının kaydedileceği dosya

## Eğitim İstatistikleri

Eğitim istatistikleri `training_one_by_one_status.json` dosyasında tutulur ve şu bilgileri içerir:

- Geçerli epoch ve toplam epoch sayısı
- İşlenen görüntü sayısı ve toplam görüntü sayısı
- Eğitim ilerlemesi (yüzde olarak)
- Geçen süre ve tahmini kalan süre
- Son kaydedilen checkpoint konumu
- Son loss değeri

## Test Sonuçları

Test sonuçları `test_results.json` dosyasında tutulur ve şu bilgileri içerir:

- Doğruluk (Accuracy), Kesinlik (Precision), Duyarlılık (Recall) ve F1 skoru
- Karmaşıklık matrisi (Confusion matrix)
- Yanlış sınıflandırılan görüntülerin listesi 