'''
CDP Toplu Görsel Test Scripti - Çoklu QR kod görselleri için
'''

from __future__ import print_function

import argparse
import yaml
import sys
import os
import time
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import skimage.io
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.ClassifierDataLoader import ClassifierDataLoader
from libs.ClassificationModel import ClassificationModel

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TEST: CDP Toplu Görsel Test Scripti")
parser.add_argument("--config_path", default="./supervised_classification/configuration.yml", type=str, help="The config file path")
parser.add_argument("--images_dir", default=None, type=str, help="Test edilecek görsellerin klasör yolu")
parser.add_argument("--image_type", default="rgb", type=str, choices=['rgb', 'gray'], help="The image type")
parser.add_argument("--templates_path", default="./data/binary_templates/", type=str, help="Binary templates path")
parser.add_argument("--n_classes", default=2, type=int, choices=[2, 5], help="The number classes: 2 - original or fakes, 5 - original and 4 types of fakes")
parser.add_argument("--is_max_pool", default=True, type=int, help="Is to use max pooling in the trained model?")
parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the checkpoint to test")
parser.add_argument("--is_debug", default=True, type=int, help="Is debug mode?")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate used for training")
parser.add_argument("--save_results", default=True, type=bool, help="Save the results to a JSON file")
parser.add_argument("--results_file", default="batch_test_results.json", type=str, help="File to save test results")
parser.add_argument("--save_visualizations", default=True, type=bool, help="Save visualization images for each test")
parser.add_argument("--data_paths", default=["./data/original/rgb/",
                                           "./data/fakes_1/paper_white/rgb/",
                                           "./data/fakes_1/paper_gray/rgb/",
                                           "./data/fakes_2/paper_white/rgb/",
                                           "./data/fakes_2/paper_gray/rgb/"], type=str, help="The data paths")

args = parser.parse_args()
# ======================================================================================================================
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())
log.info("CDP TOPLU GÖRSEL TEST BAŞLIYOR")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ======================================================================================================================

def find_latest_checkpoint():
    """
    En son checkpoint'i bulur
    """
    checkpoint_dir = f"checkpoints/{args.image_type}_supervised_classifier_n_{args.n_classes}_one_by_one"
    
    if not os.path.exists(checkpoint_dir):
        log.error(f"Checkpoint klasörü bulunamadı: {checkpoint_dir}")
        return None
    
    files = os.listdir(checkpoint_dir)
    checkpoint_files = [f for f in files if f.endswith('.weights.h5') and 'final' in f]
    
    if not checkpoint_files:
        checkpoint_files = [f for f in files if f.endswith('.weights.h5')]
    
    if not checkpoint_files:
        log.error(f"Checkpoint dosyası bulunamadı: {checkpoint_dir}")
        return None
    
    # En yeni checkpoint'i bul
    checkpoint_files.sort(reverse=True)
    return os.path.join(checkpoint_dir, checkpoint_files[0])

def load_and_normalize_image(image_path, config, args):
    """
    Görüntüyü yükleyip normalleştirir
    """
    from libs.ClassifierDataLoader import ClassifierDataLoader
    
    # BaseClass'tan işlevleri almak için boş bir loader nesnesi oluşturalım
    dummy_loader = ClassifierDataLoader(config, args, type="test", is_debug_mode=args.is_debug)
    
    try:
        image_x = skimage.io.imread(image_path).astype(np.float64)
        original_image = skimage.io.imread(image_path)
        
        # Görüntünün RGB olup olmadığını kontrol et
        if len(image_x.shape) == 2:  # Gri tonlamalı görüntü (sadece yükseklik ve genişlik)
            log.info(f"Gri tonlamalı görüntü RGB formatına dönüştürülüyor: {image_path}")
            # Gri tonlamalı görüntüyü RGB'ye dönüştür (3 kanallı)
            image_x = np.stack((image_x,) * 3, axis=-1)
        elif len(image_x.shape) == 3 and image_x.shape[2] == 4:  # RGBA görüntü
            log.info(f"RGBA görüntü RGB formatına dönüştürülüyor: {image_path}")
            # Alpha kanalını kaldır
            image_x = image_x[:, :, :3]
        
        # Görüntüyü ön işleme
        image_x = dummy_loader._ClassifierDataLoader__centralCrop(image_x, targen_size=config.models['classifier']["target_size"])
        image_x = dummy_loader.normaliseDynamicRange(image_x, config.dataset['args'])
        return image_x, original_image
    except Exception as e:
        log.error(f"Görüntü yükleme hatası: {e}")
        return None, None

def save_visualization(image, prediction, class_names, file_name, save_dir):
    """
    Tahmin sonuçlarını görselleştirir ve kaydeder
    """
    plt.figure(figsize=(10, 6))
    
    # Orijinal görüntüyü göster
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Test Görseli")
    plt.axis('off')
    
    # Tahmin sonuçlarını göster
    plt.subplot(1, 2, 2)
    max_idx = np.argmax(prediction)
    confidence = prediction[max_idx] * 100
    
    colors = ['green' if max_idx == 0 else 'red'] + ['blue'] * (len(class_names)-1)
    plt.bar(class_names, prediction, color=colors)
    plt.ylim(0, 1)
    plt.title(f"Tahmin: {class_names[max_idx]} (%.2f%%)" % confidence)
    
    plt.suptitle("CDP Görsel Tahmin Sonucu", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{file_name}_prediction.png")
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def batch_test_images():
    """
    Bir klasördeki tüm görüntüleri test eder
    """
    # Görüntü klasörünü kontrol et
    if not args.images_dir:
        log.error("Görüntü klasörü belirtilmedi! --images_dir parametresini kullanın.")
        return
    
    if not os.path.exists(args.images_dir) or not os.path.isdir(args.images_dir):
        log.error(f"Belirtilen klasör bulunamadı veya geçerli bir klasör değil: {args.images_dir}")
        return
    
    # Sonuçların kaydedileceği klasörü oluştur
    results_dir = "results/batch_tests"
    os.makedirs(results_dir, exist_ok=True)
    
    # Görüntü dosyalarını bul
    image_files = []
    valid_extensions = ['.png', '.jpg', '.jpeg']
    
    for ext in valid_extensions:
        image_files.extend(list(Path(args.images_dir).glob(f"*{ext}")))
    
    if not image_files:
        log.error(f"Klasörde hiç görüntü dosyası bulunamadı: {args.images_dir}")
        return
    
    log.info(f"Toplam {len(image_files)} görüntü dosyası bulundu.")
    
    # Konfigürasyon ve model yükleme
    log.info(f"Konfigürasyon dosyası: {args.config_path}")
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))

    args.checkpoint_dir = "%s_supervised_classifier_n_%d_one_by_one" % (args.image_type, args.n_classes)
    args.dir = "supervised_classifier_lr%s_one_by_one" % args.lr
    log.info(f"Checkpoint klasörü: {args.checkpoint_dir}")
    
    # Model yükleme
    log.info("Model hazırlanıyor...")
    model = ClassificationModel(config, args)
    Classifier = model.ClassifierModel
    
    # Checkpoint yükleme
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = find_latest_checkpoint()
        
    if not checkpoint_path:
        log.error("Checkpoint dosyası bulunamadı. Test yapılamıyor.")
        return
    
    log.info(f"Checkpoint yükleniyor: {checkpoint_path}")
    Classifier.load_weights(checkpoint_path)
    
    # Sınıf adlarını belirle
    if args.n_classes == 2:
        class_names = ["Orijinal", "Sahte"]
    else:
        class_names = ["Orijinal"] + [f"Sahte {i+1}" for i in range(args.n_classes-1)]
    
    # Her bir görüntü için test yap
    results = []
    start_time = time.time()
    
    for i, image_path in enumerate(image_files):
        image_path_str = str(image_path)
        file_name = os.path.basename(image_path_str).split('.')[0]
        
        log.info(f"[{i+1}/{len(image_files)}] Test ediliyor: {image_path_str}")
        
        # Görüntüyü yükle
        normalized_image, original_image = load_and_normalize_image(image_path_str, config, args)
        
        if normalized_image is None:
            log.warning(f"Görüntü yüklenemedi, atlanıyor: {image_path_str}")
            continue
        
        # Tahmin yap
        image_batch = np.expand_dims(normalized_image, axis=0)
        prediction = Classifier.predict(image_batch)[0]
        
        # Sonuçları analiz et
        max_idx = np.argmax(prediction)
        confidence = prediction[max_idx] * 100
        predicted_class = class_names[max_idx]
        
        log.info(f"  - Tahmin: {predicted_class} (%.2f%%)" % confidence)
        
        # Görselleştirme
        visualization_path = None
        if args.save_visualizations:
            visualization_path = save_visualization(original_image, prediction, class_names, file_name, results_dir)
            log.info(f"  - Görselleştirme kaydedildi: {visualization_path}")
        
        # Sonuçları kaydet
        result = {
            "file_name": file_name,
            "file_path": image_path_str,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "probabilities": {name: float(prediction[i] * 100) for i, name in enumerate(class_names)},
            "visualization_path": visualization_path
        }
        
        results.append(result)
    
    # İstatistikleri hesapla
    total_time = time.time() - start_time
    
    class_counts = {}
    for result in results:
        predicted_class = result["predicted_class"]
        if predicted_class in class_counts:
            class_counts[predicted_class] += 1
        else:
            class_counts[predicted_class] = 1
    
    # Sonuçları JSON olarak kaydet
    if args.save_results:
        summary = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "model_checkpoint": checkpoint_path,
            "total_images": len(results),
            "total_time_seconds": total_time,
            "time_per_image": total_time / max(1, len(results)),
            "class_distribution": class_counts,
            "results": results
        }
        
        with open(os.path.join(results_dir, args.results_file), 'w') as f:
            json.dump(summary, f, indent=4)
        
        log.info(f"Sonuçlar kaydedildi: {os.path.join(results_dir, args.results_file)}")
    
    # Özet rapor göster
    log.info("=" * 50)
    log.info("TEST SONUÇLARI ÖZETİ:")
    log.info(f"Toplam görüntü: {len(results)}")
    log.info(f"Toplam süre: {total_time:.2f} saniye")
    log.info(f"Görüntü başına ortalama süre: {total_time / max(1, len(results)):.4f} saniye")
    log.info("Sınıf dağılımı:")
    for class_name, count in class_counts.items():
        log.info(f"  - {class_name}: {count} görüntü (%.1f%%)" % (count / len(results) * 100))
    log.info("=" * 50)

# ======================================================================================================================
if __name__ == "__main__":
    batch_test_images() 