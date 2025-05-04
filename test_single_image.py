'''
CDP Tek Görsel Test Scripti - QR kod görselleri için
'''

from __future__ import print_function

import argparse
import yaml
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.ClassifierDataLoader import ClassifierDataLoader
from libs.ClassificationModel import ClassificationModel

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TEST: CDP Tek Görsel Test Scripti")
parser.add_argument("--config_path", default="./supervised_classification/configuration.yml", type=str, help="The config file path")
parser.add_argument("--image_path", default=None, type=str, help="Test edilecek görsel yolu")
parser.add_argument("--image_type", default="rgb", type=str, choices=['rgb', 'gray'], help="The image type")
parser.add_argument("--templates_path", default="./data/binary_templates/", type=str, help="Binary templates path")
parser.add_argument("--n_classes", default=2, type=int, choices=[2, 5], help="The number classes: 2 - original or fakes, 5 - original and 4 types of fakes")
parser.add_argument("--is_max_pool", default=True, type=int, help="Is to use max pooling in the trained model?")
parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the checkpoint to test")
parser.add_argument("--is_debug", default=True, type=int, help="Is debug mode?")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate used for training")
parser.add_argument("--show_image", default=True, type=bool, help="Display the image with prediction")
parser.add_argument("--save_result", default=True, type=bool, help="Save the result as an image")
parser.add_argument("--data_paths", default=["./data/original/rgb/",
                                           "./data/fakes_1/paper_white/rgb/",
                                           "./data/fakes_1/paper_gray/rgb/",
                                           "./data/fakes_2/paper_white/rgb/",
                                           "./data/fakes_2/paper_gray/rgb/"], type=str, help="The data paths")

args = parser.parse_args()
# ======================================================================================================================
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())
log.info("CDP TEK GÖRSEL TEST BAŞLIYOR")

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
        log.info(f"Görüntü yüklendi, boyut: {image_x.shape}")
        
        # Görüntünün RGB olup olmadığını kontrol et
        if len(image_x.shape) == 2:  # Gri tonlamalı görüntü (sadece yükseklik ve genişlik)
            log.info("Gri tonlamalı görüntü RGB formatına dönüştürülüyor...")
            # Gri tonlamalı görüntüyü RGB'ye dönüştür (3 kanallı)
            image_x = np.stack((image_x,) * 3, axis=-1)
            log.info(f"Dönüştürme sonrası boyut: {image_x.shape}")
        elif len(image_x.shape) == 3 and image_x.shape[2] == 4:  # RGBA görüntü
            log.info("RGBA görüntü RGB formatına dönüştürülüyor...")
            # Alpha kanalını kaldır
            image_x = image_x[:, :, :3]
            log.info(f"Dönüştürme sonrası boyut: {image_x.shape}")
        
        # Görüntüyü ön işleme
        image_x = dummy_loader._ClassifierDataLoader__centralCrop(image_x, targen_size=config.models['classifier']["target_size"])
        image_x = dummy_loader.normaliseDynamicRange(image_x, config.dataset['args'])
        return image_x
    except Exception as e:
        log.error(f"Görüntü yükleme hatası: {e}")
        return None

def display_result(image, prediction, class_names, save_path=None):
    """
    Tahmin sonuçlarını görselleştirir
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
    
    if save_path:
        plt.savefig(save_path)
        log.info(f"Sonuç görüntüsü kaydedildi: {save_path}")
    
    if args.show_image:
        plt.show()

def test_single_image():
    """
    Tek bir görüntüyü test eder
    """
    # Görüntü yolunu kontrol et
    if not args.image_path:
        log.error("Görüntü yolu belirtilmedi! --image_path parametresini kullanın.")
        return
    
    if not os.path.exists(args.image_path):
        log.error(f"Belirtilen görüntü bulunamadı: {args.image_path}")
        return
    
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
    
    # Görüntüyü yükle
    log.info(f"Test edilecek görüntü: {args.image_path}")
    image = load_and_normalize_image(args.image_path, config, args)
    
    if image is None:
        return
    
    # Tahmin yap
    log.info("Tahmin yapılıyor...")
    # Modelin beklediği formata getir (batch boyutu ekle)
    image_batch = np.expand_dims(image, axis=0)
    prediction = Classifier.predict(image_batch)[0]
    
    # Sınıf adlarını belirle
    if args.n_classes == 2:
        class_names = ["Orijinal", "Sahte"]
    else:
        class_names = ["Orijinal"] + [f"Sahte {i+1}" for i in range(args.n_classes-1)]
    
    # Sonuçları göster
    max_idx = np.argmax(prediction)
    confidence = prediction[max_idx] * 100
    
    log.info("=" * 50)
    log.info("TAHMIN SONUCU:")
    log.info(f"Sınıf: {class_names[max_idx]}")
    log.info(f"Güven: %.2f%%" % confidence)
    log.info("-" * 50)
    
    for i, name in enumerate(class_names):
        log.info(f"{name}: %.2f%%" % (prediction[i] * 100))
    
    # Sonuçları görselleştir
    original_image = skimage.io.imread(args.image_path)
    
    if args.show_image or args.save_result:
        # Kaydedilecek dosya yolunu belirle
        save_path = None
        if args.save_result:
            save_dir = "results/single_tests"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{os.path.basename(args.image_path).split('.')[0]}_prediction.png"
        
        display_result(original_image, prediction, class_names, save_path)
    
    return class_names[max_idx], confidence

# ======================================================================================================================
if __name__ == "__main__":
    test_single_image() 