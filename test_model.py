'''
CDP Model Test Scripti
'''

from __future__ import print_function

import argparse
import yaml
import sys
import os
import time
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.ClassifierDataLoader import ClassifierDataLoader
from libs.ClassificationModel import ClassificationModel

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TEST: CDP Model Test Scripti")
parser.add_argument("--config_path", default="./supervised_classification/configuration.yml", type=str, help="The config file path")
# datset parameters
parser.add_argument("--image_type", default="rgb", type=str, choices=['rgb', 'gray'], help="The image type")
parser.add_argument("--data_paths", default=["./data/original/rgb/",
                                           "./data/fakes_1/paper_white/rgb/",
                                           "./data/fakes_1/paper_gray/rgb/",
                                           "./data/fakes_2/paper_white/rgb/",
                                           "./data/fakes_2/paper_gray/rgb/"], type=str, help="The data paths")
parser.add_argument("--templates_path", default="./data/binary_templates/", type=str, help="Binary templates path")
# model parameters
parser.add_argument("--n_classes", default=2, type=int, choices=[2, 5], help="The number classes: 2 - original or fakes, 5 - original and 4 types of fakes")
parser.add_argument("--is_max_pool", default=True, type=int, help="Is to use max pooling in the trained model?")
parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the checkpoint to test")
parser.add_argument("--results_file", default="test_results.json", type=str, help="File to save test results")
# log mode
parser.add_argument("--is_debug", default=True, type=int, help="Is debug mode?")
parser.add_argument("--test_size", default=100, type=int, help="Number of images to test from each class")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate used for training")

args = parser.parse_args()
# ======================================================================================================================
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())
log.info("CDP MODEL TEST BAŞLIYOR")

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

def load_and_normalize_images(data_path, config, args, max_count):
    """
    Belirtilen klasörden görüntüleri yükler ve normalleştirir
    """
    import skimage.io
    from libs.ClassifierDataLoader import ClassifierDataLoader
    
    # BaseClass'tan işlevleri almak için boş bir loader nesnesi oluşturalım
    dummy_loader = ClassifierDataLoader(config, args, type="test", is_debug_mode=args.is_debug)
    
    images = []
    file_paths = []
    
    try:
        files = os.listdir(data_path)
        files.sort()
        
        count = 0
        for file_name in files:
            if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                try:
                    image_path = os.path.join(data_path, file_name)
                    image_x = skimage.io.imread(image_path).astype(np.float64)
                    image_x = dummy_loader._ClassifierDataLoader__centralCrop(image_x, targen_size=config.models['classifier']["target_size"])
                    image_x = dummy_loader.normaliseDynamicRange(image_x, config.dataset['args'])
                    
                    images.append(image_x)
                    file_paths.append(image_path)
                    
                    count += 1
                    if count >= max_count:
                        break
                except Exception as e:
                    log.error(f"Görüntü yükleme hatası: {image_path}, hata: {e}")
    except Exception as e:
        log.error(f"Veri yolu okunamadı: {data_path}, hata: {e}")
    
    return np.array(images), file_paths

def test_model():
    log.info("test_model fonksiyonu başladı")
    
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
    
    # Test verilerini hazırlama
    all_test_images = []
    all_test_labels = []
    all_file_paths = []
    
    log.info(f"Her sınıftan {args.test_size} görüntü test edilecek")
    
    for j, data_path in enumerate(args.data_paths):
        log.info(f"Test verileri yükleniyor: {data_path}")
        
        test_images, file_paths = load_and_normalize_images(data_path, config, args, args.test_size)
        
        if len(test_images) > 0:
            log.info(f"{data_path}: {len(test_images)} görüntü yüklendi")
            
            # Etiketleri hazırla
            if args.n_classes == 1:
                test_labels = np.array([0 if j == 0 else 1] * len(test_images))
            elif args.n_classes != len(args.data_paths):
                test_labels = []
                for _ in range(len(test_images)):
                    onehot_label = [0] * args.n_classes
                    if j == 0:
                        onehot_label[0] = 1
                    else:
                        onehot_label[1] = 1
                    test_labels.append(onehot_label)
                test_labels = np.array(test_labels)
            else:
                test_labels = []
                for _ in range(len(test_images)):
                    onehot_label = [0] * len(args.data_paths)
                    onehot_label[j] = 1
                    test_labels.append(onehot_label)
                test_labels = np.array(test_labels)
            
            all_test_images.append(test_images)
            all_test_labels.append(test_labels)
            all_file_paths.extend(file_paths)
        else:
            log.warning(f"{data_path}: Görüntü yüklenemedi")
    
    if not all_test_images:
        log.error("Test için görüntü bulunamadı.")
        return
    
    # Verileri birleştir
    X_test = np.concatenate(all_test_images, axis=0)
    y_test = np.concatenate(all_test_labels, axis=0)
    
    log.info(f"Toplam test seti: {X_test.shape[0]} görüntü")
    
    # Model ile tahmin yap
    log.info("Tahminler yapılıyor...")
    
    batch_size = 10
    num_batches = (X_test.shape[0] + batch_size - 1) // batch_size
    
    all_predictions = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_test.shape[0])
        
        batch_images = X_test[start_idx:end_idx]
        y_pred_batch = Classifier.predict(batch_images)
        
        all_predictions.append(y_pred_batch)
        
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            log.info(f"Tahmin ilerleme: {i+1}/{num_batches} batch işlendi")
    
    y_pred = np.concatenate(all_predictions, axis=0)
    
    # Sonuçları değerlendirme
    log.info("Sonuçlar değerlendiriliyor...")
    
    # İkili sınıflandırma için
    if args.n_classes == 2:
        y_test_class = np.argmax(y_test, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_test_class, y_pred_class)
        precision = precision_score(y_test_class, y_pred_class, average='weighted')
        recall = recall_score(y_test_class, y_pred_class, average='weighted')
        f1 = f1_score(y_test_class, y_pred_class, average='weighted')
        
        conf_matrix = confusion_matrix(y_test_class, y_pred_class)
        
        log.info(f"Test sonuçları:")
        log.info(f"Doğruluk (Accuracy): {accuracy:.4f}")
        log.info(f"Kesinlik (Precision): {precision:.4f}")
        log.info(f"Duyarlılık (Recall): {recall:.4f}")
        log.info(f"F1 Skoru: {f1:.4f}")
        log.info(f"Karmaşıklık Matrisi:")
        log.info(f"{conf_matrix}")
        
        # Hatalı sınıflandırılan görüntüleri bul
        misclassified_indices = np.where(y_test_class != y_pred_class)[0]
        log.info(f"Hatalı sınıflandırılan görüntü sayısı: {len(misclassified_indices)}")
        
        # Detaylı sonuçları kaydet
        results = {
            "checkpoint_path": checkpoint_path,
            "test_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            },
            "confusion_matrix": conf_matrix.tolist(),
            "class_names": ["Orijinal", "Sahte"],
            "misclassified_count": int(len(misclassified_indices)),
            "misclassified_images": []
        }
        
        # Hatalı sınıflandırılan görüntüleri kaydet
        for idx in misclassified_indices:
            results["misclassified_images"].append({
                "file_path": all_file_paths[idx],
                "true_class": int(y_test_class[idx]),
                "predicted_class": int(y_pred_class[idx]),
                "confidence": float(y_pred[idx][y_pred_class[idx]])
            })
        
        # Sonuçları dosyaya kaydet
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        log.info(f"Test sonuçları kaydedildi: {args.results_file}")
    
    log.info("Test tamamlandı!")

# ======================================================================================================================
if __name__ == "__main__":
    test_model() 