'''
Tek tek eğitim yapan CDP eğitim scripti - GPU yükünü minimum tutmak için
Author: Original Training script based on Olga TARAN's work, University of Geneva, 2021
'''

from __future__ import print_function

import argparse
import yaml
import sys
import os
import time
import json
import gc
import numpy as np
import tensorflow as tf
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.ClassifierDataLoader import ClassifierDataLoader
from libs.ClassificationModel import ClassificationModel

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TRAIN: Tek tek eğitim yapan CDP eğitim scripti")
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
parser.add_argument("--lr", default=1e-4, type=float, help="Training learning rate")
parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
parser.add_argument("--n_classes", default=2, type=int, choices=[2, 5], help="The number classes: 2 - original or fakes, 5 - original and 4 types of fakes")
parser.add_argument("--is_max_pool", default=True, type=int, help="Is to use max pooling in the trained model?")
parser.add_argument("--status_file", default="training_one_by_one_status.json", type=str, help="File to save training status")
# log mode
parser.add_argument("--is_debug", default=True, type=int, help="Is debug mode?")
# Augmentation
parser.add_argument("--no_augmentation", default=True, type=int, help="Turn off data augmentation")
# Checkpoint saving
parser.add_argument("--save_interval", default=50, type=int, help="How often to save checkpoint (in images)")
# Seed
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")

args = parser.parse_args()
# ======================================================================================================================
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())
log.info("TEK TEK EĞİTİM MODU: GPU yükünü minimize etmek için görüntüler tek tek işleniyor")

# GPU bellek kullanımını sınırla
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        log.info(f"GPU bellek büyümesi dinamik olarak ayarlandı.")
    except RuntimeError as e:
        log.error(f"GPU bellek ayarı yapılamadı: {e}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ======================================================================================================================

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    progress_str = f'\r{prefix} |{bar}| {percent}% {suffix}'
    log.info(progress_str)
    # Print new line on complete
    if iteration == total:
        log.info('')
        
def update_status_file(epoch, total_epochs, image_index, total_images, elapsed_time, loss, checkpoint_path):
    """
    Eğitim durumunu bir dosyaya kaydeder
    """
    overall_progress = ((epoch * total_images) + image_index) / (total_epochs * total_images) * 100
    
    status = {
        'current_epoch': epoch + 1,
        'total_epochs': total_epochs,
        'current_image': image_index + 1,
        'total_images': total_images,
        'epoch_progress_percent': (image_index / total_images) * 100,
        'overall_progress_percent': overall_progress,
        'elapsed_time_seconds': elapsed_time,
        'elapsed_time_hours': elapsed_time / 3600,
        'last_loss': float(loss),
        'last_checkpoint': checkpoint_path,
        'estimated_time_remaining_seconds': (elapsed_time / max(1, (epoch * total_images) + image_index)) * 
                                          ((total_epochs * total_images) - ((epoch * total_images) + image_index)),
        'status': 'in_progress' if epoch < total_epochs-1 or image_index < total_images-1 else 'completed',
        'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(args.status_file, 'w') as f:
        json.dump(status, f, indent=4)
    
    if (image_index + 1) % 10 == 0 or image_index == 0 or image_index == total_images - 1:
        log.info(f"Durum dosyası güncellendi: {args.status_file}")
    
    # Eğitim tamamlandığında son durumu ekrana basalım
    if epoch == total_epochs-1 and image_index == total_images-1:
        log.info(f"\n\n========== EĞİTİM TAMAMLANDI! ==========")
        log.info(f"Toplam süre: {elapsed_time:.2f} saniye ({elapsed_time/3600:.2f} saat)")
        log.info(f"Son checkpoint: {checkpoint_path}")
        log.info(f"==========================================\n\n")

def configure_no_augmentation():
    """
    Veri artırmayı devre dışı bırakır
    """
    config_path = args.config_path
    with open(config_path, 'r') as f:
        config_content = yaml.load(f, Loader=yaml.SafeLoader)
    
    # Konfigürasyonun bir kopyasını oluşturalım
    new_config_path = args.config_path.replace('.yml', '_no_aug.yml')
    config_content['dataset']['args']['augmentation'] = False
    
    with open(new_config_path, 'w') as f:
        yaml.dump(config_content, f)
    
    log.info(f"Veri artırma devre dışı bırakıldı. Yeni konfigürasyon dosyası: {new_config_path}")
    return new_config_path

def prepare_single_image_data(image_x, j, n_classes):
    """
    Tek bir görüntü için etiket hazırlar
    
    Args:
        image_x: İşlenecek görüntü
        j: Veri yolu indeksi (0=orijinal, 1+ sahte)
        n_classes: Sınıf sayısı
        
    Returns:
        data, label: Numpy dizileri
    """
    data = np.array([image_x])
    
    if n_classes == 1:
        if j == 0:
            labels = np.array([0])
        else:
            labels = np.array([1])
    elif n_classes != len(args.data_paths):
        onehot_label = [0 for _ in range(n_classes)]
        if j == 0:
            onehot_label[0] = 1
        else:
            onehot_label[1] = 1
        labels = np.array([onehot_label])
    else:
        onehot_label = [0 for _ in range(len(args.data_paths))]
        onehot_label[j] = 1
        labels = np.array([onehot_label])
    
    return data, labels

def load_and_normalize_image(image_path, config, args):
    """
    Görüntüyü yükleyip normalleştirir
    """
    import skimage.io
    from libs.ClassifierDataLoader import ClassifierDataLoader
    
    # BaseClass'tan işlevleri almak için boş bir loader nesnesi oluşturalım
    dummy_loader = ClassifierDataLoader(config, args, type="train", is_debug_mode=args.is_debug)
    
    try:
        image_x = skimage.io.imread(image_path).astype(np.float64)
        image_x = dummy_loader._ClassifierDataLoader__centralCrop(image_x, targen_size=config.models['classifier']["target_size"])
        image_x = dummy_loader.normaliseDynamicRange(image_x, config.dataset['args'])
        return image_x
    except Exception as e:
        log.error(f"Görüntü yükleme hatası: {e}")
        return None

def train_one_by_one():
    log.info("train_one_by_one fonksiyonu başladı")
    
    if args.no_augmentation:
        args.config_path = configure_no_augmentation()
    
    log.info(f"Konfigürasyon dosyası: {args.config_path}")
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))

    args.checkpoint_dir = "%s_supervised_classifier_n_%d_one_by_one" % (args.image_type, args.n_classes)
    args.dir = "supervised_classifier_lr%s_one_by_one" % args.lr
    log.info(f"Checkpoint klasörü: {args.checkpoint_dir}")
    log.info(f"Sonuç klasörü: {args.dir}")

    # Klasörleri oluşturalım
    checkpoint_dir = f"checkpoints/{args.checkpoint_dir}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(f"results/{args.dir}/prediction", exist_ok=True)
    os.makedirs(f"TensorBoard/{args.dir}", exist_ok=True)

    log.info("Start Model preparation.....")
    model = ClassificationModel(config, args)
    Classifier = model.ClassifierModel
    
    # Checkpoint yükleme
    last_epoch = 0
    last_image_index = 0
    if os.path.exists(args.status_file):
        try:
            with open(args.status_file, 'r') as f:
                status = json.load(f)
                last_epoch = status.get('current_epoch', 1) - 1
                last_image_index = status.get('current_image', 1) - 1
                
                if last_epoch >= 0 and last_image_index >= 0:
                    # En son kaydedilen checkpoint'i yükle
                    checkpoint_path = status.get('last_checkpoint')
                    if checkpoint_path and os.path.exists(checkpoint_path):
                        log.info(f"En son checkpoint yükleniyor: {checkpoint_path}")
                        Classifier.load_weights(checkpoint_path)
                        log.info(f"Eğitime devam ediliyor: Epoch {last_epoch+1}, Görüntü {last_image_index+1}")
                    else:
                        log.warning(f"Checkpoint bulunamadı: {checkpoint_path}, eğitime baştan başlanıyor.")
                        last_epoch = 0
                        last_image_index = 0
                        
                if status.get('status') == 'completed':
                    log.info("Eğitim zaten tamamlanmış! Tekrar başlatmak için durum dosyasını silin.")
                    return
        except Exception as e:
            log.error(f"Durum dosyası okunamadı: {e}. Eğitim baştan başlayacak.")
            last_epoch = 0
            last_image_index = 0

    # === model scheme visualisation ===============================================================================
    if args.is_debug:
        log.info("ClassificationModel:")
        model.Classifier.summary()

    # === Veri dosyalarını listeleme =================================================================================================
    all_image_files = []
    
    # Her veri yolu için dosya listesi oluşturalım
    for j, data_path in enumerate(args.data_paths):
        log.info(f"Veri yolu taranıyor: {data_path}")
        try:
            files = os.listdir(data_path)
            files.sort()
            
            # Her görüntü için path ve sınıf bilgisini kaydet
            for file_name in files:
                if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                    all_image_files.append({
                        'path': os.path.join(data_path, file_name),
                        'class_index': j,
                        'file_name': file_name
                    })
            
            log.info(f"Veri yolu {j+1}/{len(args.data_paths)}: {len(files)} dosya bulundu")
        except Exception as e:
            log.error(f"Veri yolu okunamadı: {data_path}, hata: {e}")
    
    total_images = len(all_image_files)
    log.info(f"Toplam {total_images} görüntü işlenecek")
    
    if total_images == 0:
        log.error("İşlenecek görüntü bulunamadı! Veri yollarını kontrol edin.")
        return

    # Rasgele karıştırma
    np.random.seed(args.seed)
    np.random.shuffle(all_image_files)
    
    # Devam etme durumunda, son kaldığımız yerden devam edelim
    if last_epoch > 0 or last_image_index > 0:
        start_epoch = last_epoch
        start_image_index = last_image_index
    else:
        start_epoch = 0
        start_image_index = 0
    
    # === Eğitim =================================================================================================
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        log.info(f"--- EPOCH {epoch+1}/{args.epochs} BAŞLIYOR ---")
        
        epoch_losses = []
        epoch_start_time = time.time()
        
        # Her bir görüntüyü tek tek işleyelim
        for i, image_info in enumerate(all_image_files):
            # Eğer devam ediyorsak ve bu görüntüyü zaten işlemişsek atla
            if epoch == start_epoch and i < start_image_index:
                continue
                
            image_start_time = time.time()
            
            try:
                # Görüntüyü yükle ve normalleştir
                image_path = image_info['path']
                class_index = image_info['class_index']
                file_name = image_info['file_name']
                
                # Yalnızca belirli aralıklarla log yazdıralım
                if i % 10 == 0 or i == 0 or i == total_images - 1:
                    log.info(f"Görüntü işleniyor {i+1}/{total_images}: {file_name} (sınıf: {class_index})")
                
                # Görüntüyü yükle
                image_x = load_and_normalize_image(image_path, config, args)
                
                if image_x is not None:
                    # Eğitim için veriyi hazırla
                    data, labels = prepare_single_image_data(image_x, class_index, args.n_classes)
                    
                    # Tek görüntü üzerinde eğitim yap
                    loss = Classifier.train_on_batch(data, labels)
                    epoch_losses.append(loss)
                    
                    # Her x görüntüde bir model ağırlıklarını kaydet
                    if (i + 1) % args.save_interval == 0 or i == total_images - 1:
                        save_path = f"{checkpoint_dir}/Classifier_epoch_{epoch+1}_image_{i+1}.weights.h5"
                        Classifier.save_weights(save_path)
                        log.info(f"Model kaydedildi: {save_path}")
                    else:
                        save_path = f"{checkpoint_dir}/Classifier_last.weights.h5"
                        
                    # İlerleme durumunu güncelle
                    elapsed_time = time.time() - start_time
                    update_status_file(epoch, args.epochs, i, total_images, elapsed_time, loss, save_path)
                    
                    # Her 10 görüntüde bir veya önemli noktalarda ilerleme bilgisini yazdır
                    if i % 10 == 0 or i == 0 or i == total_images - 1:
                        avg_loss = np.mean(np.asarray(epoch_losses[-min(10, len(epoch_losses)):]))
                        image_time = time.time() - image_start_time
                        
                        # Kalan süreyi hesapla
                        processed_images = (epoch * total_images) + (i + 1)
                        total_to_process = args.epochs * total_images
                        remaining_images = total_to_process - processed_images
                        avg_time_per_image = elapsed_time / processed_images
                        est_time_remaining = avg_time_per_image * remaining_images
                        
                        print_progress_bar(i+1, total_images,
                                         prefix=f'Epoch {epoch+1}/{args.epochs}',
                                         suffix=f'İşlendi: {i+1}/{total_images}, Loss: {loss:.6f}, Avg: {avg_loss:.6f}',
                                         length=30)
                        
                        log.info(f"Görüntü işleme süresi: {image_time:.2f}s")
                        log.info(f"Tahmini kalan süre: {est_time_remaining:.2f}s (~{est_time_remaining/3600:.2f} saat)")
                
                # GPU belleğini temizle
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                log.error(f"Görüntü işlenirken hata: {e}")
                continue

        # Epoch tamamlandı, epoch sonu checkpoint'i kaydet
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = np.mean(np.asarray(epoch_losses))
        
        save_path = f"{checkpoint_dir}/Classifier_epoch_{epoch+1}_final.weights.h5"
        Classifier.save_weights(save_path)
        
        log.info(f"EPOCH {epoch+1} TAMAMLANDI: Ortalama Loss = {avg_epoch_loss:.6f}")
        log.info(f"Epoch süresi: {epoch_time:.2f} saniye ({epoch_time/3600:.2f} saat)")
        
        elapsed_time = time.time() - start_time
        remaining_epochs = args.epochs - (epoch + 1)
        est_time_remaining = (elapsed_time / (epoch + 1)) * remaining_epochs if epoch > 0 else 0
        
        log.info(f"Toplam eğitim süresi şu ana kadar: {elapsed_time:.2f} saniye ({elapsed_time/3600:.2f} saat)")
        log.info(f"Tahmini kalan süre: {est_time_remaining:.2f} saniye (~{est_time_remaining/3600:.2f} saat)")
        log.info(f"===============================================================")

    # Eğitim tamamlandı
    total_elapsed = time.time() - start_time
    log.info(f"EĞİTİM TAMAMLANDI! Toplam süre: {total_elapsed:.2f} saniye ({total_elapsed/3600:.2f} saat)")

# ======================================================================================================================
if __name__ == "__main__":
    train_one_by_one() 