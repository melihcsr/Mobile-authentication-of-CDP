'''
Daha basit bir CDP eğitim scripti - augmentation olmadan
Author: Training script based on Olga TARAN's work, University of Geneva, 2021
'''

from __future__ import print_function

import argparse
import yaml
import sys
import os
import time
import json
sys.path.insert(0,'..')

import libs.yaml_utils as yaml_utils
from libs.utils import *

from libs.ClassifierDataLoader import ClassifierDataLoader
from libs.ClassificationModel import ClassificationModel

# ======================================================================================================================
parser = argparse.ArgumentParser(description="TRAIN: a simpler version without augmentation")
parser.add_argument("--config_path", default="./supervised_classification/configuration.yml", type=str, help="The config file path")
# datset parameters
parser.add_argument("--image_type", default="rgb", type=str, choices=['rgb', 'gray'], help="The image type")
parser.add_argument("--data_paths", default=["./data/original/rgb/",
                                             "./data/fakes_1/paper_white/rgb/"], type=str, help="The data paths - using fewer paths for speed")
parser.add_argument("--templates_path", default="./data/binary_templates/", type=str, help="Binary templates path")
# model parameters
parser.add_argument("--lr", default=1e-4, type=float, help="Training learning rate")
parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
parser.add_argument("--n_classes", default=2, type=int, choices=[2, 5], help="The number classes: 2 - original or fakes, 5 - original and 4 types of fakes")
parser.add_argument("--is_max_pool", default=True, type=int, help="Is to use max pooling in the trained model?")
parser.add_argument("--log_interval", default=1, type=int, help="How often to log batch progress (in batches)")
parser.add_argument("--status_file", default="training_simple_status.json", type=str, help="File to save training status")
# log mode
parser.add_argument("--is_debug", default=True, type=int, help="Is debug mode?")
# Augmentation
parser.add_argument("--no_augmentation", default=True, type=int, help="Turn off data augmentation")

args = parser.parse_args()
# ======================================================================================================================
set_log_config(args.is_debug)
log.info("PID = %d\n" % os.getpid())
log.info("BASİT EĞİTİM MODU: Sadece 2 veri yolu, veri artırma kapalı")

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
        
def update_status_file(current_epoch, total_epochs, elapsed_time, loss, checkpoint_path):
    """
    Eğitim durumunu bir dosyaya kaydeder
    """
    status = {
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'progress_percent': (current_epoch / total_epochs) * 100,
        'elapsed_time_seconds': elapsed_time,
        'elapsed_time_hours': elapsed_time / 3600,
        'last_loss': float(loss),
        'last_checkpoint': checkpoint_path,
        'status': 'in_progress' if current_epoch < total_epochs else 'completed',
        'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(args.status_file, 'w') as f:
        json.dump(status, f, indent=4)
    
    log.info(f"Durum dosyası güncellendi: {args.status_file}")
    
    # Ayrıca son durumu ekrana basalım
    if current_epoch == total_epochs:
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

def train_simple():
    log.info("train_simple fonksiyonu başladı")
    
    if args.no_augmentation:
        args.config_path = configure_no_augmentation()
    
    log.info(f"Konfigürasyon dosyası: {args.config_path}")
    config = yaml_utils.Config(yaml.load(open(args.config_path), Loader=yaml.SafeLoader))

    args.checkpoint_dir = "%s_supervised_classifier_n_%d_simple" % (args.image_type, args.n_classes)
    args.dir = "supervised_classifier_lr%s_simple" % args.lr
    log.info(f"Checkpoint klasörü: {args.checkpoint_dir}")
    log.info(f"Sonuç klasörü: {args.dir}")

    # Klasörleri oluşturalım
    os.makedirs(f"checkpoints/{args.checkpoint_dir}", exist_ok=True)
    os.makedirs(f"results/{args.dir}/prediction", exist_ok=True)
    os.makedirs(f"TensorBoard/{args.dir}", exist_ok=True)

    log.info("Start Model preparation.....")
    model = ClassificationModel(config, args)
    Classifier = model.ClassifierModel

    log.info("Start Train Data loading.....")
    # Seed değişkenini ekleyelim
    if not hasattr(args, 'seed'):
        args.seed = 42  # Sabit bir seed değeri
    log.info(f"Seed değeri: {args.seed}")
    
    log.info("DataGenTrain oluşturuluyor...")
    DataGenTrain = ClassifierDataLoader(config, args, type="train", is_debug_mode=args.is_debug)
    log.info("DataGenTrain.initDataSet() çağrılıyor...")
    DataGenTrain.initDataSet()
    log.info("DataGenTrain.initDataSet() tamamlandı")

    # === model scheme visualisation ===============================================================================
    if args.is_debug:
        log.info("EstimationModel")
        model.Classifier.summary()

    # Veri kümesi hakkında bilgi
    log.info(f"Veri kümesi bilgisi: {DataGenTrain.n_batches} batch, batch_size={config.batchsize}")
    log.info(f"Toplam eğitim örneği: {DataGenTrain.n_batches * config.batchsize}")
    
    # Eğer önceki durumdan devam ediyorsak, durum dosyasını okuyalım
    last_completed_epoch = 0
    if os.path.exists(args.status_file):
        try:
            with open(args.status_file, 'r') as f:
                status = json.load(f)
                last_completed_epoch = status.get('current_epoch', 0)
                log.info(f"Durum dosyası bulundu. En son tamamlanan epoch: {last_completed_epoch}/{args.epochs}")
                
                # Eğer zaten tamamlanmışsa, kullanıcıya bildir
                if status.get('status') == 'completed':
                    log.info("Eğitim zaten tamamlanmış! Tekrar başlatmak için durum dosyasını silin.")
                    return
        except:
            log.info("Durum dosyası okunamadı. Eğitim baştan başlayacak.")
    
    # === Training =================================================================================================
    start_time = time.time()
    
    for epoch in range(last_completed_epoch, args.epochs):
        epoch_start_time = time.time()
        log.info(f"--- EPOCH {epoch+1}/{args.epochs} BAŞLIYOR ---")
        
        Loss = []
        batches = 0
        save_each = saveSpeed(epoch)
        
        log.info(f"Epoch {epoch+1} için {DataGenTrain.n_batches} batch işlenecek")
        
        for x_batch, labels in DataGenTrain.datagen:
            batch_start_time = time.time()
            
            # Her batch için işlem
            log.info(f"Batch {batches+1} işleniyor, boyut: {x_batch.shape}")
            loss = Classifier.train_on_batch(x_batch, labels)
            Loss.append(loss)
            batches += 1
            
            # Her log_interval batch'de bir ilerleme bilgisi yazdır
            if batches % args.log_interval == 0 or batches == 1 or batches == DataGenTrain.n_batches:
                batch_time = time.time() - batch_start_time
                avg_loss = np.mean(np.asarray(Loss[-min(args.log_interval, len(Loss)):]))
                elapsed = time.time() - start_time
                
                # Kalan süreyi hesapla
                batch_per_sec = batches / (time.time() - epoch_start_time)
                remaining_batches = DataGenTrain.n_batches - batches
                est_time_epoch = remaining_batches / batch_per_sec if batch_per_sec > 0 else 0
                
                # İlerleme çubuğu ve detaylı bilgi
                print_progress_bar(batches, DataGenTrain.n_batches, 
                                 prefix=f'Epoch {epoch+1}/{args.epochs}', 
                                 suffix=f'İşlendi: {batches}/{DataGenTrain.n_batches}, Loss: {avg_loss:.6f}', 
                                 length=30)
                
                log.info(f"Batch {batches}/{DataGenTrain.n_batches}, Loss: {loss:.6f}, Avg Loss: {avg_loss:.6f}")
                log.info(f"Batch süresi: {batch_time:.2f}s, Tahmini kalan süre (epoch): {est_time_epoch:.2f}s")
            
            if batches >= DataGenTrain.n_batches:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        # Epoch tamamlandı
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = np.mean(np.asarray(Loss))
        
        log.info(f"EPOCH {epoch+1} TAMAMLANDI: Ortalama Loss = {avg_epoch_loss:.6f}")
        log.info(f"Epoch süresi: {epoch_time:.2f} saniye")

        # Her epoch sonunda model ağırlıklarını kaydet
        save_path = "%s/Classifier_epoch_%d.weights.h5" % (model.checkpoint_dir, epoch+1)
        log.info(f"Model ağırlıkları kaydediliyor: {save_path}")
        Classifier.save_weights(save_path)
        
        elapsed_time = time.time() - start_time
        remaining_epochs = args.epochs - (epoch + 1)
        est_time_remaining = (elapsed_time / (epoch + 1)) * remaining_epochs if epoch > 0 else 0
        
        log.info(f"Toplam eğitim süresi şu ana kadar: {elapsed_time:.2f} saniye")
        log.info(f"Tahmini kalan süre: {est_time_remaining:.2f} saniye (~{est_time_remaining/3600:.2f} saat)")
        
        # Durum dosyasını güncelle
        update_status_file(epoch+1, args.epochs, elapsed_time, avg_epoch_loss, save_path)
        
        log.info(f"===============================================================")

    # Eğitim tamamlandı
    total_elapsed = time.time() - start_time
    log.info(f"EĞİTİM TAMAMLANDI! Toplam süre: {total_elapsed:.2f} saniye ({total_elapsed/3600:.2f} saat)")

# ======================================================================================================================
if __name__ == "__main__":
    train_simple() 