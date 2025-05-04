'''
CDP Eğitim İzleme Aracı
'''

import json
import time
import argparse
import os
import sys
from datetime import datetime, timedelta

# Parse arguments
parser = argparse.ArgumentParser(description="CDP Eğitim İzleme Aracı")
parser.add_argument("--status_file", default="training_one_by_one_status.json", type=str, help="Status file to monitor")
parser.add_argument("--refresh_interval", default=5, type=int, help="Refresh interval in seconds")
args = parser.parse_args()

def format_time(seconds):
    """Saniye cinsinden süreyi okunabilir formata dönüştürür"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def format_eta(seconds):
    """Kalan süreyi tarih-saat formatında gösterir"""
    if seconds <= 0:
        return "Tamamlandı"
    now = datetime.now()
    eta = now + timedelta(seconds=seconds)
    return eta.strftime("%Y-%m-%d %H:%M:%S")

def print_progress_bar(progress, width=50):
    """İlerleme çubuğu oluşturur"""
    filled = int(width * progress / 100)
    bar = '█' * filled + '-' * (width - filled)
    return f"|{bar}| {progress:.2f}%"

def monitor_training():
    """Eğitim durumunu gerçek zamanlı olarak izler"""
    
    print("\033[2J\033[H")  # Terminal ekranını temizle
    print("CDP Eğitim İzleme Aracı - Ctrl+C ile çıkış")
    print("=" * 80)
    
    while True:
        try:
            # Status dosyasını oku
            if not os.path.exists(args.status_file):
                print(f"\rStatus dosyası bulunamadı: {args.status_file}. Yeniden deneniyor...", end="")
                time.sleep(args.refresh_interval)
                continue
                
            with open(args.status_file, 'r') as f:
                status = json.load(f)
            
            # Temel bilgileri çıkar
            current_epoch = status.get('current_epoch', 0)
            total_epochs = status.get('total_epochs', 0)
            current_image = status.get('current_image', 0)
            total_images = status.get('total_images', 0)
            overall_progress = status.get('overall_progress_percent', 0)
            epoch_progress = status.get('epoch_progress_percent', 0)
            elapsed_time = status.get('elapsed_time_seconds', 0)
            estimated_time = status.get('estimated_time_remaining_seconds', 0)
            last_loss = status.get('last_loss', 0)
            last_update = status.get('last_update', '')
            training_status = status.get('status', 'unknown')
            
            # Terminal ekranını temizle ve bilgileri göster
            print("\033[H")  # İmleci başa al
            print(f"CDP Eğitim İzleme Aracı - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Ctrl+C ile çıkış")
            print("=" * 80)
            
            # Genel durum
            print(f"Durum:            {training_status.upper()}")
            print(f"Son Güncelleme:   {last_update}")
            print(f"Toplam İlerleme:  {print_progress_bar(overall_progress)}")
            
            # Epoch bilgisi
            print(f"\nEpoch:            {current_epoch}/{total_epochs}")
            print(f"Epoch İlerleme:   {print_progress_bar(epoch_progress)}")
            
            # Görüntü bilgisi
            print(f"\nGörüntü:          {current_image}/{total_images}")
            
            # Zaman bilgisi
            print(f"\nGeçen Süre:       {format_time(elapsed_time)} ({elapsed_time/3600:.2f} saat)")
            print(f"Tahmini Kalan:    {format_time(estimated_time)} ({estimated_time/3600:.2f} saat)")
            print(f"Tahmini Bitiş:    {format_eta(estimated_time)}")
            
            # Performans bilgisi
            print(f"\nSon Loss:         {last_loss:.6f}")
            
            # Görüntü/saat hesapla
            if elapsed_time > 0:
                processed_images = (current_epoch - 1) * total_images + current_image
                images_per_hour = (processed_images / elapsed_time) * 3600
                print(f"Hız:              {images_per_hour:.2f} görüntü/saat")
            
            sys.stdout.flush()
            time.sleep(args.refresh_interval)
            
        except KeyboardInterrupt:
            print("\n\nİzleme sonlandırıldı.")
            break
        except Exception as e:
            print(f"\nHata oluştu: {e}")
            time.sleep(args.refresh_interval)

if __name__ == "__main__":
    monitor_training() 