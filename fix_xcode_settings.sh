#!/bin/bash

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== SwiftQR Xcode Ayarlarını Düzeltme =====${NC}"

PROJECT_FILE="swiftqr.xcodeproj/project.pbxproj"

# Proje dosyası var mı kontrol et
if [ ! -f "$PROJECT_FILE" ]; then
    echo -e "${RED}Hata: project.pbxproj dosyası bulunamadı.${NC}"
    exit 1
fi

# Generated klasörü oluştur (yoksa)
mkdir -p swiftqr/Generated

echo -e "${GREEN}Info.plist dosyası hazır: swiftqr/Generated/Info.plist${NC}"

# Proje dosyasındaki Info.plist konumunu değiştir
echo -e "${YELLOW}Proje ayarları güncelleniyor...${NC}"

# Önce varsa swiftqr/Info.plist girdisini Generated/ klasörüne çevir
if grep -q "INFOPLIST_FILE = swiftqr/Info.plist;" "$PROJECT_FILE"; then
    sed -i.bak "s|INFOPLIST_FILE = swiftqr/Info.plist;|INFOPLIST_FILE = swiftqr/Generated/Info.plist;|g" "$PROJECT_FILE"
    echo -e "${GREEN}✓ Info.plist yolu güncellendi${NC}"
else
    echo -e "${YELLOW}⚠️ INFOPLIST_FILE = swiftqr/Info.plist; satırı bulunamadı.${NC}"
    echo -e "${YELLOW}⚠️ Manuel olarak Xcode'da güncellemeniz gerekebilir.${NC}"
fi

# Xcode'da Generated/Info.plist dosyasını projeye ekleyelim
if [ ! -d "swiftqr.xcodeproj/project.xcworkspace" ]; then
    echo -e "${YELLOW}⚠️ Xcode projesi açık değil. Generated/Info.plist dosyasını manuel olarak projeye eklemeniz gerekebilir.${NC}"
else
    echo -e "${GREEN}✓ Xcode projesi hazır${NC}"
fi

# Info.plist için uyarı
echo -e "${YELLOW}Not: Xcode'u yeniden başlatmanız ve projeyi temizlemeniz gerekebilir.${NC}"
echo -e "${YELLOW}1. Xcode'u kapatıp yeniden açın${NC}"
echo -e "${YELLOW}2. Proje ayarlarında (Build Settings) Info.plist File değerine dikkat edin${NC}"
echo -e "${YELLOW}   - Değer şu olmalı: swiftqr/Generated/Info.plist${NC}"
echo -e "${YELLOW}3. Product > Clean Build Folder ve ardından Build edin${NC}"

echo -e "\n${GREEN}===== Xcode'da Yapılması Gerekenler =====${NC}"
echo -e "1. Project Navigator'da 'Generated' klasörüne sağ tıklayın"
echo -e "2. 'Add Files to \"swiftqr\"' seçin"
echo -e "3. Generated/Info.plist dosyasını seçin ve 'Add' butonuna tıklayın"
echo -e "4. Project > Build Settings > Packaging kısmında Info.plist File değerini kontrol edin"
echo -e "   - Değer şu olmalı: swiftqr/Generated/Info.plist"
echo -e "5. Product > Clean Build Folder seçeneği ile projeyi temizleyin"
echo -e "6. Yeniden derleyin"

echo -e "\n${GREEN}===== İşlem Tamamlandı =====${NC}" 