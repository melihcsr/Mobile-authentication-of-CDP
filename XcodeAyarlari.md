# SwiftQR Xcode Ayarları Kılavuzu

Bu kılavuz, SwiftQR uygulamasının Info.plist dosyasını doğru bir şekilde yapılandırmanız için gereken adımları içerir.

## Info.plist Ayarlarını Yapılandırma

"Build input file cannot be found: '/Users/melihcesur/Documents/GitHub/Mobile-authentication-of-CDP/swiftqr/swiftqr/Info.plist'." hatasını çözmek için şu adımları uygulayın:

### 1. Xcode'da Projeyi Açın

Xcode'da SwiftQR projesini açın.

### 2. Proje Ayarlarına Gidin

- Sol taraftaki gezinti bölmesinde, proje dosyanıza tıklayın (swiftqr.xcodeproj)
- Sol tarafta açılan panelden "swiftqr" hedefini seçin
- Üst kısımdaki sekmelerden "Build Settings" sekmesine tıklayın

### 3. Info.plist Konumunu Güncelleme

- Arama kutusunda "Info.plist" yazarak ilgili ayarı bulun
- "Info.plist File" ayarını "swiftqr/Generated/Info.plist" olarak değiştirin
  
  ![Info.plist Ayarı](https://i.imgur.com/example.png)

### 4. İzinler ve ATS Ayarları (Alternatif Yöntem)

Eğer Info.plist dosyasını doğrudan Xcode'da düzenlemek isterseniz:

- Info sekmesinde:
  - Camera Usage Description: "QR kodlarını taramak için kamera erişimi gereklidir."
  - Photo Library Additions Usage Description: "QR kod görüntülerini galeriye kaydetmek için izin gereklidir."

- Kaynak olarak Info.plist'i açın ve şu App Transport Security ayarlarını ekleyin:
```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
    <key>NSExceptionDomains</key>
    <dict>
        <key>192.168.1.152</key>
        <dict>
            <key>NSExceptionAllowsInsecureHTTPLoads</key>
            <true/>
            <key>NSIncludesSubdomains</key>
            <true/>
        </dict>
        <key>localhost</key>
        <dict>
            <key>NSExceptionAllowsInsecureHTTPLoads</key>
            <true/>
            <key>NSIncludesSubdomains</key>
            <true/>
        </dict>
    </dict>
</dict>
```

### 5. Projeyi Temizleme

Değişiklikleri yaptıktan sonra:

1. Xcode menüsünden "Product" > "Clean Build Folder" seçeneğini tıklayın
2. Ardından "Product" > "Build" ile projeyi yeniden derleyin

## Sorun Devam Ederse

Eğer sorun devam ederse, şu adımları deneyebilirsiniz:

1. Xcode'u tamamen kapatıp yeniden açın
2. Derived Data klasörünü temizleyin:
   - Xcode > Preferences > Locations > Derived Data konumundaki klasörü silin veya
   - Terminal'de şu komutu çalıştırın: `rm -rf ~/Library/Developer/Xcode/DerivedData/*`
3. Projeyi tekrar açıp derleyin

Bu adımlar, Info.plist dosyasıyla ilgili yapılandırma hatalarını çözmelidir. 