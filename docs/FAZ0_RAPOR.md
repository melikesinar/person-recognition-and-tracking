# Faz 0 Raporu: Temel Altyapı ve CPU Optimizasyonu

**Tarih:** 17.11.2026 · **Branch:** `feature/multicam-reid`

## 1. Projenin amacı

Birden fazla kamerada görünen kişileri **anonim** olarak takip etmek. Bir kişi bir kameradan çıkıp başka bir kamerada göründüğünde aynı kişi olduğunu anlamak ve böylece hareket yolunu ve bekleme sürelerini çıkarmak. Sistem kişinin kim olduğunu öğrenmeye çalışmaz, isim kullanmaz ve önceden kayıt gerektirmez. Her kişiye geçici bir numara verilir (P1, P2, ...).

Kullanım alanları: ofis giriş ve hareketleri, marketlerde reyonda bekleme süresi, AVM'de mağazalar arası geçişler.

**Kısıtlar:**
- Kamera sayısı belli değil, sistem kaç kamera olursa olsun çalışmalı.
- Kameralar farklı alanları görüyor (örneğin kapı, ofis içi, koridor).
- Yüz her zaman görünmeyebilir.
- Geliştirme bilgisayarında GPU yok.

## 2. Başlangıç durumu (önceki çalışma)

Önceki stajyerin projesi sağlam bir temel sunuyordu:
- YOLOv8 ile kişi tespiti
- BoostTrack ile takip
- OSNet ile görünüm vektörü
- Kişi başına çok vektörlü galeri
- FAISS ile arama
- MongoDB kaydı

Faz 0'daki incelemede şu sorunlar bulundu:

| # | Sorun | Etkisi |
|---|---|---|
| 1 | ReID modeline görüntü **BGR** renk sırasıyla veriliyordu. Model RGB ile eğitilmiş. | Renk kanalları ters (kırmızı ↔ mavi), eşleştirme doğruluğu düşer. |
| 2 | Her kamera OSNet'in **kendi kopyasını** yüklüyordu (+1 ana kopya). | Kamera sayısı kadar fazladan bellek ve yükleme süresi. |
| 3 | Her kişinin vektörü her karede **iki kez** hesaplanıyordu (tracker içinde ve kimlik mantığında). | En pahalı işlem iki katına çıkıyordu. |
| 4 | FAISS indeksi her güncellemede MongoDB'den baştan kuruluyordu, ama yazmalar asenkron ve gecikmeliydi. | Yeni kaydedilen kişi bir süre aramada görünmüyor, aynı kişi ikinci kez kaydedilebiliyordu. |
| 5 | MongoDB zorunluydu. | Veritabanı çalışmıyorsa sistem hiç açılmıyordu. |
| 6 | Kameralar ana döngüde sırayla okunuyordu, kamera listesi kodun içindeydi. | Yavaş bir kamera diğerlerini bekletiyordu. Kamera eklemek için kod değiştirmek gerekiyordu. |
| 7 | Tracker'ın tespit eşiği (0.60) sistemin tespit eşiğinden (0.55) yüksekti. | 0.55–0.60 güvenli tespitler sessizce atılıyordu. |
| 8 | Sabit kameralarda gereksiz kamera hareketi telafisi (ECC) açıktı. | CPU'da boşa harcanan işlem. |

**Ölçülen başlangıç performansı** (EPFL test videosu, 2 kamera, 4 kişi, CPU):
60 saniyede kamera başına sadece **52 ve 33 kare** işlenebildi (saniyede yaklaşık 0.5–0.9 kare). Hiç kimlik oluşmadı.

## 3. Yapılan değişiklikler

### 3.1 Kamera sayısından bağımsız mimari
- Kameralar artık `config/cameras.yaml` dosyasından okunuyor. Kamera eklemek için bir satır yazmak yeterli, kod değişmiyor.
- Kaynak olarak webcam, video dosyası veya IP/RTSP akışı kullanılabiliyor. Telefon da "IP Webcam" uygulamasıyla kamera olarak eklenebiliyor.
- Her kamera **kendi thread'inde** okunuyor ve sadece en yeni kare tutuluyor. Yavaş bir kamera diğerlerini etkilemiyor, gecikme birikmiyor.
- Her kamera için **işleme hızı** (`process_fps`) ayarlanabiliyor. Yürüyen bir insanı takip etmek için saniyede 30 kare gerekmiyor, 5–8 kare yeterli.
- Ekran düzeni kamera sayısına göre otomatik ızgara oluyor.

### 3.2 Doğruluk düzeltmeleri
- **Renk sırası hatası düzeltildi** (BGR → RGB).
- Tracker tespit eşiği sistem eşiğiyle aynı yapıldı.

### 3.3 Tek ReID modeli ve "ihtiyaç halinde" hesaplama
- OSNet sistemde **tek bir kez** yükleniyor ve tüm kameralar onu paylaşıyor.
- Kare kare takip, ReID olmadan hareket ve kutu örtüşmesiyle (Kalman + IoU) yapılıyor. Sabit kamerada bu yeterli.
- ReID vektörü sadece kimlik kararı için gerektiğinde hesaplanıyor:
  - Kimliği belli olmayan kişi: her işlenen karede (hızlı karar için).
  - Kimliği belli olan kişi: saniyede bir (galeri güncellemesi ve kimlik kontrolü için).

### 3.4 CPU hızlandırmaları (her biri ölçülerek eklendi)

| Değişiklik | Önce | Sonra | Çıktı değişti mi? |
|---|---|---|---|
| **Denormal sayı düzeltmesi** (`flush_denormal`) | 1452 ms / 4 kişi | 351 ms / 4 kişi | Hayır (fark: 0.0) |
| **TorchScript** ile model dondurma | 191 ms / kişi | 95 ms / kişi | Hayır (kosinüs: 1.0) |
| YOLO giriş boyutu 640 → 416 | ~150 ms / kare | ~95–130 ms / kare | Aynı tespitler |
| Thread sayısı sınırı (12 → 4) | | Küçük gruplarda daha hızlı | Hayır |
| Kare gönderimi ayrı thread'e alındı | Dedektör %59–92 dolu | Dedektör %99 dolu | Hayır |

*Denormal nedir?* Sinir ağı, hesap sırasında sıfıra son derece yakın sayılar üretiyor. Intel ve AMD işlemciler bu özel sayılarla normal sayılara göre çok daha yavaş işlem yapıyor. Bunları sıfır kabul etmek sonucu değiştirmiyor ama hesaplamayı 4 kat hızlandırıyor.

*TorchScript nedir?* PyTorch modelini Python'dan bağımsız, optimize edilmiş bir hale çeviriyor ve bazı katmanları birleştiriyor. OSNet çok sayıda küçük katmandan oluştuğu için katman başına gelen ek yük süreyi belirliyordu; TorchScript bu yükü kaldırıyor.

### 3.5 Veri katmanı
- FAISS + MongoDB'den yeniden kurma yapısı kaldırıldı. Yerine bellekte tutulan, her değişiklikte **anında güncellenen** ve tam (yaklaşık değil) sonuç veren bir vektör araması (`VectorIndex`) geldi.
  - Ofis ve mağaza ölçeğinde (yüzlerce kişi) bu hem daha hızlı hem de her zaman tutarlı.
  - Milyonlarca vektöre çıkılırsa aynı arayüzün arkasına tekrar FAISS konabilir.
- MongoDB **isteğe bağlı** oldu. Tanımlı değilse veya ulaşılamıyorsa sistem uyarı verip bellek modunda çalışıyor.

### 3.6 Ölçüm araçları
- Çalışma sırasında her 10 saniyede kamera başına gerçek işleme hızı ve dedektör doluluğu loglanıyor.
- `tools/eval_reid.py`: Standart **Market-1501** veri setinde (6 kamera, 1.501 kişi) kameralar arası tanıma başarısını ölçüyor (Rank-1, Rank-5, mAP). Farklı ayarları karşılaştırabiliyor, örneğin eski BGR hatası ile düzeltilmiş hali. Metrik hesabı elle doğrulanmış test durumlarıyla kontrol edildi.
- Ekransız test modu: `--headless --max-seconds N`.

## 4. Sonuç

Aynı test (EPFL oda videosu, 2 kamera, 4 kişi, aynı laptop, sadece CPU):

| | Önce | Sonra |
|---|---|---|
| İşlenen kare (kamera başına, dakikada) | 52 / 33 | ~237 / ~181 |
| Gerçek işleme hızı (kamera başına) | ~0.5–0.9 kare/sn | ~3–4 kare/sn |
| Oluşan kimlik | 0 | 6 (gerçekte 4 kişi var) |
| Yüklenen OSNet kopyası | 3 | 1 |
| MongoDB olmadan çalışma | Hayır | Evet |

Toplamda yaklaşık **4–5 kat hızlanma** sağlandı.

## 5. Bilinen sınırlar ve sonraki adımlar

1. **Bazı kişiler birden fazla kimlik alıyor** (4 kişiye 6 kimlik). Bu, tek bir genel görünüm vektörünün yetersiz kaldığını gösteriyor ve **Faz 1'in konusu**. Faz 1'de kıyafet renkleri (üst/alt beden), saç bölgesi, boy ve güvenilirliğe göre ağırlıklandırma eklenecek.
2. **CPU sınırı:** Bu laptopta tüm kameralar için toplam yaklaşık 6 kare/sn işlenebiliyor; darboğaz artık YOLO. Sonraki seçenekler:
   - **ONNX Runtime / OpenVINO:** Intel işlemcide genelde 2–4 kat hızlanma sağlar. Ek paket kurulumu gerekiyor.
   - Kamera sayısı arttıkça kamera başına işleme hızını düşürmek (config ayarı).
   - Kalıcı kurulumda GPU'lu veya Intel iGPU'lu bir mini PC.
3. **Gerçek doğruluk ölçümü:** Market-1501 veri seti indirildiğinde `tools/eval_reid.py` ile eski ve yeni halin sayısal karşılaştırması yapılacak.
4. **Arama kuralı:** En yakın 5 vektörden en çok oyu alan kişi seçiliyor. Bu kural, galerisinde çok vektör olan kişileri kayırabilir. Faz 1'de gözden geçirilecek.
5. **Kameralar arası geçiş grafı** (Faz 2): Hangi kameradan hangisine kaç saniyede geçilebileceğini tanımlamak. Birbirini görmeyen kameralarda yanlış eşleşmeleri ciddi oranda azaltır.

## 6. Nasıl çalıştırılır

```bash
# Laptop kamerası ile
python src/__main__.py

# EPFL test videoları ile (videos/room/ altına indirilmiş olmalı)
python src/__main__.py --cameras config/cameras.epfl-room.yaml

# ReID doğruluk ölçümü
python tools/eval_reid.py --data <Market-1501 klasörü> --num-ids 200
```
