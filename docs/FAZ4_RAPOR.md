# Faz 4 Raporu: İstatistik Paneli, Isı Haritası, Alanlar, Giriş-Çıkış Sayacı, Gizlilik, Sistem Sağlığı

**Tarih:** 25.11.2026 · **Branch:** `feature/faz4-analytics`

## 1. Hedef

Faz 0–3'te sistem "kim kimdir" sorusunu çözmeye odaklandı. Bu bilginin bir **kullanıcıya** gösterilmesi ise henüz yoktu: rotalar sadece kapanışta bir CSV dosyasına ve loga yazılıyordu.

Faz 4'ün hedefi, projenin asıl ürününü görünür kılmak:
- Kim nerede, ne kadar kaldı, nereye gitti?
- Hangi alanlar yoğun?
- Kaç kişi girdi, kaç kişi çıktı?

Ayrıca ofise kurulum öncesi gereken iki konu: **gizlilik** (KVKK) ve **sistem sağlığı** (kamera koparsa ne olur).

**Gerçek ofis verisi olmadan nasıl?** Panelin ihtiyacı olan veri, sistemin her çalıştırmada ürettiği ham kayıt. Bu kayıt şu an EPFL videosu, WILDTRACK veri seti, laptop kamerası veya telefon kamerasıyla üretilebiliyor. Ofis kameraları bağlandığında panel hiç değişmeden onların verisini gösterecek.

## 2. Mimari: tek bir ham kayıt, her şey ondan hesaplanıyor

```
Sistem (canlı / replay / WILDTRACK)
   │  her kimlikli kişinin ayak noktası, 0.5 sn'de bir
   ▼
output/sessions/<tarih_saat>[_etiket]/
   meta.json              kameralar, alanlar, sayaç çizgileri, görüntü boyutu, saat türü
   positions.csv          t, kamera, kişi, x, y
   merges.csv             birleştirilen kimlikler (P5 → P3)
   health.csv             kamera başına işleme hızı ve bağlantı durumu (10 sn'de bir)
   background_cam<N>.jpg  ısı haritası için boş sahne görüntüsü
   │
   ▼
src/analytics/stats.py    kalış, geçiş, yoğunluk, ısı haritası, sayaç hesapları
   │
   ▼
dashboard/app.py          Streamlit web paneli (http://localhost:8501)
```

**Neden ham kayıt + sonradan hesap?**
1. **Tek doğruluk kaynağı:** Kalış süresi, rota, alan, sayaç aynı veriden hesaplanıyor, birbiriyle çelişemiyor.
2. **Ayarlar sonradan değişebiliyor:** "Ziyaret arası kaç saniye olsun" gibi ayarlar panelde değiştirilip sonuç anında yeniden hesaplanıyor. Kaydı tekrar almak gerekmiyor.
3. **Kimlik birleştirmeleri geriye dönük uygulanıyor:** Sistem 2 dakika sonra "P5 aslında P3'müş" derse bu `merges.csv`'ye yazılıyor ve panel P5'in bütün geçmişini P3'e aktarıyor. Zincirleme birleştirmeler de çözülüyor (P5 → P3, sonra P3 → P1 ise P5 → P1).
4. **Ucuz:** Sadece sayılar yazılıyor. 150 saniyelik EPFL kaydı 1.094 satır (~30 KB).

## 3. Yapılan değişiklikler

### 3.1 Oturum kaydı (`src/analytics/recorder.py`, yeni)
- `PresenceRegistry.observe` (Faz 2'den, her karede her kişi için çağrılıyor) artık kayıt nesnesine de haber veriyor. Kişi başına kamera başına en fazla 0.5 sn'de bir satır yazılıyor (`ANALYTICS_SAMPLE_SECONDS`).
- **Boş sahne arka planı:** 2 saniye arayla alınan 15 karenin piksel bazında **ortancası** (median) alınıyor. Yürüyen biri bir pikselde 15 karenin çoğunda bulunmadığı için ortancada kayboluyor; geriye boş oda kalıyor. Her örnekte kafalar ayrıca pikselleştiriliyor.
- Dosyalar birkaç saniyede bir diske yazılıyor (flush). Panel, sistem çalışırken de veriyi okuyabiliyor.
- Canlı sistemde varsayılan olarak açık (`ANALYTICS_ENABLED`). `tools/replay.py --record etiket` ve `tools/eval_wildtrack.py --record etiket` ile kayıtlı veriden de oturum üretiliyor.

### 3.2 Hesap modülü (`src/analytics/stats.py`, yeni)
| Fonksiyon | Ne hesaplıyor |
|---|---|
| `visits` | Bir kişinin bir yerde (kamera ya da alan) kesintisiz kalışı. `gap` saniyeden uzun görünmezse ziyaret biter. Her örnek 0.5 sn sayılır, tek görülme 0 sn olmaz. |
| `transitions` | Bir kişinin art arda iki farklı yerdeki ziyareti = geçiş. Zamanda örtüşen ziyaretler (aynı odayı gören iki kamera) geçiş sayılmaz. |
| `occupancy` | Zaman dilimi başına kamerada görülen farklı kişi sayısı |
| `heatmap` | Görüntü ızgarasında, her hücrede geçirilen toplam saniye |
| `zone_samples` | Hangi örnek hangi alanın içinde (ışın atma / ray casting yöntemiyle çokgen içinde nokta testi) |
| `line_events` | Sayaç çizgisinden geçişler ve yönü (giriş / çıkış) |
| `routes` | Kişi başına "kapı 8 sn → koridor 41 sn" metni |

### 3.3 Alanlar (zones)
Kamera dosyasında her kameraya isimli çokgenler eklenebiliyor:
```yaml
cameras:
  - id: 0
    name: room-c0
    source: videos/room/4p-c0.avi
    zones:
      desenli_hali: [[4, 140], [118, 90], [189, 114], [40, 178]]
      mavi_hali: [[240, 112], [358, 160], [358, 288], [60, 288], [15, 225]]
    lines:
      orta_hat: [[180, 110], [180, 288]]
```
- Bir kişi, **ayak noktası** (kutunun alt ortası) alanın içindeyse o alanda sayılıyor. Alanlar bu yüzden zemine çizilmeli.
- Kalış süresi artık "kamera" yerine "reyon 3", "kasa", "toplantı masası" gibi **alan** bazında da hesaplanıyor. Market senaryosu ancak bununla gerçekten karşılanıyor.
- Alanlar arası geçişler sadece **aynı kamera içinde** sayılıyor. İki kamera aynı zemini görüyorsa, kameralar arası "alan geçişi" yanıltıcı olurdu (ilk denemede EPFL'de 1 saniyelik sahte geçişler çıktı, düzeltildi).

### 3.4 Giriş-çıkış sayacı
Bir kameraya A→B çizgisi çekiliyor. A'da durup B'ye bakıldığında **sağ tarafa** geçiş "giriş", sol tarafa geçiş "çıkış" sayılıyor. Çizim aracı giriş yönünü okla gösteriyor.

Yanlış saymayı önleyen üç kural:
1. **Ölü bölge:** Kişinin tarafı, çizgiden 6 pikselden uzaktayken belirleniyor. Çizginin üstünde duran biri ileri geri sallansa da sayılmıyor.
2. **Gerçek kesişim:** Hareket, çizilen **doğru parçasını** kesmeli. Çizginin uzantısından geçen sayılmıyor.
3. **Boşluk kuralı:** 3 saniyeden uzun aralıklı iki konum birleştirilmiyor (kişi o arada görünmüyordu, arada ne yaptığı bilinmiyor).

Panel giriş, çıkış ve "giriş − çıkış" değerlerini gösteriyor. Tek girişli bir mekânda bu fark, içerideki kişi sayısı.

### 3.5 Alan ve çizgi çizme aracı (`tools/draw_zones.py`, yeni)
```bash
.venv\Scripts\python.exe tools\draw_zones.py --cameras config\cameras.yaml --camera 0
```
- Kameradan bir kare alıp açıyor, fareyle noktalara tıklanıyor.
- `z` ile alan, `l` ile çizgi bitiriliyor, adı terminale yazılıyor.
- `s` ile kamera dosyasına yapıştırılacak YAML metni yazdırılıyor.
- Var olan alanları ve çizgileri de gösteriyor. Çizgilerde giriş yönü okla işaretli.

### 3.6 Panel (`dashboard/app.py`, yeni; Streamlit)
**Streamlit nedir?** Python koduyla web sayfası yapmaya yarayan bir kütüphane. HTML/JavaScript yazmadan tablo, grafik ve resim gösterilebiliyor. Grafikler Altair ile çiziliyor (Streamlit ile birlikte geliyor).

| Sekme | İçerik |
|---|---|
| Üst satır | Görülen kişi, şu an görünen, ortalama kalış, en yoğun kamera, süre |
| 📊 Genel bakış | Zaman içinde kamera başına kişi sayısı grafiği; kamera bazında kalış tablosu (ortalama, medyan, en uzun, toplam); şu an kim nerede |
| 🧭 Rotalar | Kameralar arası geçiş matrisi ve süreleri (topolojideki `min_seconds` değerleri buradan okunabilir); kişi başına rota tablosu |
| 🔥 Isı haritası | Kamera seçilir; boş sahne üstüne yoğunluk renkleri; alanlar ve sayaç çizgileri çizili; en yoğun noktada geçen süre |
| 🟦 Alanlar | Alan bazında kalış tablosu ve grafiği; alanlar arası geçişler |
| 🚪 Giriş-çıkış | Çizgi başına giriş / çıkış / fark; zaman içinde "içerideki" grafiği; tüm geçişler listesi |
| 🩺 Sistem sağlığı | Kamera başına gerçek işleme hızı / hedef hız, bağlantı durumu, zaman grafiği |

Kenar çubuğu:
- Oturum seçimi (en yeni üstte).
- **Canlı yenile:** Sistem çalışırken sayfa 5 saniyede bir kendini günceller.
- "Ziyaret arası" ve "En kısa ziyaret" ayarları.
- Gizlilik notu.

Canlı oturumlarda zaman ekseni saat (16:03:25), kayıtlarda saniye olarak gösteriliyor.

### 3.7 Gizlilik (`src/utils/privacy.py`, yeni)
1. **Kafa pikselleştirme:** Ekranda gösterilen ve diske yazılan her görüntüde (canlı pencere, olay ekran görüntüleri, panel arka planları), her kişi kutusunun kafa bölgesi pikselleştiriliyor.
   - Yüz tespit modeli gerekmiyor: kafa, kişi kutusunun üst kısmı. Kişi kameraya arkasını dönmüş olsa da çalışıyor ve CPU'ya ek yük getirmiyor.
   - Kafa yüksekliği = max(kutu yüksekliğinin %22'si, kutu genişliğinin 0.6 katı). İkinci terim, kameraya yakın durup sadece üst bedeni görünen kişiler için.
   - Bulanıklaştırma değil **pikselleştirme:** keskinleştirmeyle geri getirilemez.
   - Pikselleştirme, kimlik kararları **verildikten sonra** yapılıyor. Doğrulama: WILDTRACK ölçümü pikselleştirme açıkken birebir aynı sonucu verdi.
   - Takip edilen kişilerin yanında, henüz takibe girmemiş ham tespitler de pikselleştiriliyor.
2. **Saklama süresi:** `RETENTION_DAYS` (varsayılan 30) günden eski oturum klasörleri sistem açılışında siliniyor (`0` = hiç silme).
3. **Saklanmayanlar:** İsim, yüz görüntüsü veya kişi kırpıntısı diske yazılmıyor. Oturum kaydında sadece numara, zaman ve koordinat var. Görünüm vektörleri (OSNet, renk histogramı) sadece bellekte tutuluyor; MongoDB açıksa orada da saklanıyor.

### 3.8 Kamera kopması ve sağlık (`src/core/camera_stream.py`)
- **Canlı kamera** (webcam / RTSP / IP) 3 saniye görüntü vermezse kapatılıp yeniden açılıyor. Denemeler arası bekleme 1, 2, 4, 8... saniye, en fazla 30 saniye.
- Açılışta ulaşılamayan canlı kamera artık sistemi "kamera yok" durumunda bırakmıyor; bağlanana kadar denenmeye devam ediliyor.
- Her kameranın durumu: `ok` (bağlı) / `stalled` (görüntü gelmiyor) / `reconnecting` (yeniden bağlanıyor) / `failed` (video dosyası açılamadı). Durum 10 saniyede bir hem loga hem `health.csv`'ye yazılıyor; panel gösteriyor.
- Test: var olmayan bir RTSP adresiyle sistem çökmedi, "yeniden bağlanıyor" durumunda kalıp denemeye devam etti.

## 4. Denemeler

| Veri | Ne denendi | Sonuç |
|---|---|---|
| EPFL (150 sn, replay, `--record epfl`) | Alanlar, sayaç, ısı haritası, rotalar | 6 kişi, 1.094 konum satırı. Mavi halı en yoğun alan (iki kamerada da ~4 dk toplam). Sayaç: 18 giriş, 21 çıkış. |
| EPFL canlı (`src/__main__.py`, 75 sn) | Canlı yenileme, sağlık sekmesi | Panel 5 sn'de bir güncellendi; iki kamera 5.9–6.0 / 6 kare/sn, "bağlı" |
| WILDTRACK (3 kamera, 100 sn, `--record wildtrack`) | HD kamerada ısı haritası; kaydın ölçüme etkisi | Isı haritası çalışıyor. Kimlik ölçümü kayıt ve pikselleştirme açıkken **birebir aynı** (global IDP %55.2) |
| Ulaşılamayan RTSP adresi | Kopma ve yeniden bağlanma | Sistem çökmedi, yeniden bağlanmayı denedi |
| Birim testleri (`tests/test_faz4_analytics.py`) | Hesapların doğruluğu | **9/9 geçti:** ziyaret ve geçiş süreleri, uzun boşlukta ziyaretin bölünmesi, örtüşen kameralarda sahte geçiş olmaması, zincirleme birleştirme, örnekleme sıklığı, alan içi süre, sayaç yönü ve gürültü kuralları, ısı haritası, kafa pikselleştirme |
| Önceki testler | Gerileme var mı | Faz 2 (6), metrik (4), toparlama (5) testleri geçiyor |

## 5. Dürüst değerlendirme ve sınırlar

1. **İstatistikler, kimlik doğruluğu kadar iyi.** Bir kişi iki numaraya bölünürse panelde iki kişi görünür. Numaralar karışırsa rotalar da karışır. Faz 1–3'teki iyileştirmelerin hepsi doğrudan panelin doğruluğunu belirliyor.
2. **Sadece numara almış kişiler sayılıyor.** Henüz numara almamış (`T12 (?)`) ya da hiç numara alamamış kişiler panelde yok. Seyrek ortamda (ofis) etkisi küçük. Kalabalıkta (WILDTRACK, bulunanların sadece %34'ü numara alıyor) ısı haritası ve sayaç eksik sayar.
   - Çözüm önerisi: Isı haritası ve sayaç kimlik gerektirmiyor, kamera içindeki **track** konumlarıyla da hesaplanabilir. Bir sonraki adım olarak eklenebilir.
3. **EPFL'deki sayaç değerleri anlamlı değil.** Çizgi bir odanın ortasında; insanlar sürekli ileri geri yürüyor. Sayaç bir kapı ya da koridor girişinde anlam kazanır.
4. **EPFL'de kameralar arası "geçiş" anlamsız.** İki kamera aynı odayı görüyor. Panel bunu not olarak yazıyor. Gerçek geçiş ölçümü ancak ayrı alanları gören kameralarla mümkün.
5. **Boş sahne görüntüsü kalabalıkta iz bırakabiliyor.** Çok kalabalık ya da uzun süre aynı yerde duran insanlar ortanca görüntüde silik izler bırakıyor (WILDTRACK'te görüldü). Kafalar yine pikselli.
6. **RTSP bağlantısı tamamen donarsa** (`read()` hiç dönmezse) yeniden bağlanma devreye giremez; OpenCV bunun için zaman aşımı sunmuyor. Gerçek kamerayla denenmeli.
7. **Panelde kullanıcı girişi yok.** Şu an sadece bu bilgisayardan (`localhost`) açılıyor. Ofis ağında paylaşılacaksa önce kullanıcı adı/şifre eklenmeli.

## 6. Nasıl denenir

```bash
# 1) Bir oturum üret (EPFL videosu, alanlar ve sayaç tanımlı)
.venv\Scripts\python.exe tools\replay.py --cameras config\cameras.epfl-room.yaml --seconds 150 --record epfl

# 2) Paneli aç, tarayıcıda http://localhost:8501
.venv\Scripts\streamlit.exe run dashboard\app.py

# Canlı: sistem çalışırken panelde en yeni oturum 5 sn'de bir yenilenir
.venv\Scripts\python.exe src\__main__.py --cameras config\cameras.epfl-room.yaml

# Kendi kamerana alan / sayaç çizgisi çiz
.venv\Scripts\python.exe tools\draw_zones.py --cameras config\cameras.yaml --camera 0

# WILDTRACK'ten oturum
.venv\Scripts\python.exe tools\eval_wildtrack.py --data D:\datasets\Wildtrack --cams 1,2,6 --record wildtrack

# Testler
.venv\Scripts\python.exe tests\test_faz4_analytics.py
```

**Evde gerçek test önerisi:** `config/cameras.yaml` dosyasına laptop kamerası (`source: 0`) ve telefon (IP Webcam uygulaması, `source: http://<telefon-ip>:8080/video`) iki ayrı oda olarak eklenir. Topolojiye aralarındaki yürüme süresi yazılır, kapıya sayaç çizgisi çekilir. Odalar arasında yürünür; panelde kendi rotan, kalış sürelerin ve giriş-çıkışların görünür. Bu, birbirini görmeyen kameralarla yapılan **ilk gerçek test** olur.
