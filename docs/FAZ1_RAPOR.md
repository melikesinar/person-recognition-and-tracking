# Faz 1 Raporu: Çok İpuçlu Kimlik Eşleştirme

**Tarih:** 17.11.2026 · **Branch:** `feature/faz1-multi-cue`

## 1. Hedef

Faz 0 sonunda sistem kişiyi tek bir genel görünüm vektörüyle (OSNet) tanıyordu. Test videosunda 4 kişiye 6 kimlik verildi; kadrajdan çıkıp geri gelen kişiler çoğunlukla tanınamadı.

Faz 1'in hedefi: Vücudun **farklı bölgelerinden ayrı ipuçları** çıkarmak, görünmeyen bölgeyi hesaptan düşürüp kalanlarla karar vermek.

## 2. Önce ölçüm: hangi ipucu ne kadar ayırt edici?

Etiketli veri olmadan bir ölçüm yöntemi kurduk. Tracker'ın aynı kişi olarak takip ettiği görüntüler (en az 2 saniye arayla) "aynı kişi", aynı karede görünen farklı kişiler "farklı kişi" sayıldı. EPFL oda videosu, 2 kamera; yaklaşık 6.800 aynı kişi çifti ve 2.200 farklı kişi çifti.

**AUC:** Rastgele bir "aynı kişi" çiftine, rastgele bir "farklı kişi" çiftinden daha yüksek benzerlik verme olasılığı. 1.0 kusursuz, 0.5 yazı-tura.

| İpucu | Kamera 0 AUC | Kamera 1 AUC |
|---|---|---|
| OSNet (genel görünüm) | 0.825 | 0.777 |
| **Alt beden rengi** | **0.812** | **0.780** |
| Üst beden rengi | 0.695 | 0.719 |
| Kafa/saç bölgesi | 0.574 | 0.519 |

**Bulgular:**
1. Basit bir renk histogramı olan **alt beden rengi neredeyse OSNet kadar ayırt edici**.
2. Kafa bölgesi bu çözünürlükte (kafa 10–15 piksel) neredeyse yazı-tura düzeyinde. Varsayılan ağırlığı 0 yapıldı; yüksek çözünürlüklü kameralarda açılabilir.
3. Bu videoda aynı kişinin OSNet benzerliği ortalama sadece **0.72–0.76**. Sistemin eşiği ise 0.88'di.

## 3. Eski eşik neden çalışmıyordu?

| Eşik (sadece OSNet) | Aynı kişiyi tanıma oranı | Yanlış eşleşme oranı |
|---|---|---|
| 0.85 (kayıt sırasındaki kontrol) | %6.4 | %0.04 |
| 0.88 (yeni gelen kişi) | **%1.8** | %0.00 |

Sistem yanlış eşleşmeden kaçınmak için o kadar temkinliydi ki geri gelen kişilerin yalnızca %2'sini tanıyabiliyordu. Fazladan kimlik oluşmasının ana sebebi buydu.

## 4. Birleştirme: aynı yanlış eşleşme oranında ne kazandık?

Adil karşılaştırma için iki yöntem **aynı yanlış eşleşme oranına** ayarlandı:

| Yanlış eşleşme oranı | Sadece OSNet: tanıma | OSNet + renk: tanıma |
|---|---|---|
| %0.5 | %14.4 | **%41.6** |
| %1 | %19.9 | **%46.7** |
| %2 | %27.6 | **%56.4** |

Aynı güvenlik seviyesinde **2–3 kat daha fazla doğru tanıma**. Genel AUC 0.787'den 0.851'e çıktı; hata oranı yaklaşık %30 azaldı.

Not: Bu oranlar tek bir görüntüyü tek bir görüntüyle karşılaştırıyor. Sistemde her kişinin galerisinde birçok görüntü var; gerçek tanıma oranı daha yüksek olur.

## 5. Yapılan değişiklikler

### 5.1 Bölgesel renk ipuçları (`src/core/appearance_cues.py`, yeni)
- Kişi kutusu sabit oranlarla bölgelere ayrılıyor:
  - Kafa: %0–17
  - Üst beden: %20–52
  - Alt beden: %55–90 (ayakkabı ve zemin dışarıda bırakılıyor)
- Kutunun yanlarındaki arka planı almamak için her bölgenin sadece orta kısmı kullanılıyor.
- Renk tanımlayıcısı:
  - Doygunluğu ve parlaklığı yeterli pikseller ton × doygunluğa göre 36 kutuya ayrılıyor.
  - Siyah, gri ve beyaz pikseller (tonun anlamsız olduğu renkler) parlaklığa göre 4 kutuya ayrılıyor. Böylece siyah pantolon ile lacivert pantolon ayrılabiliyor.
  - Komşu ton kutuları yumuşatılıyor: iki kamera arasındaki küçük renk kaymaları tolere ediliyor.
  - Histogramın karekökü alınıyor; böylece iki tanımlayıcının nokta çarpımı **Bhattacharyya benzerliği** oluyor (0–1).
- **Görünmeyen bölgeler:** Kutu çerçevenin üstüne veya altına değiyorsa kişi kesilmiş demektir. Tam boy, kutu genişliğinden tahmin ediliyor ve büyük kısmı görünmeyen bölgeler hesaba katılmıyor.
- **Çakışma:** Kişi başka biriyle üst üste biniyorsa renkler karışacağı için renk ipuçları kullanılmıyor ve öğrenilmiyor.
- **Maliyet:** Birkaç histogram hesabı; ölçülebilir bir hız kaybı yok (replay süresi 118 sn'ye karşı 119 sn).

### 5.2 İki aşamalı eşleştirme (`src/tracking/identity_matcher.py`, yeni)
1. **Aday listesi:** OSNet ile en benzer 5 kişi seçiliyor. Her kişinin skoru, galerisindeki en benzer vektörün skoru.
2. **Yeniden puanlama:** Her aday için birleşik skor hesaplanıyor: `0.5 × OSNet + 0.2 × üst beden + 0.3 × alt beden`. Görünmeyen bölgenin ağırlığı kalanlara dağıtılıyor.
3. **Veto:** Üst ve alt beden renklerinin ikisi de açıkça farklıysa (benzerlik < 0.35) aday reddediliyor.

OSNet'in ilk elemeyi yapması bilinçli bir tercih: aynı renkte giyinmiş ama tamamen farklı görünen bir yabancı, sadece renk benzerliğiyle eşleşemiyor.

Ayrıca eski "en yakın 5 vektörden en çok oyu alan kazanır" kuralı yerine "her kişinin en iyi skoru" kullanılıyor. Eski kural, galerisinde çok vektör olan kişileri kayırıyordu.

### 5.3 Renk galerisi (`person_manager.py`)
- Her kişi için son 30 renk gözlemi tutuluyor. Kişi kameradan kameraya geçtikçe farklı ışık koşullarındaki görünümü öğreniliyor.
- Karşılaştırmada her bölge için en iyi eşleşen gözlem kullanılıyor.

### 5.4 Yeni eşikler (birleşik skor ölçeğinde)
- Yeni gelen kişi (tek kare): **0.84** (eski: OSNet 0.88)
- Kayıt sırasındaki son kontrol (yaklaşık 1.7 sn biriken kanıt): **0.82** (eski: OSNet 0.85)

İkisi de bu videoda yaklaşık %0.5–1 yanlış eşleşme oranına denk geliyor.

### 5.5 Hata düzeltmesi: birbirinden farklı olduğu kanıtlanmış kişiler birleştirilmiyor (`identity_verifier.py`)
Eski otomatik birleştirme mantığı, aynı kamerada **aynı anda** görünen iki kimliği de birleştirebiliyordu. Oysa aynı karede aynı anda görünen iki kişi fiziksel olarak farklı kişilerdir. Test videosunda aynı kıyafetli iki farklı adam bu yüzden yanlışlıkla birleştirilmişti.

Artık birlikte görülen kimlik çiftleri bir **"birleştirilemez" listesinde** tutuluyor ve hiçbir zaman birleştirilmiyor. Birleştirme olduğunda liste de güncelleniyor.

### 5.6 Homografi kamera dosyasına taşındı
İki kamera arasındaki zemin eşleme matrisi artık kamera dosyasında (`config/cameras.epfl-room.yaml`). Bir kurulumun matrisi başka bir kurulumda yanlışlıkla kullanılamaz.

Mustafa'nın config'teki matrislerinden hangisinin oda videosuna ait olduğu ölçülerek bulundu:

| Matris | İzdüşürülen noktaların gerçek ayağa ≤30 px uzaklıkta olma oranı |
|---|---|
| Etkin olan (etiketsiz) | **%61** (doğru matris) |
| "room" yorumlu | %9 |
| "campus" yorumlu | %0 |

`tools/calibrate_homography.py` artık çıktıyı YAML formatında veriyor.

### 5.7 Tekrarlanabilir test aracı (`tools/replay.py`, yeni)
- Video dosyalarını kare kare ve her zaman aynı şekilde işliyor.
- Koddaki tüm zamanlama kuralları **videonun kendi saatine** bağlanıyor. Bu yüzden sonuç bilgisayarın o anki hızına bağlı değil; aynı ayarla iki çalıştırma birebir aynı sonucu veriyor (doğrulandı).
- `--mode osnet` ile Faz 0 kuralı, `--mode multicue` ile Faz 1 kuralı çalıştırılıp karşılaştırılabiliyor.
- `--snapshots klasör` ile her kimlik olayının (yeni kişi, tanıma, birleştirme...) ekran görüntüsü kaydediliyor.

## 6. Sistem düzeyinde sonuç (EPFL oda videosu, ilk 120 saniye, 2 kamera, 4 kişi)

| | Faz 0 kuralı | Faz 1 |
|---|---|---|
| Oluşturulan kimlik | 6 | **5** |
| Geri gelen kişiyi tanıma (ReID + kayıt sırasında eşleşme) | 3 | **15** |
| Otomatik birleştirme | 2 | 1 |
| İşleme hızı | aynı | aynı |

## 7. Dürüst değerlendirme ve sınırlar

**Ekran görüntüleri tek tek incelendi:**
- Kişiler farklı giyindiğinde kameralar arası tanıma tutarlı. Örneğin uzun saçlı kadın ve kahverengi kazaklı adam iki kamerada da doğru numarayı aldı.
- **Bu test videosu zor bir durum:** 4 kişiden 3'ü neredeyse aynı giyinmiş (koyu üst, bej pantolon) ve sürekli birbirinin önünden geçiyorlar. Bu anlarda tracker kişileri karıştırıyor ve numaralar kişiler arasında kayabiliyor. Renk ipuçları aynı kıyafeti doğal olarak ayırt edemiyor. Gerçek hayatta bunun karşılığı **iş üniforması** olan ortamlar.
- Bu video için hazır doğru cevap (etiket) olmadığı için doğruluk yüzdesi (örneğin IDF1) verilemiyor. Yukarıdaki sayılar olay sayıları ve çift bazlı ölçümler.

**Sonraki adımlar:**
1. **Etiketli değerlendirme:**
   - Market-1501 ile kameralar arası Rank-1/mAP (veri seti indirilince, `tools/eval_reid.py`).
   - Kendi çekeceğimiz kısa ve etiketli bir test videosu: 2 oda, 3–4 kişi, farklı ve benzer kıyafetler.
2. **Faz 2, kamera topolojisi ve zaman:**
   - Ayrı alanları gören kameralarda "hangi kameradan hangisine kaç saniyede geçilebilir" kuralı.
   - Aynı kıyafetli kişileri ayırt etmenin en güçlü yolu, fiziksel olarak imkânsız eşleşmeleri elemek.
3. **Kalabalıkta takip:** Çakışma anlarında kimlik kaymasını azaltmak. Çakışan kişilerin numaralarını, ayrıldıktan sonra yeniden doğrulamak.
4. **Boy:** Kamera zemin kalibrasyonu ile gerçek boy tahmini. Faz 2'deki kamera kurulumuyla birlikte yapılacak.

**Not (hız):** Bu oturumdaki canlı testte sistem kamera başına hedeflenen 6 kare/sn'ye ulaştı. Faz 0 ölçümlerinde 2.5–4 kare/sn görülmüştü. Kodda hızı bu ölçüde etkileyecek bir değişiklik yapılmadı; farkın büyük kısmı muhtemelen laptopun güç durumundan (pil / priz) kaynaklanıyor. Kesin performans karşılaştırması için ölçümler aynı güç koşullarında tekrarlanmalı.

## 8. Nasıl denenir

```bash
# Canlı (laptop kamerası)
.venv\Scripts\python.exe src\__main__.py

# Faz 0 ve Faz 1 karşılaştırması (aynı video, tekrarlanabilir)
.venv\Scripts\python.exe tools\replay.py --cameras config\cameras.epfl-room.yaml --mode osnet
.venv\Scripts\python.exe tools\replay.py --cameras config\cameras.epfl-room.yaml --mode multicue --snapshots snaps

# Loglarda bir tanıma kararının gerekçesi şöyle görünür:
# ReID match: T4 -> P2 fused=0.90 (reid=0.82; upper=0.98, lower=0.96)
```
