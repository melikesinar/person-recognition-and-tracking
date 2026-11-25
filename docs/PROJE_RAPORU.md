# Proje Raporu: Anonim Çok Kameralı Kişi Takibi, Baştan Sona

**Tarih:** 18.11.2026 · **Kapsam:** Mustafa'nın projesi (başlangıç) → Faz 0 → Faz 1 → Faz 2 → Faz 3 (devam ediyor)

Bu rapor projeyi hiç bilmeyen birinin baştan sona anlayabilmesi için yazıldı. Her kavram ilk geçtiği yerde açıklanıyor. Kod referansları `dosya:satır` biçiminde. VS Code'da `Ctrl+P` ile dosyayı açıp `:satır` yazarak o satıra gidebilirsin. Satır numaraları `feature/faz3-eval-consolidation` branch'indeki koda göre.

**İçindekiler**
1. Proje ne yapıyor, neden zor?
2. Temel kavramlar (sözlük)
3. Kullanılan teknolojiler
4. Başlangıç: Mustafa'nın kurduğu sistem
5. Faz 0: Altyapı, hata düzeltmeleri, CPU hızlandırma
6. Faz 1: Çok ipuçlu tanıma (renk + OSNet)
7. Faz 2: Fiziksel kurallar, "emin değilim", rota çıktısı
8. Faz 3: Gerçek etiketli veriyle ölçüm ve kimlik toparlama (devam ediyor)
9. Bütün sonuçlar tek tabloda
10. Bir karenin sistemdeki yolculuğu (kodla birlikte)
11. Dürüst değerlendirme: ne biliyoruz, ne bilmiyoruz
12. Sıradaki adımlar
13. Komutlar

---

## 1. Proje ne yapıyor, neden zor?

### 1.1 Amaç
Bir binada (ofis, market, AVM) birden fazla kamera var. Sistem şunu yapmalı:

> Bir kişi Kamera A'dan çıkıp Kamera B'de göründüğünde, **aynı kişi olduğunu anlamak**. Böylece her kişinin **rotası** (kapı → koridor → ofis) ve her yerde **ne kadar kaldığı** çıkarılabilsin.

Örnek çıktı (sistemin gerçekten ürettiği format):
```
P1: kapi 8s -> koridor 41s -> ofis 312s
```

### 1.2 Kısıtlar
| Kısıt | Anlamı |
|---|---|
| **Anonim** | Sistem kimin kim olduğunu bilmez, isim kullanmaz. Herkes geçici bir numara alır: P1, P2, ... |
| **Kayıt yok (open-set)** | Kimsenin fotoğrafı önceden sisteme yüklenmez. Sistem kişiyi ilk gördüğünde tanımaya başlar. |
| **N kamera** | Kamera sayısı belli değil. Önce 3 kamera, sonra daha fazla. |
| **Kameralar farklı yerleri görüyor** | Kapı kamerası ile ofis kamerası aynı alanı görmüyor. Kişi bir süre hiçbir kamerada görünmüyor. |
| **Yüz görünmeyebilir** | Kamera tavanda, kişi arkasını dönmüş olabilir. Bu yüzden kıyafet, renk ve vücut görünümü kullanılıyor. |
| **Sadece CPU** | Ekran kartı (GPU) yok. Her şey işlemcide çalışmalı. |

### 1.3 Neden zor?
- Aynı kişi farklı kamerada **farklı görünür**: ışık farklı, açı farklı (önden / arkadan), uzaklık farklı.
- Farklı kişiler **aynı görünebilir**: aynı renk mont, iş üniforması.
- Kişiler birbirinin önünden geçer, kutular karışır.
- Yanlış eşleşmenin bedeli büyük: iki kişinin rotası birbirine karışır ve bu sonradan fark edilmez.

---

## 2. Temel kavramlar (sözlük)

| Terim | Basit açıklama |
|---|---|
| **Tespit (detection)** | "Bu karede insan nerede?" sorusunun cevabı. Her insan için bir kutu: `[x1, y1, x2, y2]` ve bir güven skoru (0–1). |
| **Takip (tracking)** | "Bu karedeki kutu, önceki karedeki hangi kutunun devamı?" Her kişiye kamera içinde geçici bir **track ID** verir (T5, T12). Kişi kadrajdan çıkınca bu numara kaybolur. |
| **P-ID (kalıcı kimlik)** | Sistemin kişiye verdiği numara (P1, P2). Bütün kameralarda aynı kalmalı. **Projenin asıl işi track ID'leri doğru P-ID'lere bağlamak.** |
| **ReID (yeniden tanıma)** | Bir kişinin görüntüsünü bir sayı dizisine (vektöre) çevirip başka görüntülerle karşılaştırmak. |
| **Vektör / embedding** | Görüntünün 512 sayıdan oluşan "parmak izi". Aynı kişinin farklı fotoğrafları yakın vektörler verir, farklı kişilerinki uzak. |
| **Kosinüs benzerliği** | İki vektörün ne kadar aynı yöne baktığı. 1.0 = aynı, 0 = alakasız. Vektörler uzunluğu 1 olacak şekilde normalize edildiği için, iki vektörün çarpımı (nokta çarpımı) doğrudan bu benzerliği verir. |
| **Galeri** | Bir kişi için saklanan vektörler (önden, arkadan, yandan...). Kişi başına en fazla 60. |
| **Eşik (threshold)** | "Benzerlik bundan büyükse aynı kişi say" sınırı. Yüksek eşik = az yanlış eşleşme ama çok kaçırma. Düşük eşik = tersi. |
| **Homografi** | Düz zemin için iki kamera görüntüsü arasında nokta taşıyan 3×3'lük matris. "Kamera 0'da şu pikselde duran ayak, Kamera 1'de şu piksele denk gelir." Sadece **aynı alanı gören** kameralarda işe yarar. |
| **Topoloji** | Kameraların birbirine göre yerleşimi: hangileri aynı alanı görüyor, hangileri arasında kaç saniyelik yürüme yolu var. |
| **Parçalanma (fragmentation)** | Bir kişiye birden fazla numara verilmesi (Ali'ye hem P3 hem P7). Kötü ama sonradan düzeltilebilir. |
| **Karışma (mixing / ID switch)** | Bir numaranın farklı kişilere verilmesi (P3 önce Ali, sonra Veli). Çok kötü, çünkü sessizce yanlış bilgi üretir. |
| **IDF1 / IDP / IDR** | Etiketli veriyle ölçülen kimlik doğruluğu (bölüm 8'de ayrıntılı). |

---

## 3. Kullanılan teknolojiler

| Teknoloji | Ne için | Nerede |
|---|---|---|
| **Python 3.12** | Bütün proje | — |
| **OpenCV (`cv2`)** | Kamera/video okuma, görüntü işleme, çizim, homografi | her yerde |
| **YOLOv8n** (Ultralytics) | İnsan tespiti. "n" = nano, en küçük ve en hızlı sürüm. | `src/core/detector.py` |
| **BoostTrack** (BoxMOT kütüphanesi) | Kamera içi takip. Kalman filtresi (hareket tahmini) + IoU (kutu örtüşmesi). | `src/core/camera_node.py:54` |
| **OSNet x1_0** (MSMT17 ile eğitilmiş) | Kişi görüntüsünden 512 boyutlu görünüm vektörü çıkarma | `src/core/reid_extractor.py` |
| **PyTorch + TorchScript** | OSNet'i CPU'da çalıştırma ve hızlandırma | `reid_extractor.py:55` |
| **NumPy** | Vektör hesapları, benzerlik matrisleri | her yerde |
| **SciPy** (`linear_sum_assignment`) | Macar algoritması: ölçümde kutuları ve kimlikleri en iyi şekilde eşleştirme | `tools/id_metrics.py` |
| **HSV renk histogramı** (Faz 1, kendi yazdığımız) | Kafa / üst beden / alt beden renk ipucu | `src/core/appearance_cues.py` |
| **MongoDB** (isteğe bağlı) | Kimlikleri kalıcı saklama | `src/database/mongo_handler.py` |
| **YAML** | Kamera listesi, topoloji, homografi ayarları | `config/cameras*.yaml` |
| **multiprocessing / threading** | YOLO ayrı process'te; her kamera ayrı thread'de | `__main__.py`, `camera_stream.py` |
| **Git / GitHub** | Sürüm takibi; her faz ayrı branch ve PR | — |

**Test verileri:**
- **EPFL "room"** videosu: 2 kamera aynı odayı görüyor, 4 kişi, düşük çözünürlük (360×288). Etiketi yok.
- **WILDTRACK** (EPFL): 7 HD kamera aynı meydanı görüyor, her kişi her karede etiketli. Faz 3'te kullanılıyor.
- **Market-1501**: ReID değerlendirme aracı hazır (`tools/eval_reid.py`) ama veri seti henüz indirilmedi, ölçüm yapılmadı.

---

## 4. Başlangıç: Mustafa'nın kurduğu sistem

Projeyi önceki stajyer Mustafa Aykut başlattı (MIT lisansı). Sağlam bir iskelet kurmuştu. Bu bölümde onun yapısını ve iyi fikirlerini anlatıyorum; sonraki fazlarda neyi neden değiştirdiğimiz ancak bununla anlaşılır.

### 4.1 Mimari
```
Kameralar ──> Ana döngü (sırayla cap.read()) ──> [kuyruk] ──> YOLO process'i
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    ▼
         Her kamera için CameraNode: takip → ReID → kimlik kararı → çizim
                    │
                    ▼
         Ortak hafıza: PersonManager (galeriler) + FAISS (arama) + MongoDB (kayıt)
```

### 4.2 Mustafa'nın iyi kararları (korunanlar)
1. **YOLO ayrı bir process'te.** Python'da thread'ler aynı anda hesap yapamaz (GIL kilidi). Process'ler yapabilir. Böylece tespit, kimlik işlemleriyle paralel yürüyor.
2. **Kare ile tespit sonucu `frame_id` ile eşleştiriliyor.** Sonuç yanlış kareye çizilmiyor.
3. **Çok vektörlü galeri** (`person_manager.py`). Kişi önden, arkadan ve yandan farklı görünür; tek vektör yetmez. Her kişi için 60'a kadar vektör tutuluyor:
   - Yeni vektör galeridekilerden birine 0.90'dan fazla benziyorsa eklenmiyor (o poz zaten var).
   - Galeri dolunca birbirine en çok benzeyen iki vektörün ortalaması alınıp teke indiriliyor. Yer açılıyor ama çeşitlilik korunuyor. Akıllıca bir yöntem.
4. **Kalite kontrolü** (`quality_checker`). Bulanık, kenarda kesilmiş, çok küçük veya karanlık kırpıntılar ReID'e gönderilmiyor. Kötü görüntü galeriyi bozar.
5. **CLAHE** kontrast dengeleme: karanlık/parlak kamerada detayları ortaya çıkarır.
6. **Görsel koruma (Visual Guard).** Kimliği olan bir kişinin yeni vektörü galerisine 0.60'tan az benziyorsa numara hemen geri alınıyor. İki kişi yan yana geçerken tracker kutuları karıştırırsa ("ID swap"), A'nın numarası B'ye geçmesin diye.
7. **Kimlik doğrulayıcı** (`identity_verifier.py`):
   - Yanlış numarayı düşürmek veya değiştirmek için **3 kez üst üste onay** isteniyor. Tek bir kötü kare numarayı bozmasın.
   - Değişiklikten sonra 5 saniye kilit: numara iki kişi arasında gidip gelmesin.
   - Tekrar eden kimlikleri birleştirme (2 saniyede bir).
8. **Homografi kalibrasyon aracı** (`tools/calibrate_homography.py`): iki kamerada zemindeki aynı 4+ noktaya tıklayınca matrisi hesaplıyor.
9. **Asenkron veritabanı yazıcı:** MongoDB'ye yazma ayrı thread'de. Disk beklemesi görüntüyü dondurmuyor.

### 4.3 Kimlik kararı (Mustafa'nın akışı, bugün de omurga bu)
Kimliği olmayan her track için sırayla:
1. **Konum hafızası:** Son 30 saniyede bu kamerada kaybolan biri, aynı yerin yakınında belirdiyse ona bağla.
2. **ReID araması:** Vektörü bütün galerilerde ara. Eşik 0.88.
3. **Yeni kayıt:** Eşleşme yoksa vektörleri biriktir; yeterince örnek olunca ortalamayla son bir arama yap ("Final Safety Check"), yine yoksa yeni P-ID ver.

### 4.4 İncelemede bulunan sorunlar
| # | Sorun | Etkisi |
|---|---|---|
| 1 | ReID modeline görüntü **BGR** renk sırasıyla veriliyordu, model RGB ile eğitilmiş | Kırmızı ↔ mavi ters; doğruluk düşer |
| 2 | Her kamera OSNet'in **kendi kopyasını** yüklüyordu | 2 kamerada 3 kopya; bellek ve süre israfı |
| 3 | Her kişinin vektörü her karede **iki kez** hesaplanıyordu | En pahalı işlem iki katı |
| 4 | FAISS indeksi MongoDB'den yeniden kuruluyordu; yazmalar gecikmeliydi | Yeni kişi ~30 sn aramada görünmüyor → aynı kişi iki kez kaydediliyor |
| 5 | MongoDB zorunluydu | Veritabanı yoksa sistem açılmıyor |
| 6 | Kameralar sırayla okunuyordu; kamera listesi kodun içindeydi | Yavaş kamera herkesi bekletiyor; kamera eklemek için kod değişiyor |
| 7 | Tracker eşiği (0.60) sistem eşiğinden (0.55) yüksekti | 0.55–0.60 arası tespitler sessizce atılıyor |
| 8 | Sabit kamerada kamera hareketi telafisi (ECC) açıktı | Boşa CPU |
| 9 | ReID eşiği 0.88 çok yüksekti (Faz 1'de ölçüldü) | Geri gelen kişilerin sadece ~%2'si tanınabiliyordu |
| 10 | Konum hafızası görünüşe hiç bakmıyordu (Faz 2'de ölçüldü) | Kalabalıkta numaralar yanlış kişilere taşınıyordu |

**Ölçülen başlangıç hızı:** EPFL, 2 kamera, CPU. 60 saniyede kamera başına sadece **52 ve 33 kare** işlendi (saniyede ~0.5–0.9 kare). Bu sürede hiç kimlik oluşmadı.

---

## 5. Faz 0: Altyapı, hata düzeltmeleri, CPU hızlandırma

**Hedef:** Sistemi doğru ve kullanılabilir hızda çalıştırmak; kamera sayısından bağımsız yapmak.

### 5.1 Kamera sayısından bağımsız mimari
- Kameralar `config/cameras.yaml` dosyasından okunuyor. Kamera eklemek = bir satır yazmak. Kaynak webcam, video dosyası veya RTSP/IP kamera olabilir.
- **Her kamera kendi thread'inde** okunuyor ve **sadece en yeni kare** tutuluyor (`src/core/camera_stream.py:54`):
  ```python
  def _reader(self):                 # ayrı thread'de sürekli çalışır
      ret, frame = self.cap.read()
      with self._lock:
          self._frame = frame        # eski kare atılır, en yenisi kalır
  ```
  Neden: `cap.read()` yeni kare gelene kadar bekler. Sırayla okunduğunda yavaş kamera herkesi bekletiyordu. Canlı kamerada eski kareyi işlemenin anlamı da yok; gecikme birikmesin.
- **Kare gönderimi ayrı thread** (`__main__.py:214` `_feed_detector`): Önceden ReID hesaplanırken dedektöre kare gitmiyordu. Dedektör doluluğu %59–92'den %99'a çıktı.
- Kamera başına **işleme hızı** (`process_fps`, varsayılan 6). Yürüyen insanı takip etmek için saniyede 30 kare gerekmiyor.

### 5.2 Renk sırası hatası (en önemli doğruluk düzeltmesi)
`src/core/reid_extractor.py:93`:
```python
return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
OpenCV görüntüyü Mavi-Yeşil-Kırmızı sırasıyla okur, OSNet Kırmızı-Yeşil-Mavi bekler. Eski kodda model kırmızı tişörtü mavi görüyordu.

### 5.3 Tek model, ihtiyaç halinde hesaplama
- OSNet sistemde **bir kez** yükleniyor, bütün kameralar paylaşıyor.
- Tracker artık ReID kullanmıyor (`camera_node.py:69`, `with_reid=False`). Kare kare takip Kalman + IoU ile yapılıyor; sabit kamerada yeterli.
- ReID sadece gerektiğinde (`camera_node.py:78` `_needs_embedding`):
  - Kimliği olmayan kişi: her karede (hızlı karar için).
  - Kimliği olan kişi: saniyede bir (galeri güncelleme ve kontrol için).
- `use_ecc=False` (`camera_node.py:75`), tracker eşiği sistem eşiğine eşitlendi.

### 5.4 CPU hızlandırmaları (her biri ölçülerek, çıktı değişmeden)
| Değişiklik | Kod | Önce | Sonra |
|---|---|---|---|
| Denormal sayıları sıfırlama | `utils/torch_utils.py:17` `torch.set_flush_denormal(True)` | 1452 ms / 4 kişi | 351 ms / 4 kişi |
| TorchScript ile modeli dondurma | `reid_extractor.py:66-67` | 191 ms / kişi | 95 ms / kişi |
| YOLO giriş boyutu 640 → 416 | `config.py:235` | ~150 ms / kare | ~95–130 ms / kare |
| Thread sınırı 12 → 4 | `DETECTOR_NUM_THREADS` | — | küçük işlerde daha hızlı |

**Denormal nedir?** Sinir ağı hesap sırasında sıfıra *aşırı* yakın sayılar üretiyor (10⁻³⁸'den küçük). İşlemciler bu özel formatta 10–100 kat yavaş çalışıyor. Bunları sıfır saymak sonucu değiştirmiyor (ölçülen fark tam 0.0) ama 4 kat hız kazandırıyor. İpucu: 12 thread'in 1 thread'den yavaş çalışması garipti; sebebi buydu.

**TorchScript nedir?** OSNet çok sayıda küçük katmandan oluşuyor. Normal PyTorch'ta her katman için Python üzerinden ayrı çağrı yapılıyor ve bu ek yük asıl hesaptan uzun sürüyordu. `torch.jit.trace` modeli bir kez çalıştırıp işlem grafiğini kaydediyor, `freeze` + `optimize_for_inference` katmanları birleştiriyor:
```python
traced = torch.jit.trace(model, example)
optimized = torch.jit.optimize_for_inference(torch.jit.freeze(traced.eval()))
```
Çıktı aynı (kosinüs benzerliği 1.0).

### 5.5 FAISS yerine anında güncellenen arama (`src/database/vector_index.py`)
- Her kişinin galerisi değiştiği anda indekse yazılıyor (`set_person`, satır 41). Yarış durumu yok; yeni kişi hemen aranabiliyor.
- Arama tek bir matris çarpımı: `sims = matrix @ query`. 300 kişi × 60 vektör için birkaç milisaniye. FAISS'in avantajı milyonlarca vektörde ortaya çıkar; ofis ölçeğinde gerekmiyor.
- Sonuç tam (FAISS HNSW yaklaşık sonuç verir).
- MongoDB isteğe bağlı oldu; yoksa sistem bellekte çalışıyor.

### 5.6 Faz 0 sonucu (EPFL, 2 kamera, 4 kişi, aynı laptop)
| | Önce | Sonra |
|---|---|---|
| İşlenen kare (kamera başına, dakikada) | 52 / 33 | ~237 / ~181 |
| Gerçek işleme hızı | ~0.5–0.9 kare/sn | ~3–4 kare/sn |
| Oluşan kimlik | 0 | 6 (gerçek: 4) |
| OSNet kopyası | 3 | 1 |
| MongoDB olmadan çalışma | Hayır | Evet |

**Toplamda ~4–5 kat hızlanma.** Kalan sorun: 4 kişiye 6 kimlik.

---

## 6. Faz 1: Çok ipuçlu tanıma (renk + OSNet)

**Sorun:** Tek bir OSNet vektörü yetmiyordu. Geri gelen kişiler tanınmıyor, yeni numara alıyordu.

### 6.1 Önce ölçtük: hangi ipucu ne kadar işe yarıyor?
Etiket olmadan bir ölçüm kurduk: tracker'ın aynı kişi diye takip ettiği görüntüler (en az 2 sn arayla) = "aynı kişi"; aynı karede görünen farklı kişiler = "farklı kişi". ~6.800 aynı + ~2.200 farklı çift.

**AUC** = rastgele bir "aynı" çiftin, rastgele bir "farklı" çiftten daha yüksek puan alma olasılığı. 1.0 mükemmel, 0.5 yazı-tura.

| İpucu | Kamera 0 | Kamera 1 |
|---|---|---|
| OSNet | 0.825 | 0.777 |
| **Alt beden rengi** | **0.812** | **0.780** |
| Üst beden rengi | 0.695 | 0.719 |
| Kafa/saç | 0.574 | 0.519 |

Şaşırtıcı bulgu: **basit bir pantolon rengi histogramı, derin öğrenme modeli kadar ayırt edici.** Kafa bu çözünürlükte (10–15 piksel) neredeyse yazı-tura; ağırlığı 0 yapıldı.

**İkinci bulgu, eski eşik neden çalışmıyordu:** Bu videoda aynı kişinin OSNet benzerliği ortalama 0.72–0.76. Sistemin eşiği 0.88. Sonuç: geri gelen kişilerin sadece **%1.8**'i tanınabiliyordu.

### 6.2 Renk ipucu nasıl çıkarılıyor? (`src/core/appearance_cues.py`)
1. Kişi kutusu sabit oranlarla bölgelere ayrılıyor (poz modeli yok, CPU'ya yük yok):
   ```python
   PART_RANGES = {"head": (0.00, 0.17), "upper": (0.20, 0.52), "lower": (0.55, 0.90)}
   ```
   Ayakkabı ve zemin dışarıda; yanlardaki arka plan da kırpılıyor.
2. Her bölge HSV renk uzayına çevriliyor (`_describe`, satır 69):
   - Renkli pikseller: ton (12 kutu) × doygunluk (3 kutu) = 36 kutu.
   - Siyah/gri/beyaz pikseller: sadece parlaklık (4 kutu). Siyah ile lacivert pantolon böylece ayrılıyor.
   - Komşu ton kutuları yumuşatılıyor (iki kamera arasında küçük renk kaymaları tolere edilsin).
   - Histogramın **karekökü** alınıyor (satır 94). Böylece iki histogramın nokta çarpımı **Bhattacharyya benzerliği** oluyor (0–1).
3. Kutu çerçeveye değiyorsa kişi kesilmiştir; görünmeyen bölgeler hesaba katılmıyor (satır 97 `extract_part_cues`).
4. Kişi başkasıyla üst üste biniyorsa renkler karışık olduğu için renk kullanılmıyor.

### 6.3 Puanları birleştirme (`appearance_cues.py:150` `fuse_scores`)
```python
CUE_WEIGHTS = {"reid": 0.50, "upper": 0.20, "lower": 0.30, "head": 0.0}   # config.py:176
fused = (0.5*osnet + 0.2*üst + 0.3*alt) / (görünen ağırlıkların toplamı)
vetoed = üst < 0.35 ve alt < 0.35                                          # satır 171
```
- Görünmeyen bölge hesaptan düşüyor, ağırlıklar yeniden ölçekleniyor.
- **Veto:** Hem üst hem alt renk açıkça farklıysa, OSNet ne derse desin eşleşme reddediliyor.

### 6.4 İki aşamalı eşleştirme (`src/tracking/identity_matcher.py:80`)
1. **Aday listesi:** OSNet ile en benzer 5 kişi (`vector_index.py:85` `top_persons`). Her kişinin puanı galerisindeki en iyi vektör:
   ```python
   person_sims = np.maximum.reduceat(sims, starts)   # vector_index.py:97
   ```
2. **Yeniden puanlama:** Her aday için birleşik puan.

Neden önce OSNet? Aynı renkte giyinmiş ama tamamen farklı görünen bir yabancı, sadece renkle eşleşemesin.

Ayrıca Mustafa'nın "en yakın 5 vektörden en çok oyu alan kazanır" kuralı bırakıldı; galerisi büyük kişileri kayırıyordu.

### 6.5 Yeni eşikler (birleşik puan ölçeğinde)
- Yeni gelen kişi (tek kare): **0.84** (`REENTRY_SIMILARITY_THRESHOLD`, `config.py:37`)
- Kayıt sırasındaki son kontrol (~1.7 sn biriken kanıt): **0.82** (`config.py:33`)

### 6.6 Diğer Faz 1 işleri
- **Birlikte görülen kişiler birleştirilmiyor:** Aynı karede yan yana görünen iki kişi kesinlikle farklıdır. Eski otomatik birleştirme bunu bilmiyordu ve aynı kıyafetli iki adamı birleştirmişti.
- **Homografi kamera dosyasına taşındı;** Mustafa'nın 3 matrisinden hangisinin oda videosuna ait olduğu ölçülerek bulundu (%61 isabet; diğerleri %9 ve %0).
- **`tools/replay.py`:** Videoyu her seferinde aynı şekilde işleyen tekrarlanabilir test aracı. Bütün zamanlama kuralları videonun kendi saatine bağlanıyor; sonuç bilgisayarın hızına bağlı değil.

### 6.7 Faz 1 sonucu
Aynı yanlış eşleşme oranında doğru tanıma (tek görüntü vs tek görüntü):

| Yanlış eşleşme oranı | Sadece OSNet | OSNet + renk |
|---|---|---|
| %0.5 | %14.4 | **%41.6** |
| %1 | %19.9 | **%46.7** |
| %2 | %27.6 | **%56.4** |

**Aynı güvenlik seviyesinde 2–3 kat daha fazla doğru tanıma.** AUC 0.787 → 0.851.

Sistem düzeyinde (EPFL, 120 sn): kimlik 6 → **5**, geri gelen kişiyi tanıma 3 → **15** olay. Hız aynı.

**Görülen sınır:** EPFL'de 4 kişiden 3'ü aynı giyinmiş (koyu üst, bej pantolon). Renk bunları ayıramıyor; numaralar karışıyordu. Senin testindeki gözlemin de buydu. → Faz 2.

---

## 7. Faz 2: Fiziksel kurallar, "emin değilim", rota çıktısı

**Sorun:** Görünüş, aynı giyinmiş kişileri ayırmaya yetmiyor. Farklı kişiler arasında bile birleşik benzerlik 0.93–0.98 çıkıyordu.

**Fikir:** Görünüşün yetmediği yerde fizik kullan. Bir insan aynı anda iki yerde olamaz ve yürüme süresinden hızlı yer değiştiremez.

### 7.1 Etiketsiz bir doğruluk ölçütü kurduk (`tools/eval_consistency.py`)
EPFL'deki iki kamera aynı odayı görüyor. Homografi ile Kamera 0'daki ayak noktasını Kamera 1'e taşıyıp şu soruyu soruyoruz: **o noktada duran kişi, iki kamerada da aynı P numarasını taşıyor mu?** Taşıyorsa "tutarlı".

Faz 1'in durumu: toplam **%35**. Odada 2 kişi varken %100, 3. ve 4. kişi girince çöküyor.

### 7.2 Asıl hata kaynağını bulduk: konum hafızası
Her değişikliği tek tek açıp kapatıp ölçtük. Asıl suçlu Mustafa'nın **konum hafızası** kuralıydı: kaybolan kişinin yerinde 30 saniye içinde beliren **herkes**, görünüşüne bakılmadan onun numarasını alıyordu. Küçük bir odada 4 kişi dolaşırken bu sürekli oluyor.

| Konum hafızası ayarı | Tutarlılık |
|---|---|
| Eski hâli (görünüşe bakmadan, 30 sn) | %18 |
| Görünüş doğrulamalı | %57 |
| Görünüş doğrulamalı, 2 sn ve dar alan | %69 |
| **Kapalı** | **%78** |

Karar: kapatıldı (`SPATIAL_REACQ_ENABLED = False`, `config.py:208`; kontrol `camera_node.py:307`). Ayardan açılabilir; seyrek ortamlarda faydalı olabilir.

### 7.3 Kamera topolojisi (`src/tracking/topology.py`)
Kamera dosyasına yazılıyor:
```yaml
topology:
  overlapping: [[3, 4]]                                    # aynı alanı gören kameralar
  transitions:                                             # yürüme yolları (iki yönlü)
    - {cameras: [0, 1], min_seconds: 3, max_seconds: 60}   # kapı <-> koridor
    - {cameras: [1, 2], min_seconds: 5, max_seconds: 120}  # koridor <-> ofis
```
Doğrudan bağlı olmayan kameralar için en kısa yol hesaplanıyor (Floyd-Warshall algoritması): kapı → ofis = 3 + 5 = 8 sn.

Üç kesin kural (`topology.py:131` `impossible_reason`):
1. **Aynı anda iki yer olmaz:** Aynı alanı görmeyen iki kamerada 1 saniye içinde görülen kişi aynı kişi olamaz.
2. **Yürüme süresi:** Kapıdan 2 saniye önce çıkan, ofiste beliremez (en az 8 sn gerekir).
3. **Aynı nokta:** Aynı alanı gören kameralarda, P2'nin Kamera 0'daki ayağı Kamera 1'e taşındığında, oradaki kişinin ayağına 60 pikselden uzaksa o kişi P2 olamaz.

Kodda:
```python
gap = t - seen
if gap < self.simultaneous_window:
    return f"visible in Cam {cam} at the same time"
need = self.min_travel(cam, camera_id)
if gap < need:
    return f"left Cam {cam} {gap:.1f}s ago, walking to Cam {camera_id} takes >= {need:.0f}s"
```

**Neden puan değil de kesin kural?** Bunlar fiziksel imkânsızlıklar; görünüş ne kadar benzer olursa olsun geçerli. Aynı üniformalı iki çalışanı ayırmanın en güvenilir yolu.

**`max_seconds` neden kural değil?** Kişi yolda kahve almak için durmuş olabilir. Uzun süre "farklı kişi" kanıtı değildir; sadece raporlamada kullanılıyor.

### 7.4 "Emin değilim" kuralı (`identity_matcher.py`)
Önce: "En yüksek puanlı aday eşiği geçtiyse o kişidir."
Şimdi kazanan **hem eşiği geçmeli hem de açıkça önde olmalı:**
- İkinci aday en az **0.03** geride olmalı (`AMBIGUITY_MARGIN`, `config.py:190`).
- İmkânsız bir aday (örneğin şu an başka kamerada olan biri) ondan daha benzer olmamalı.

Sağlanmazsa karar verilmiyor. Kişi ekranda **`T12 (?)`** olarak görünüyor, sistem en fazla 6 sn daha kanıt topluyor (`camera_node.py:388`). Log:
```
[Cam 0] NOT SURE: T16 P3 0.95 vs P5 0.93 (possible). Waiting.
```
**Neden?** Yanlış kimlik iki kişinin rotasını birden bozar. Birkaç saniyelik "bilinmiyor" sadece küçük bir boşluk bırakır.

### 7.5 "Yan yana görüldü" kanıtı (`camera_node.py:185`)
```python
self.seen_with.setdefault(t[1], set()).update(p for p in used_pids if p != own)
```
Bir track'in yanında aynı karede görülen kişiler kaydediliyor; o track asla o kişilere dönüşemiyor. Log:
```
NEW PERSON: T10 -> P5, looks like P3 0.96 but seen next to this person in this camera
```

### 7.6 Rota ve kalış süresi (`src/tracking/presence.py`) — projenin asıl ürünü
Program kapanırken `output/routes_<tarih>.csv` yazılıyor (`__main__.py:359`):
```
person,camera_id,camera,enter,exit,dwell_seconds
P1,0,kapi,2026-11-17 09:02:11,2026-11-17 09:02:19,8.0
P1,1,koridor,2026-11-17 09:02:24,2026-11-17 09:03:05,41.0
```
Loga gözlenen geçiş süreleri de yazılıyor; topoloji süreleri bilinmiyorsa buradan okunabilir:
```
travel kapi -> koridor: n=14, min 3.2s, median 5.1s, max 22.0s
```

### 7.7 Faz 2 sonucu (EPFL, 150 sn, 2 kamera, 4 kişi)
| | Faz 1 | Faz 2 |
|---|---|---|
| Kameralar arası tutarlılık (sıkı) | %35 | **%85** |
| Kameralar arası tutarlılık (gevşek) | %41 | **%78** |
| Numara "ışınlanması" | 3 | **1** |
| Sona kalan kimlik (gerçek: 4) | **4** | 7 |
| Numara taşıyan kişi oranı | **%87** | %73 |

**Kazanç:** Sistem "bu P3" dediğinde artık büyük olasılıkla doğru (%35 → %85).
**Bedel:** Daha fazla parçalanma (4 kişiye 7 numara) ve daha uzun "bilinmiyor" süresi. **Bilinçli tercih:** parçalanmış ama doğru rota sonradan birleştirilebilir; karışmış rota sessizce yanlış bilgi üretir. → Faz 3'ün konusu bu parçalanmayı geri toplamak.

---

## 8. Faz 3: Gerçek etiketli veriyle ölçüm ve kimlik toparlama (devam ediyor)

**İki hedef:**
1. Şimdiye kadarki ölçümler etiketsizdi. **Gerçek doğru cevabı olan** bir veri setiyle, literatürdeki standart metrikle ölçmek.
2. Faz 2'nin bedeli olan **parçalanmayı geri toplamak**.

### 8.1 Standart metrik: IDF1 (`tools/id_metrics.py`)
Literatürdeki ölçüt (Ristani ve ark., 2016). İki adımda hesaplanıyor:
1. **Kutu eşleştirme:** Her karede, sistemin kutuları ile etiket kutuları bire bir eşleştiriliyor (Macar algoritması, `linear_sum_assignment`). Yeterince örtüşmeyen (IoU eşiğinin altında) çiftler sayılmıyor.
2. **Kimlik eşleştirme:** Her gerçek kişi en fazla bir P numarasıyla eşleştiriliyor. Seçim, "kaç karede birlikte göründüler" sayısını en büyütecek şekilde yapılıyor. Bir kutu, ancak P numarası o kişinin eşi olan numaraysa "doğru" (IDTP) sayılıyor.

| Metrik | Formül | Anlamı |
|---|---|---|
| **IDR** (kimlik geri çağırma) | IDTP / etiketli kutular | "Kişi göründüğü anların yüzde kaçında doğru numarayı taşıdı?" |
| **IDP** (kimlik kesinliği) | IDTP / numaralı kutular | "Sistem bir numara verdiğinde yüzde kaç doğruydu?" |
| **IDF1** | ikisinin dengeli ortalaması | Tek sayıda genel başarı |

- **Global:** Bütün kameralarda tek eşleştirme. Kişi kameradan kameraya aynı numarayı taşımalı. **Bizim asıl hedefimiz bu.**
- **Kamera içi:** Her kamera ayrı. Sadece takip kalitesi.
- **Parçalanma:** Bir kişiye ortalama kaç numara verildi (1.0 = hiç bölünmedi).
- **Karışma:** Bir numara ortalama kaç farklı kişiye verildi (1.0 = hiç karışmadı).

Formüller elle hesaplanmış örneklerle test edildi (`tests/test_id_metrics.py`, 4 test geçiyor).

### 8.2 Kimlik toparlama (`src/tracking/consolidator.py`)
Her 5 saniyede bütün kimlik çiftlerine şu soru soruluyor: "Bu ikisi aslında aynı kişi mi?"

**Neden ilk karardan daha iyi karar verebilir?** İlk karar 1–2 saniyelik kanıtla verildi. Şimdi her kimliğin galerisi büyüdü; çok daha fazla bilgi var.

**Puan: bütün galeri karşılaştırması** (`consolidator.py:74`):
```python
s = x @ y.T                                    # her vektör çifti
return 0.5 * (s.max(axis=1).mean() + s.max(axis=0).mean())
```
A'daki her vektörün B'deki en iyi eşi, ortalaması; ve tersi. Tek bir şanslı benzer kareye karşı sağlam. OSNet ve her renk bölgesi için ayrı hesaplanıp Faz 1 ağırlıklarıyla birleştiriliyor.

**Birleştirme kuralları** (`find_merges`, satır 95):
1. **Kanıtlanmış farklılar asla birleşmez.** Bu "birleştirilemez" liste her karede öğreniliyor (`presence.py:71` `note_simultaneous`):
   - aynı kamerada aynı anda görünen iki kimlik,
   - aynı alanı görmeyen iki kamerada aynı anda görünen iki kimlik,
   - aynı alanı gören kameralarda aynı anda ama zeminde açıkça farklı noktada duran iki kimlik.
2. Puan **0.80** üstünde olmalı (`CONSOLIDATE_MIN_SCORE`, `config.py:217`).
3. **Benzer kalibrasyonu** (satır 121): A'dan kesinlikle farklı olduğunu bildiğimiz biri A'ya, B kadar benziyorsa, bu benzerlik seviyesi "aynı kişi" anlamına gelmiyor demektir. Birleştirme yok.
4. **Rakip kuralı** (satır 124): B, A'dan farklı olduğu kanıtlanmış bir C'ye de aynı derecede benziyorsa, B'nin kime ait olduğu yazı-tura. Birleştirme yok.
5. Her turda en fazla 1 birleştirme; galeriler birleşince kalan çiftler bir sonraki turda yeniden değerlendiriliyor.

Birleştirme her yere uygulanıyor: galeriler, rota geçmişi, kameralardaki canlı track'ler, "yan yana görüldü" listeleri (`merge_everywhere`, satır 142).

Testler (`tests/test_faz3_consolidation.py`, 5 test geçiyor): parça birleşiyor; kanıtlanmış farklı birleşmiyor; "iki yerde aynı anda" öğreniliyor; benzer üçlüde yazı-tura birleştirme yapılmıyor; birleşince "birleştirilemez" bilgisi aktarılıyor.

**Bu testin yakaladığı gerçek bir hata:** İlk sürümde kural tek taraflıydı ve iki kişiye eşit benzeyen bir parçayı birleştiriyordu. Kural iki taraflı hâle getirildi.

EPFL'de toparlama açıkken: 1 birleştirme, tutarlılık değişmedi (gerileme yok), 7 → 6 kimlik.

### 8.3 Ölçüm altyapısı
- **`tools/offline_pipeline.py`:** Canlı sistemin aynısını video saatine bağlı, tekrarlanabilir şekilde çalıştırıyor. `replay.py` de bunun üstüne yeniden yazıldı (sonuçlar birebir aynı çıktı: 8 oluşturulan / 7 yaşayan kimlik, 142 tanıma, 21 "emin değilim").
- **`tools/eval_wildtrack.py`:** WILDTRACK'te uçtan uca ölçüm. Etiketleri okuyor, kalibrasyon dosyalarından homografiyi hesaplıyor (`ground_homographies`, satır 67):
  ```python
  H[v] = K @ [r1 r2 t]                 # zemin (z=0) → kamera v görüntüsü
  H_ab = H[b] @ inv(H[a])              # kamera a görüntüsü → kamera b görüntüsü
  ```
  Doğrulama: bir kişinin ayağı kameradan kameraya ortalama **14 piksel** hatayla taşınıyor (HD görüntüde çok iyi).
- **`tools/fetch_wildtrack_subset.py`:** Veri seti 6.8 GB, C: diskinde 9.3 GB boş yer vardı. Sunucu dosyanın istenen parçasının indirilmesine izin veriyor (HTTP Range). Araç zip'in içindekiler listesini uzaktan okuyup sadece gereken dosyaları indiriyor. **1.6 GB indirildi** (C1, C2, C6 kameraları × 200 kare, `D:\datasets\Wildtrack`).
  - Zip macOS'ta hatalı oluşturulmuş: 4 GB'tan sonraki dosya konumları taşmış (2³² modülünde). Python bunları yanlış okuyordu. `fix_offsets` (satır 96) taşma sayısını sayıp gerçek konumu hesaplıyor.
- **Etiket formatı nüansı:** WILDTRACK kutuları dar insan kutuları değil, zemin konumundan izdüşürülmüş geniş sabit kutular. Dedektör kutularıyla IoU ≥ 0.5'te kişilerin sadece %52'si eşleşiyordu, IoU ≥ 0.3'te %89'u. Bu yüzden eşleştirme eşiği **0.3** seçildi.

### 8.4 İlk WILDTRACK sonucu (Faz 2 sistemi, toparlama kapalı, varsayılan ayarlar)
3 HD kamera (C1, C2, C6), 200 kare (100 saniye), 169 gerçek kişi, 12.590 etiketli kutu. Süre: 326 sn.

| Ölçüt | Değer | Yorum |
|---|---|---|
| Bulunan kişi oranı | %20.6 | **Ana darboğaz.** Kalabalık meydanda dedektör kişilerin çoğunu kaçırıyor. |
| Bulunanlardan numara alan | %34.2 | İkinci darboğaz: kişiler numara alacak kadar uzun görünmeden kayboluyor. |
| **IDP (global)** | **%55.2** | Numara verildiğinde kameralar arası doğruluk |
| IDP (kamera içi) | %64.3 | |
| IDF1 (global) | %7.3 | Yukarıdaki iki darboğaz yüzünden düşük |
| Kullanılan P numarası | 32 (39 oluşturuldu) | 169 kişiye |
| Parçalanma / karışma | 1.24 / 1.97 | Kişiler az bölünüyor ama numaralar karışıyor |

Kamera bazında IDP: C1 %85, C2 %39, C6 %90. C2'de belirgin bir sorun var, incelenecek.

**Toparlama tanısı:** Kalan 20 net kimlik arasında "aynı kişi" çifti yok, 135 "farklı kişi" çifti var. Eşik 0.80'de 1 yanlış birleştirme olurdu, 0.85'te hiç olmazdı. Yani bu sahnede toparlamanın düzelteceği parça yok; eşik ise biraz yükseltilmeli.

**Bu sayılar nasıl okunmalı?**
- WILDTRACK bir **kalabalık dış mekân meydanı** (karede 30+ kişi), ofis değil. Bilerek zor seçildi.
- IDF1'in düşüklüğü büyük ölçüde **tespit ve ilk numara verme** aşamasından geliyor, kimlik eşleştirmeden değil. Ölçüm bu iki darboğazı ilk kez sayıyla gösterdi.
- Dedektör ayarı denemesi (eşik 0.55 → 0.35) başlatıldı ama bilgisayarın belleği yetmediği için yarıda kaldı. Ölçümler bundan sonra tek tek çalıştırılacak.

---

## 9. Bütün sonuçlar tek tabloda

| Faz | Ne değişti | Ana ölçüt | Önce | Sonra |
|---|---|---|---|---|
| 0 | Mimari + CPU optimizasyonu | İşleme hızı (kare/sn/kamera) | 0.5–0.9 | **3–4** (~4–5×) |
| 0 | BGR→RGB, tek model, anında indeks | Oluşan kimlik (60 sn) | 0 | 6 (gerçek 4) |
| 1 | Renk ipuçları + iki aşamalı eşleştirme | Doğru tanıma, %1 yanlış eşleşmede | %19.9 | **%46.7** |
| 1 | | AUC | 0.787 | **0.851** |
| 1 | | Geri gelen kişiyi tanıma (120 sn) | 3 | **15** |
| 2 | Fizik kuralları + "emin değilim" + konum hafızası kapalı | Kameralar arası tutarlılık (sıkı) | %35 | **%85** |
| 2 | | Kimlik sayısı (gerçek 4) | 4 | 7 (bedel) |
| 3 | Toparlama | EPFL kimlik sayısı | 7 | 6 (tutarlılık korunarak) |
| 3 | İlk etiketli ölçüm (WILDTRACK) | Global IDP / IDF1 | — | %55 / %7 (başlangıç çizgisi) |

---

## 10. Bir karenin sistemdeki yolculuğu (kodla birlikte)

```
1. CameraStream._reader (camera_stream.py:54)      kameradan en yeni kare
2. _feed_detector (__main__.py:214)                 kareyi YOLO kuyruğuna at
3. DetectorWorker (detector.py)                     YOLO: insan kutuları
4. CameraNode.process_frame (camera_node.py:91)
   a. _process_tracking_and_reid (:119)             BoostTrack + gerekenlere OSNet + renk ipucu
   b. check_occlusions                              üst üste binenleri işaretle (renk kullanılmasın)
   c. _resolve_identities (:172)
        - "yan yana görüldü" listesini güncelle (:185)
        - numarası olan track: Visual Guard, galeri güncelleme
        - numarası olmayan: _attempt_match_or_enroll (:292)
             → IdentityMatcher.match (identity_matcher.py:80)
                 OSNet ile 5 aday → renkle yeniden puan → veto
                 → topology.impossible_reason ile fiziksel kontrol
                 → net kazanan mı? değilse "emin değilim"
             → eşleşme yoksa _process_enrollment (:340): 10 kare biriktir,
               ortalamayla son kontrol, gerekirse 6 sn bekle, sonra yeni P-ID
        - presence.observe: rota/ziyaret kaydı
   d. _verify_and_correct_identities (:435)         yanlış numaraları 3 onayla düzelt
   e. _perform_cleanup (:443)                       kaybolan track'leri temizle
5. consolidator.step (__main__.py:188)              "birleştirilemez" listesini öğren;
                                                    5 sn'de bir parçaları birleştir
6. Kapanışta _write_routes (__main__.py:359)        output/routes_*.csv
```

**Ayarlar tek yerde:** `src/utils/config.py`. En önemlileri:

| Ayar | Değer | Satır | Ne yapar |
|---|---|---|---|
| `DETECTION_SCORE_THRESHOLD` | 0.55 | 28 | YOLO güven eşiği |
| `MATCH_SIMILARITY_THRESHOLD` | 0.82 | 33 | Kayıt sırasındaki son kontrol eşiği |
| `REENTRY_SIMILARITY_THRESHOLD` | 0.84 | 37 | Tek karede tanıma eşiği |
| `MIN_ENROLL_FRAMES` | 10 | 47 | Yeni kayıt için gereken kare |
| `CUE_WEIGHTS` | 0.5/0.2/0.3/0 | 176 | OSNet / üst / alt / kafa ağırlıkları |
| `COLOR_VETO_THRESHOLD` | 0.35 | 179 | Renk vetosu |
| `AMBIGUITY_MARGIN` | 0.03 | 190 | "Emin değilim" farkı |
| `ENROLL_MAX_DEFER_SECONDS` | 6 | 193 | En fazla bekleme |
| `SPATIAL_GATE_PX` | 60 | 199 | Aynı nokta toleransı |
| `SPATIAL_REACQ_ENABLED` | False | 208 | Konum hafızası |
| `CONSOLIDATE_MIN_SCORE` | 0.80 | 217 | Toparlama eşiği |
| `DETECTION_IMG_SIZE` | 416 | 235 | YOLO giriş boyutu (HD kamerada 960) |

---

## 11. Dürüst değerlendirme: ne biliyoruz, ne bilmiyoruz

**Bildiklerimiz (ölçtük):**
- Sistem başlangıca göre ~4–5 kat hızlı ve CPU'da gerçek zamanlıya yakın çalışıyor.
- Renk ipuçları, aynı güvenlik seviyesinde tanımayı 2–3 katına çıkardı.
- Fizik kuralları ve "emin değilim", verilen numaraların güvenilirliğini %35'ten %85'e çıkardı (EPFL, aynı odayı gören 2 kamera).
- Eski konum hafızası kalabalıkta en büyük hata kaynağıydı.
- WILDTRACK'te asıl darboğaz kimlik eşleştirme değil, **tespit ve ilk numara verme**.

**Bilmediklerimiz:**
1. **Ayrı alanları gören kameralarda gerçek performans.** "Yürüme süresi" ve "aynı anda iki yer" kuralları sadece birim testleriyle doğrulandı (`tests/test_faz2_rules.py`, 6 test). Hem EPFL hem WILDTRACK aynı alanı gören kameralar.
2. **Ofiste daha iyi mi kötü mü çalışacağı.** Lehine: ofis kameraları muhtemelen daha az kalabalık, kişiler daha büyük görünecek. Aleyhine: çalışanlar benzer giyinebilir, kamera açıları farklı olabilir. Ancak gerçek görüntüyle ölçülebilir.
3. **Market-1501 ile ReID doğruluğu** henüz ölçülmedi (araç hazır, veri indirilmedi).
4. **Boy ipucu** henüz yok. Kamera kalibrasyonu gerektiriyor.
5. Test verilerimiz kısa (100–150 sn). Uzun süreli (saatler) davranış bilinmiyor.

---

## 12. Sıradaki adımlar

**Faz 3'ün kalanı (WILDTRACK):**
1. Dedektör ayarı (eşik, giriş boyutu) ve kayıt süresi denemeleri, **tek tek** çalıştırılarak. Hedef: "bulunan" ve "numara alan" oranlarını yükseltmek.
2. C2 kamerasındaki düşük kesinliğin (%39) sebebini bulmak.
3. Toparlama eşiğini etiketlere göre ayarlamak (ilk veri 0.80'in biraz düşük olduğunu gösteriyor).
4. Önce/sonra tablosu → `docs/FAZ3_RAPOR.md` → PR.

**Ofis kameraları gelince:**
1. `config/cameras.yaml`'a kameralar ve topoloji (hangi kameralar aynı alanı görüyor, aralarında kaç saniye).
2. 3–4 kişiyle bilinen rotada yürünerek kısa, etiketli bir test kaydı. Eşikler gerçek veriyle ayarlanacak.
3. Kamera bağlantısı kopunca yeniden bağlanma (RTSP).
4. Gizlilik: verinin ne kadar saklanacağı (retention), kırpıntıların diske yazılmaması.
5. Rota raporu (günlük özet, kalış süreleri).
6. Kalibrasyonla boy tahmini.

**Performans (isteğe bağlı):** ONNX Runtime / OpenVINO ile Intel işlemcide 2–4 kat ek hız. Ek paket kurulumu gerektiriyor.

---

## 13. Komutlar

```bash
# Canlı (laptop kamerası)
.venv\Scripts\python.exe src\__main__.py

# EPFL videosuyla
.venv\Scripts\python.exe src\__main__.py --cameras config\cameras.epfl-room.yaml

# Tekrarlanabilir çalıştırma + rota CSV + olay ekran görüntüleri
.venv\Scripts\python.exe tools\replay.py --cameras config\cameras.epfl-room.yaml --seconds 150 --routes rotalar.csv --snapshots snaps

# Etiketsiz tutarlılık ölçümü (EPFL)
.venv\Scripts\python.exe tools\eval_consistency.py --cameras config\cameras.epfl-room.yaml

# WILDTRACK'in bir kısmını indirme
.venv\Scripts\python.exe tools\fetch_wildtrack_subset.py --out D:\datasets\Wildtrack --cams 1,2,6 --frames 200

# WILDTRACK ölçümü (~5–6 dk)
.venv\Scripts\python.exe tools\eval_wildtrack.py --data D:\datasets\Wildtrack --cams 1,2,6

# Bir ayarı değiştirerek ölçme
.venv\Scripts\python.exe tools\eval_wildtrack.py --data D:\datasets\Wildtrack --cams 1,2,6 --set DETECTION_SCORE_THRESHOLD=0.35

# Testler (her biri birkaç saniye)
.venv\Scripts\python.exe tests\test_faz2_rules.py
.venv\Scripts\python.exe tests\test_id_metrics.py
.venv\Scripts\python.exe tests\test_faz3_consolidation.py
```

**Ayrıntılı faz raporları:** `docs/FAZ0_RAPOR.md`, `docs/FAZ1_RAPOR.md`, `docs/FAZ2_RAPOR.md`, `docs/KOD_REHBERI.md`.
