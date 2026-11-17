# Kod Rehberi: Sistem Nasıl Çalışıyor, Ne Değişti, Neden?

Bu doküman sistemi bir karenin kameradan okunup ekrana çizilene kadar geçtiği yol üzerinden anlatır. Her bölümde önce **Mustafa'nın kurduğu yapı**, sonra **Faz 0'da değiştirdiğimiz kısım ve nedeni** yer alır.

Satır numaraları `feature/multicam-reid` branch'indeki koda göredir. VS Code'da `Ctrl+P` ile dosyayı açıp `:satır` yazarak (örneğin `camera_node.py:196`) ilgili satıra gidebilirsin.

---

## 0. Büyük resim

```
 Kamera 0 ─┐                                ┌─> CameraNode 0 ─┐
 Kamera 1 ─┼─> [Kare kuyruğu] ─> YOLO ─>    ├─> CameraNode 1 ─┼─> Ortak: PersonManager + VectorIndex
 Kamera 2 ─┘   (process)     [Tespit kuyruğu]└─> CameraNode 2 ─┘          (tüm kameraların ortak hafızası)
```

Bir kare şu adımlardan geçer:

1. **Okuma.** Kameradan kare alınır.
2. **Tespit (YOLO).** "Bu karede insanlar nerede?" Çıktı, her insan için bir kutu `[x1, y1, x2, y2, güven, sınıf]`.
3. **Takip (BoostTrack).** "Bu kutu, önceki karedeki hangi kutunun devamı?" Her kişiye kamera içinde geçici bir **track ID** (T5, T12...) verilir. Kişi kadrajdan çıkınca bu numara kaybolur.
4. **Görünüm vektörü (ReID, OSNet).** Kişinin kırpılmış görüntüsü 512 sayılık bir vektöre dönüştürülür. Aynı kişinin farklı görüntülerinin vektörleri birbirine yakın, farklı kişilerinkiler uzak olur. Yakınlık **kosinüs benzerliği** ile ölçülür: 1.0 aynı, 0 alakasız.
5. **Kimlik (Person ID).** Vektör, sistemin hafızasındaki kişilerle karşılaştırılır. Eşleşme varsa o kişinin kalıcı numarası (**P-ID**: P1, P2...) verilir, yoksa yeni kişi oluşturulur. **Kameralar arası tanıma bu adımda gerçekleşir**, çünkü hafıza tüm kameralar için ortaktır.

**Track ID ile P-ID arasındaki fark çok önemli:**
- Track ID: Tracker'ın kısa süreli numarasıdır, tek kamera içinde geçerlidir.
- P-ID: Sistemin kişiye verdiği kalıcı numaradır, tüm kameralarda aynı kalır.

Projenin asıl işi track ID'leri doğru P-ID'lere bağlamaktır (`track_id_to_permanent_id_map`).

---

## 1. Giriş noktası: `src/__main__.py`

### Mustafa'nın yapısı
- `MultiCameraSystem` sınıfı tüm parçaları kuruyordu: MongoDB, FAISS, PersonManager, YOLO process'i ve kameralar.
- Kameralar `Config.VIDEO_SOURCES` listesinden okunuyordu. Bu liste kodun içindeydi.
- Ana döngü her turda şunları yapıyordu:
  1. Tüm kameralardan sırayla `cap.read()` ile kare oku ve dedektör kuyruğuna at.
  2. Dedektörden gelen sonuçları al.
  3. Her kamera için `node.process_frame()` çağır.
  4. Kameralar arası tekrar eden kimlikleri kontrol et (`check_redundant_identities`).
  5. Ekrana çiz, döngüyü saniyede 30 tura sabitle.

İyi kararlar:
- YOLO'yu ayrı bir **process**'e koymak. Python'da thread'ler aynı anda hesap yapamaz (GIL), process'ler yapabilir.
- Görüntü karesi ile tespit sonucunu `frame_id` ile eşleştirmek. Sonuç, ait olduğu kareye çizilir; yanlış kareye kayma olmaz.

### Faz 0'da ne değişti, neden?

| Değişiklik | Kod | Neden |
|---|---|---|
| Kameralar YAML dosyasından okunuyor | `config/cameras.yaml`, `config.py:203` `load_cameras()` | Kamera sayısı belli değil. Kamera eklemek için kod değiştirmek gerekmemeli. |
| Her kamera kendi thread'inde okunuyor | `camera_stream.py:23` | `cap.read()` bir sonraki kare gelene kadar **bekler** (bloklar). Sırayla okununca yavaş bir kamera tüm sistemi yavaşlatıyordu. |
| Kare gönderimi ayrı bir thread'de | `__main__.py:195` `_feed_detector` | Eski döngü "kare gönder → ReID yap → kare gönder" şeklindeydi. ReID sürerken dedektöre kare gitmiyordu. Ölçümde dedektör zamanının %59–92'sinde çalışıyordu; ayrı thread ile %99 oldu. |
| Kamera başına `process_fps` | `__main__.py:220` civarı | 30 FPS'yi CPU kaldıramaz. Yürüyen insanı takip etmek için 5–8 FPS yeterli. |
| Tüm sonuçlar sırayla işleniyor | `__main__.py:225` `_process_detections` | Eski kod aynı kameradan gelen birden fazla sonucun yalnızca **sonuncusunu** işleyip diğerlerini atlıyordu (`detections_map[cam_id] = ...` üzerine yazıyordu). |
| Izgara görüntü | `__main__.py:267` | Eski kod tüm kameraları yan yana diziyordu (`np.hstack`); 5 kamerada ekran taşardı. |
| `--headless`, `--max-seconds`, `--cameras` | `__main__.py:340` | Ekransız test edebilmek ve farklı kamera dosyalarıyla çalıştırabilmek için. |
| 10 saniyede bir performans logu | `__main__.py:250` | Sistemin gerçekte kaç FPS işlediği görünmüyordu. "Yavaş" demek yerine ölçebilmek için. |

---

## 2. Kamera okuma: `src/core/camera_stream.py` (yeni dosya)

```python
def _reader(self):                      # ayrı thread'de sürekli çalışır
    ret, frame = self.cap.read()
    with self._lock:
        self._frame = frame             # sadece EN YENİ kare tutulur
        self._frame_idx = idx
```

- **En yeni kare mantığı:** Canlı kamerada işlemci geride kalırsa eski kareleri işlemenin anlamı yok; kişi çoktan ilerlemiş olur. Eski kareler sessizce atlanır. Böylece gecikme birikmez.
- **Video dosyası** (`camera_stream.py:76`): Dosya normalde olabildiğince hızlı okunur. Kod dosyayı kayıt hızında oynatıyor (örneğin 25 FPS) ve sonuna gelince başa sarıyor. Böylece video dosyası gerçek bir kamera gibi davranıyor, test sonuçları gerçekçi oluyor.
- `threading.Lock`, okuyucu thread yazarken ana thread'in yarım kalmış veri okumasını engeller.

---

## 3. Tespit: `src/core/detector.py`

### Mustafa'nın yapısı
`DetectorWorker` ayrı bir process. Kuyruktan `(kamera_id, kare_id, kare)` alır, YOLO çalıştırır, `(kamera_id, kare_id, tespitler)` geri yollar. `classes=0` sadece "person" sınıfını alır. Tespit eşiği, `multiprocessing.Value` ile çalışırken `u`/`j` tuşlarıyla değiştirilebiliyor.

### Faz 0 değişiklikleri
- **`imgsz=416`** (`detector.py:96`). YOLO varsayılan olarak görüntüyü 640 piksele ölçekliyor. EPFL videosu 360x288 olduğu için görüntü **büyütülüyordu**, yani boşa iş yapılıyordu. 416'da aynı 4 kişi bulundu, süre yaklaşık 150 ms'den 95–130 ms'ye indi. Yüksek çözünürlüklü kameralarda kişiler küçük görünüyorsa `DETECTION_IMG_SIZE` tekrar 640 yapılmalı.
- **`configure_cpu_inference`.** Denormal düzeltmesi ve thread sınırı (bkz. bölüm 5).
- **Zamanlama logu.** "Saniyede kaç kare, kare başına kaç ms, yüzde kaç meşgul" bilgisi. Darboğazın dedektör olduğunu bu logla bulduk.

---

## 4. Kamera düğümü: `src/core/camera_node.py` (sistemin kalbi)

`process_frame()` (`camera_node.py:81`) her kare için şu sırayı izler:

```
1. _process_tracking_and_reid   → tracker güncelle + gerekli kişilerin vektörünü çıkar
2. check_occlusions              → üst üste binen kişileri işaretle
3. _resolve_identities           → her track'e P-ID bağla
4. _verify_and_correct_identities→ (2 karede bir) yanlış ID'leri düzelt
5. _perform_cleanup              → kaybolan track'leri temizle
6. çizim
```

### 4.1 Takip + ReID (`camera_node.py:109`)

**Mustafa'nın yapısı:**
- `BoostTrack(..., with_reid=True)`: Her kamera tracker'ı kendi OSNet kopyasını yüklüyordu. Tracker her tespit için kendi içinde vektör hesaplıyordu.
- Sonra aynı kişiler için `ReidExtractor` **bir kez daha** vektör hesaplıyordu.
- Ayrıca ana sistemde bir "master_tracker" vardı; sadece içindeki modeli ödünç almak için oluşturulmuştu.

Sonuç olarak 2 kamerada 3 model kopyası vardı ve her kişi için her karede 2 kez OSNet çalışıyordu. OSNet sistemdeki en pahalı işlem.

**Faz 0:**
```python
def _create_tracker(process_fps):          # camera_node.py:44
    return BoostTrack(with_reid=False,      # tracker kendi modelini YÜKLEMİYOR
                      det_thresh=Config.DETECTION_SCORE_THRESHOLD,
                      max_age=TRACK_MAX_AGE_SECONDS * process_fps,
                      use_ecc=False)
```
- **`with_reid=False`.** Tracker kare kare eşleştirmeyi Kalman filtresi (hareket tahmini) ve IoU (kutu örtüşmesi) ile yapıyor. Sabit kamerada iki ardışık karede kişi neredeyse aynı yerde olduğu için bu yeterli.
- **`det_thresh`.** BoostTrack'in varsayılan eşiği 0.60, sistemin eşiği 0.55. Arada kalan tespitler tracker'a girmeden atılıyordu. Eşitlendi.
- **`max_age` saniye cinsinden.** Eskiden "60 kare" idi. 30 FPS'de 2 saniye, 6 FPS'de 10 saniye eder; FPS değişince davranış da değişiyordu. Artık 4 saniye sabit.
- **`use_ecc=False`.** ECC, kamera hareket ediyorsa (araç kamerası, drone) görüntüyü hizalar. Sabit güvenlik kamerasında gereksiz ve CPU'da pahalı.

**ReID ne zaman çalışıyor?** (`camera_node.py:68` `_needs_embedding`)
```python
if track_id not in track_id_to_permanent_id_map:   # kimliği yok
    return True                                     # → her karede
return now - last_embed_time[track_id] >= 1.0       # kimliği var → saniyede bir
```
Kimliği belli olmayan kişi için hızlı karar gerekiyor, bu yüzden her karede hesaplanıyor. Kimliği belli olan kişi için saniyede bir vektör yeterli; galeri güncellemesi ve "bu hâlâ aynı kişi mi" kontrolü bununla yapılıyor.

Kalite kontrolü (`quality_checker.assess_quality`) Mustafa'nın yapısı ve aynen korundu. Bulanık, kenarda kesilmiş, çok küçük ya da karanlık kırpıntılar ReID'e gönderilmiyor. Kötü görüntüden çıkan vektör kişinin galerisini bozar.

### 4.2 Kimlik çözümleme: `_resolve_identities` (`camera_node.py:157`)
**Bu mantığın tamamı Mustafa'nın, Faz 0'da korundu.** Her track için:

**Durum A: Track'in zaten bir P-ID'si var.**
- **Görsel koruma (Visual Guard, `camera_node.py:196`):** Yeni vektör, kişinin galerisindeki en yakın vektöre 0.60'tan az benziyorsa ID hemen geri alınır. Sebebi: iki kişi yan yana geçerken tracker kutuları karıştırabilir ("ID swap"). Bu kontrol, A kişisinin numarasının B'ye geçmesini engeller.
- Geçerliyse `_handle_existing_track`, saniyede bir galeriyi günceller.

**Durum B: Track'in P-ID'si yok.** `_attempt_match_or_enroll` (`camera_node.py:265`) sırayla üç yol dener:
1. **Mekansal eşleşme** (`spatial_matcher.find_spatial_match`): Son 30 saniyede bu kamerada kaybolan biri, kaybolduğu yere yakın bir yerde tekrar belirdiyse muhtemelen odur. Tahmin alanı geçen süreyle büyüyor. Kısa süreli kayıplar (örneğin biri önünden geçti) için hızlı bir kurtarma.
2. **ReID araması:** Vektör tüm kişilerin galerisinde aranıyor. Eşik **0.88**, normal eşikten (0.85) sıkı (`REENTRY_SIMILARITY_THRESHOLD`). Yeni gelen biri yanlışlıkla eski birine bağlanırsa düzeltmesi zor; bu yüzden ihtiyatlı davranılıyor.
3. **Yeni kayıt** (`_process_enrollment`, `camera_node.py:300`): Eşleşme yoksa kişinin vektörleri biriktiriliyor. `MIN_ENROLL_FRAMES` kadar iyi kaliteli örnek toplanınca:
   - Vektörlerin **ortalaması** alınıyor. Tek karenin gürültüsü ortalamayla azalıyor.
   - Ortalama ile son bir arama daha yapılıyor ("Final Safety Check"). Tek karede eşleşmeyen kişi, ortalamayla eşleşebilir.
   - Yine bulunamazsa yeni P-ID veriliyor.

**Faz 0'daki tek değişiklik:** `MIN_ENROLL_FRAMES` 30'dan 10'a indi. 30 kare, 30 FPS'de 1 saniye ama 6 FPS'de 5 saniye eder. 10 kare 6 FPS'de yaklaşık 1.7 saniye.

### 4.3 Temizlik (`camera_node.py:367`)
Kaybolan track'in P-ID'si `lost_pid_buffer`'a konuyor. Mekansal eşleşme (yukarıdaki 1. yol) bu tampondan yararlanıyor.

---

## 5. ReID çıkarıcı: `src/core/reid_extractor.py`

### Mustafa'nın yapısı
- Kırpıntıya **CLAHE** uygulanıyor. Bu bir kontrast dengeleme yöntemi; LAB renk uzayında sadece parlaklık (L) kanalına uygulanıyor. Karanlık veya parlak kamerada detayları ortaya çıkarmak için.
- Görüntü 128x256'ya boyutlandırılıp ImageNet ortalama/standart sapmasıyla normalize ediliyor.
- İsteğe bağlı **TTA** (`t` tuşu): görüntü ve aynalanmış hali için vektörlerin ortalaması alınıyor.
- Vektör L2 ile normalize ediliyor (uzunluk 1). Böylece iki vektörün nokta çarpımı doğrudan kosinüs benzerliği oluyor.

### Faz 0 değişiklikleri

**1. Renk sırası hatası (en önemli düzeltme)** (`reid_extractor.py:93`)
```python
return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # YENİ
```
OpenCV görüntüleri **BGR** sırasıyla (Mavi-Yeşil-Kırmızı) okur. Neredeyse tüm yapay zeka modelleri gibi OSNet de **RGB** ile eğitildi. Eski kodda bu dönüşüm yoktu, yani model mavi ile kırmızısı yer değiştirmiş görüntüler görüyordu: kırmızı tişört mavi, mavi pantolon kırmızı. Model yine "bir şey" çıkarır ama eğitildiği dağılımın dışında çalışır ve doğruluk düşer. Boxmot'un kendi kodu bu dönüşümü yapıyor (`base_backend.py: cv2.COLOR_BGR2RGB`); eski kodumuz o kodu atlayıp modeli doğrudan çağırdığı için dönüşüm kaybolmuştu.

**2. Model tek kez yükleniyor** (`reid_extractor.py:27`). `ReidAutoBackend` ile doğrudan yükleniyor ve tüm kameralar aynı nesneyi kullanıyor.

**3. Denormal düzeltmesi** (`utils/torch_utils.py:17`)
```python
torch.set_flush_denormal(True)
```
Bilgisayarda ondalık sayılar belirli bir aralıkta saklanır. Sıfıra **aşırı** yakın sayılar (yaklaşık 10⁻³⁸'den küçük) "denormal" adlı özel bir formatta tutulur. x86 işlemciler bu formatla işlem yaparken **10–100 kat yavaşlar**. Ölçümde gördüğümüz garip durum (12 thread, 1 thread'den daha yavaş) bunun belirtisiydi. Bu ayar o sayıları sıfır kabul ediyor. Bu kadar küçük sayıların sonuca etkisi yok: ölçülen fark tam 0.0. Kazanç: 4 kişi için 1452 ms'den 351 ms'ye.

**4. TorchScript** (`reid_extractor.py:55`)
```python
traced = torch.jit.trace(model, example)            # modeli bir kez çalıştırıp işlem grafiğini kaydet
optimized = torch.jit.optimize_for_inference(torch.jit.freeze(traced))
```
OSNet çok sayıda **küçük** katmandan oluşuyor. Normal PyTorch'ta her katman için Python üzerinden ayrı bir çağrı yapılıyor ve bu ek yük, asıl hesaplamadan uzun sürüyordu. Kanıtı: 15 kat daha az hesap yapan en küçük OSNet bile sadece 2 kat hızlıydı. `freeze` + `optimize_for_inference`:
- Batch-normalization katmanını convolution'ın içine gömüyor (iki işlem bire iniyor).
- Python ek yükünü kaldırıyor.

Kazanç: kişi başına 191 ms'den 95 ms'ye. Çıktı aynı (kosinüs 1.0). Bir hata olursa düz modele geri dönüyor.

**5. Thread sınırı (4).** YOLO process'i ile ReID aynı çekirdekleri paylaşıyor. Küçük işlerde çok thread koordinasyon maliyeti getiriyor; ölçümde 2–4 thread en iyi sonucu verdi.

---

## 6. Vektör araması: `src/database/vector_index.py` (FAISS'in yerine)

### Mustafa'nın yapısı (`faiss_manager.py`, silindi)
- FAISS **HNSW** indeksi. Milyonlarca vektörde hızlı ama **yaklaşık** arama yapan bir graf yapısı.
- Her yeni kişide veya galeri değişikliğinde, MongoDB'den **tüm vektörler okunup indeks baştan kuruluyordu** (en fazla 30 saniyede bir, arka plan thread'inde).
- Arama mantığı: en yakın 5 vektör alınıyor, en çok oyu alan kişi kazanıyor (son commit'teki "k-NN voting").

**Sorun (yarış durumu / race condition):**
1. Yeni kişi P5 kaydedildi. MongoDB'ye yazma **kuyruğa** alındı (asenkron).
2. İndeks yeniden kurulmaya başladı ve MongoDB'yi okudu. Yazma henüz bitmediği için **P5 yok**.
3. Sonraki 30 saniye boyunca P5 aramada bulunamıyor. Aynı kişi başka kamerada görünse yeni kişi olarak (P6) kaydediliyor.

### Faz 0
```python
def set_person(self, pid, vectors):     # her kayıt/güncellemede ANINDA çağrılıyor
    self._galleries[pid] = np.stack(vectors)
    self._matrix = None                 # birleşik matris bir sonraki aramada yeniden kurulur

def search(self, query, threshold):
    sims = matrix @ query               # TÜM vektörlerle benzerlik, tek matris çarpımı (vector_index.py:98)
    top = np.argpartition(-sims, k-1)[:k]   # en yakın 5
    # ... aynı oylama kuralı
```
- **Tutarlılık:** `PersonManager` her değişiklikte `vector_index.set_person()` çağırıyor. Değişiklik bir sonraki aramada görünüyor, yarış durumu yok.
- **Hız:** 300 kişi × 60 vektör = 18.000 × 512'lik bir matrisle tek çarpım, CPU'da birkaç milisaniye sürer. HNSW'nin avantajı milyonlarca vektörde ortaya çıkar.
- **Tam sonuç:** HNSW yaklaşık sonuç verir, en yakın komşuyu bazen kaçırabilir. Bu yöntem her zaman doğru en yakın komşuyu bulur.
- Oylama kuralı aynen korundu. Bilinen bir zayıflığı var, Faz 1'de ele alınacak: galerisinde 60 vektör olan kişi, 1 vektörü olan kişiden daha çok oy toplayabilir.

---

## 7. Kişi yönetimi: `src/tracking/person_manager.py`

**Bu dosyanın mantığı tamamen Mustafa'nın**, sadece FAISS çağrıları `vector_index` ile değiştirildi.

**Çok vektörlü galeri.** Bir kişi önden, arkadan ve yandan çok farklı görünür; tek bir vektör hepsini temsil edemez. Bu yüzden her kişi için 60'a kadar vektör saklanıyor:
- `enroll_person` (`person_manager.py:40`): Kayıt anında toplanan vektörlerin ortalaması ilk galeri vektörü oluyor.
- `incremental_update` (`person_manager.py:92`): Yeni vektör galeridekilerden birine 0.90'dan fazla benziyorsa eklenmiyor; o poz zaten var. Farklıysa yeni poz olarak ekleniyor.
- **Galeri dolarsa** (`person_manager.py:143` civarı): Birbirine en çok benzeyen iki vektör bulunup ortalamaları alınıyor ve tek vektöre indiriliyor. Böylece yer açılıyor ama çeşitlilik kaybolmuyor. Akıllıca bir yöntem.
- `merge_identities`: Aynı kişiye iki numara verildiği anlaşılırsa galeriler birleştiriliyor.

---

## 8. Kimlik doğrulama: `src/tracking/identity_verifier.py`

**Bu dosyanın mantığı tamamen Mustafa'nın, aynen korundu.** İki görevi var:

**8.1 Yanlış ID düzeltme** (`verify_and_correct`, `identity_verifier.py:38`): 2 karede bir her track için:
- Son vektörlerin ortalaması mevcut P-ID'ye 0.55'ten az benziyorsa ID düşürme adayı oluyor. **3 kez üst üste** tekrarlanırsa ID düşürülüyor.
- Başka bir kişi mevcut P-ID'den en az 0.04 daha benzer çıkarsa ID değiştirme adayı oluyor. Yine 3 onay gerekiyor. Değişiklikten sonra track 5 saniye kilitleniyor ki iki ID arasında gidip gelmesin.

Bu "çoklu onay" yaklaşımı, tek bir kötü karenin ID'yi bozmasını engelliyor.

**8.2 Tekrar eden kimlik birleştirme** (`check_redundant_identities`, `identity_verifier.py:180`): 2 saniyede bir aktif kişiler ikili ikili karşılaştırılıyor. Görsel benzerlik 0.85'in üstündeyse birleştiriliyor. Kameralar **örtüşüyorsa** homografi ile konum kontrolü de yapılıyor: A kamerasındaki kişinin ayak noktası B kamerasına izdüşürülüyor; B'deki kişinin ayağına denk geliyorsa aynı fiziksel yerde duruyorlar demektir (`spatial_score > 0.80`, `identity_verifier.py:266`).

**Homografi nedir?** Zemin düz bir yüzey olduğu için, iki kameranın zemin görüntüsü arasında 3x3'lük bir matrisle dönüşüm yapılabilir. `tools/calibrate_homography.py` ile iki kamerada zemindeki aynı 4+ noktaya tıklanınca `cv2.findHomography` bu matrisi hesaplıyor.

**Faz 2 için önemli not:** Homografi sadece **örtüşen** kameralarda çalışır. Senin senaryondaki kapı / ofis içi / koridor kameraları birbirini görmüyor. Orada konum yerine **zaman ve topoloji** kullanılacak: "Kapıdan çıkan kişi 5–30 saniye içinde koridorda görünür."

---

## 9. Veritabanı: `mongo_handler.py`, `async_writer.py`

### Mustafa'nın yapısı
- `AsyncDBWriter`: MongoDB'ye yazma işi ayrı bir thread'de, kuyrukla yapılıyor. Disk ve ağ beklemesi görüntü işlemeyi durdurmuyor. Kuyruk dolarsa yazma atlanıyor; görüntünün donmasından iyidir.
- Vektörler `Binary(v.tobytes())` ile ikili veri olarak saklanıyor.

### Faz 0
- MongoDB **isteğe bağlı** (`__main__.py:113`): bağlantı yoksa `NullDBWriter` (hiçbir şey yapmayan yazıcı) kullanılıyor ve sistem çalışmaya devam ediyor. Anonim takipte kimliklerin oturum boyunca yaşaması zaten yeterli.
- Açılışta tüm kişiler tek seferde belleğe yükleniyor (`load_all_persons`). Eskiden her kişi ilk görüldüğünde tek tek MongoDB'den çekiliyordu.

---

## 10. Faz 1/2'de ele alınacak, şu an bilinen zayıflıklar

1. **Tek vektör yetmiyor.** Test videosunda 4 kişiye 6 kimlik verildi. Faz 1: kıyafet rengi (üst/alt beden), saç ve boy ipuçları, her birinin güvenilirliğine göre ağırlıklandırma.
2. **Oylama kuralı** büyük galerili kişileri kayırıyor (bölüm 6).
3. **Otomatik birleştirme sadece 20 saniyeden yeni kimliklerde yapılıyor** (`MERGE_MAX_AGE_SECONDS`). Daha eski bir tekrar kimlik hiç birleşmiyor.
4. **"Enroll Conflict" kuralı** (`camera_node.py:333`): Eşleşen kişi başka bir yerde aktifse yeni ID veriliyor. Örtüşen kameralarda doğru. Ama ayrı kameralarda kişi A kamerasından çıktıktan sonra tracker onu birkaç saniye daha aktif tutuyor; bu sürede B kamerasında görünürse yanlışlıkla yeni ID alabilir. Faz 2'de kamera topolojisi ile çözülecek.
5. **CPU sınırı:** Bu laptopta toplam yaklaşık 6 kare/sn. Sonraki adım ONNX Runtime veya OpenVINO.
