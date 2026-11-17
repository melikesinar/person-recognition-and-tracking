# Faz 2 Raporu: Uzay-Zaman Kuralları, "Emin Değilim" ve Rota Çıktısı

**Tarih:** 17.11.2026 · **Branch:** `feature/faz2-topology`

## 1. Çıkış noktası

Faz 1 testinde gözlenen sorun: **aynı kıyafetli kişiler karıştırılıyor.** EPFL videosundaki 4 kişiden 3'ü koyu üst ve bej pantolon giyiyor. Renk ipucu bu kişileri ayıramıyor; OSNet de bu çözünürlükte (kişiler 100–200 piksel boyunda) zorlanıyor. Farklı kişiler arasında bile birleşik benzerlik 0.93–0.98 çıkıyor.

Faz 2'nin hedefleri:
1. Görünüşün yetmediği yerde **fiziksel kurallar** kullanmak: bir kişi aynı anda iki yerde olamaz, iki kamera arasını yürüme süresinden hızlı geçemez.
2. Sistem emin değilse **tahmin yapmak yerine beklemesi**.
3. Projenin asıl çıktısını üretmek: her kişinin **rotası ve kalış süreleri**.

## 2. Önce ölçüm: karışıklığı sayıya dökmek

Bu video için hazır doğru cevap (etiket) yok. Ama iki kamera **aynı odayı** görüyor. Bu sayede etiketsiz bir doğruluk ölçütü kurulabiliyor: odada aynı noktada duran kişi, aynı anda iki kamerada da **aynı P numarasını** taşımalı.

Homografi (zemin eşleme matrisi), Kamera 0'daki ayak noktasını Kamera 1'deki karşılığına çeviriyor:
- O noktadaki kişinin numarası aynıysa "tutarlı", farklıysa "çelişkili" sayılıyor.
- **Sıkı ölçüt:** Sadece kimin kim olduğu net olan anlar sayılıyor (20 px içinde eşleşme, 50 px içinde başka kimse yok).
- Araç: `tools/eval_consistency.py`.

**Faz 1'in durumu (150 saniye, 2 kamera):**

| Zaman aralığı | 0–30 sn | 30–60 sn | 60–90 sn | 90–120 sn | 120–150 sn |
|---|---|---|---|---|---|
| Tutarlılık | %100 | %13 | %32 | %38 | %16 |

Odada 2 kişi varken sistem kusursuz. 3. ve 4. kişi girince çöküyor. **Toplamda sadece %35.** Testte hissedilen karışıklık ölçümde de görünüyor.

## 3. Asıl hata kaynağının bulunması

Her değişikliği tek tek açıp kapatarak ölçtük. Sonuç beklenmedikti: hatanın asıl kaynağı, önceki koddaki **"kısa süreli konum hafızası"** (spatial re-acquisition).

**Kural şöyle çalışıyordu:** Bir kişi kaybolduktan sonraki 30 saniye içinde, kaybolduğu noktanın yakınında (arama alanı 100 piksele kadar büyüyerek) beliren **herkes**, görünüşüne hiç bakılmadan o kişinin numarasını alıyordu.

**Sorun:** 360 piksel genişliğindeki bir görüntüde, 4 kişinin dolaştığı küçük bir odada, birinin kaybolduğu yerde kısa süre içinde başka biri belirmesi çok sık oluyor. Numaralar bu yüzden sürekli yanlış kişilere taşınıyordu.

| Konum hafızası ayarı | Tutarlılık |
|---|---|
| Önceki hâli (görünüşe bakmadan, 30 sn) | %18 |
| Görünüş doğrulamalı | %57 |
| Görünüş doğrulamalı, 2 sn ve dar alan | %69 |
| **Kapalı** | **%78** |

Görünüş doğrulaması da yetmedi: kişiler çok benzediği için görünüş, yanlış adayı da "onaylıyor". **Karar: kural varsayılan olarak kapatıldı** (`SPATIAL_REACQ_ENABLED = False`). Kişi geri geldiğinde artık görünüş eşleşmesi ve uzay-zaman kurallarıyla tanınıyor. Kural, görünüş doğrulamalı hâliyle ayardan açılabilir. Seyrek ortamlarda (az kişi, büyük alan) faydalı olabilir; gerçek ofis görüntüsüyle tekrar ölçülmeli.

## 4. Yapılan değişiklikler

### 4.1 Kamera topolojisi (`src/tracking/topology.py`, yeni)

Kameraların birbirine göre yerleşimi, kamera dosyasına (`config/cameras*.yaml`) yazılıyor:

```yaml
topology:
  overlapping: [[3, 4]]            # aynı alanı gören kameralar
  transitions:                     # yürüme yolları (iki yönlü)
    - {cameras: [0, 1], min_seconds: 3, max_seconds: 60}   # kapı <-> koridor
    - {cameras: [1, 2], min_seconds: 5, max_seconds: 120}  # koridor <-> ofis
```

Bu tanımdan üç **kesin kural** çıkıyor:
1. **Aynı anda iki yer olmaz:** Aynı alanı görmeyen iki kamerada aynı anda görülen kişi aynı kişi olamaz.
2. **Yürüme süresi:**
   - Kapıdan 2 saniye önce çıkan biri, en az 8 saniyelik yolu olan ofis kamerasında beliremez.
   - Doğrudan bağlı olmayan kameralar için yol üzerindeki süreler toplanıyor: kapı → koridor → ofis = 3 + 5 = 8 sn.
3. **Aynı nokta (aynı alanı gören kameralar):** P2 kamera 0'da kapının önünde duruyorsa, kamera 1'de pencere kenarındaki kişi P2 olamaz. Bu kural homografiyi kullanıyor ve homografi tam isabetli olmadığı için 60 piksel tolerans tanınıyor.

**Neden kesin kural, neden puan değil?** Bunlar fiziksel imkânsızlıklar ve görünüş ne kadar benzer olursa olsun geçerliler. Aynı üniforma giyen iki çalışanı ayırmanın en güvenilir yolu bu.

**`max_seconds` neden kural değil?** Kişi iki kamera arasında kahve almak için durmuş olabilir. Uzun bir süre, "aynı kişi değil" demek için sebep değil; bu değer sadece raporlamada kullanılıyor.

**Süreleri bilmiyorsak:** Topoloji boş bırakılabilir. Bu durumda sadece "aynı anda iki yer olmaz" kuralı uygulanıyor. Sistem gözlediği geçiş sürelerini loga (`TRANSITION` satırları) ve kapanış raporuna yazıyor; gerçek süreler oradan okunup dosyaya girilebilir.

### 4.2 "Emin değilim" kuralı (`src/tracking/identity_matcher.py`)

Önceki kural: "En yüksek puanlı aday eşiği geçtiyse o kişidir."

Yeni kural: Kazanan aday hem eşiği geçmeli **hem de rakiplerinden açıkça önde olmalı:**
- Olası başka bir aday en az 0.03 puan geride olmalı (`AMBIGUITY_MARGIN`).
- İmkânsız bir aday (örneğin şu an başka bir kamerada olan biri) ondan daha benzer olmamalı. Kişi en çok "burada olamayacak birine" benziyorsa, bu kişi bir benzeri ya da yeni biri olabilir.

Bu şartlar sağlanmazsa sistem karar vermiyor. Kişi ekranda **`T12 (?)`** olarak görünüyor ve sistem birkaç saniye daha kanıt topluyor (en fazla 6 sn). Loga gerekçe yazılıyor:

```
[Cam 0] NOT SURE: T16 P3 0.95 vs P5 0.93 (possible). Waiting.
```

**Neden?** Yanlış kimlik iki kişinin rotasını birden bozar. Birkaç saniyelik "bilinmiyor" ise rotada sadece küçük bir boşluk bırakır.

### 4.3 "Yan yana görüldü" kanıtı (`camera_node.py`)

Bir track'in yanında aynı karede görülen kişiler kaydediliyor ve o track **hiçbir zaman** o kişilere dönüşemiyor; o kişiler kadrajdan çıktıktan sonra da. Faz 1'deki "Enroll Conflict" durumunun yerini aldı.

Örnek log:

```
NEW PERSON: T10 -> P5, looks like P3 0.96 but seen next to this person in this camera
```

Bu kişi P3'e çok benziyor ama P3 ile yan yana görüldü, yani P3 değil.

### 4.4 Rota ve kalış süresi çıktısı (`src/tracking/presence.py`, yeni)

Projenin asıl ürünü bu. Aşağıdaki örnekler temsilîdir (ofis kurulumu için). Her kişi için kamera ziyaretleri (giriş, çıkış, kalış süresi) tutuluyor. Program kapanırken `output/routes_<tarih>.csv` dosyasına yazılıyor:

```
person,camera_id,camera,enter,exit,dwell_seconds
P1,0,kapi,2026-11-17 09:02:11,2026-11-17 09:02:19,8.0
P1,1,koridor,2026-11-17 09:02:24,2026-11-17 09:03:05,41.0
```

Loga özet rota ve kameralar arası gözlenen geçiş süreleri de yazılıyor:

```
P1: kapi 8s -> koridor 41s -> ofis 312s
travel kapi -> koridor: n=14, min 3.2s, median 5.1s, max 22.0s
```

Kimlikler sonradan birleştirilirse (otomatik birleştirme) geçmişleri de birleştiriliyor.

### 4.5 Diğer düzeltmeler
- **Kimlik değiştirme kontrolü** (OSNet tabanlı): Bir track, o anda aynı kamerada başka birinin taşıdığı numaraya artık geçirilmiyor.
- **Otomatik birleştirme:** Uzay-zaman kuralına göre farklı olduğu kanıtlanmış iki kimlik birleştirilmiyor.
- **Ayarlar:** EPFL kamera dosyasına "bu iki kamera aynı odayı görüyor" bilgisi eklendi. Bu bilgi olmadan sistem odadaki herkesi iki kişi sayardı; test ettik, 4 kişiye 5 kimlik verdi. **Yeni kurulumda topolojinin doğru girilmesi önemli.**

## 5. Sonuçlar (EPFL oda videosu, 150 saniye, 2 kamera, 4 kişi)

| | Faz 1 | Faz 2 |
|---|---|---|
| Kameralar arası tutarlılık (sıkı) | %35 | **%85** |
| Kameralar arası tutarlılık (gevşek) | %41 | **%78** |
| Numara "ışınlanması" (ani kayma) | 3 | **1** |
| Sona kalan kimlik sayısı (gerçek: 4) | **4** | 7 |
| Numara taşıyan kişi oranı | **%87** | %73 |
| İşleme hızı | aynı | aynı |

Faz 2'nin zaman aralıklarına göre tutarlılığı: %100 · %90 · %90 · %53 · %93.

**Kazanç:** Verilen numaralar artık çok daha güvenilir. Sistem "bu kişi P3" dediğinde, Faz 1'de büyük olasılıkla yanlıştı (%35 tutarlı); Faz 2'de büyük olasılıkla doğru (%85).

**Bedel:** Daha fazla kimlik parçalanması (4 kişiye 7 kimlik) ve daha uzun süre "bilinmiyor" kalan kişiler. Bilinçli bir tercih: **parçalanmış ama doğru bir rota**, sonradan birleştirilebilir. **Karışmış bir rota** ise sessizce yanlış bilgi üretir ve sonradan fark edilemez.

**Hız:** Arka arkaya ölçümde Faz 1 ve Faz 2 aynı sürede çalıştı (40 sn video için 112 ve 116 sn). Canlı testlerde görülen 3.5–6 kare/sn farkı laptopun güç durumundan kaynaklanıyor.

## 6. Dürüst değerlendirme ve sınırlar

1. **Topoloji kuralları bu videoda test edilemedi.** EPFL'deki iki kamera aynı odayı görüyor; "yürüme süresi" ve "aynı anda iki yer" kuralları ancak ayrı alanları gören kameralarda devreye girer. Bu kurallar şu an sadece birim testleriyle doğrulandı (`tests/test_faz2_rules.py`, 6 test). **Gerçek doğrulama ofis kameraları bağlanınca yapılacak.**
2. **Bu video en kötü senaryo:** Düşük çözünürlük (360×288), 4 kişiden 3'ü aynı giyimli, sürekli birbirinin önünden geçiyorlar. Gerçek ofis kameraları büyük olasılıkla daha yüksek çözünürlüklü olacak. Ama çalışanlar da benzer giyinebilir. Sonucun daha iyi mi kötü mü olacağı ancak gerçek görüntüyle ölçülebilir.
3. **Parçalanma:** Bir kişi yanlış numara taşıdığında, o numaranın gerçek sahibi "yan yana görüldü" kuralı yüzünden onu alamıyor ve yeni kimlik açılıyor. Kalan fazladan kimliklerin ana sebebi bu zincirleme etki.
4. **Ölçüt sınırı:** Tutarlılık ölçütü sadece aynı odayı gören kameralarda çalışır. Ayrı alanlar için etiketli test videosu gerekir.

## 7. Ofis kameraları bağlanınca yapılacaklar

1. `config/cameras.yaml` dosyasına kameralar ve **topoloji** girilecek: hangi kameralar aynı alanı görüyor, aralarında kaç saniyelik yol var. Süreler bilinmiyorsa önce boş bırakılıp sistemin ölçtüğü `travel` satırlarından alınacak.
2. 3–4 kişiyle, bilinen bir rotada yürünerek **kısa ve etiketli bir test kaydı** yapılacak. Bu kayıtla:
   - rota doğruluğu ölçülecek,
   - `AMBIGUITY_MARGIN`, eşikler ve konum hafızası ayarı gerçek veriyle seçilecek.
3. Aynı alanı gören kamera çifti varsa `tools/calibrate_homography.py` ile homografi çıkarılacak; tutarlılık ölçütü orada da kullanılabilir.

## 8. Nasıl denenir

```bash
# Kural testleri (1 saniye, video gerekmez)
.venv\Scripts\python.exe tests\test_faz2_rules.py

# EPFL videosunda tekrarlanabilir çalıştırma + rota çıktısı + olay ekran görüntüleri
.venv\Scripts\python.exe tools\replay.py --cameras config\cameras.epfl-room.yaml --seconds 150 --routes rotalar.csv --snapshots snaps

# Tutarlılık ölçümü (yaklaşık 5 dakika)
.venv\Scripts\python.exe tools\eval_consistency.py --cameras config\cameras.epfl-room.yaml

# Eski konum hafızasıyla karşılaştırma
.venv\Scripts\python.exe tools\eval_consistency.py --cameras config\cameras.epfl-room.yaml --set SPATIAL_REACQ_ENABLED=True

# Canlı (pencere açılır, q ile çıkılır; kapanışta output\routes_*.csv yazılır)
.venv\Scripts\python.exe src\__main__.py --cameras config\cameras.epfl-room.yaml
```

Ekranda **`T12 (?)`** etiketi, sistemin o kişi hakkında henüz emin olmadığını gösterir.
