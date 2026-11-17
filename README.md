# Çoklu Kamera Kişi Tanıma ve Takip Sistemi (Multi-Camera Person Recognition & Tracking System)

Bu proje, birden fazla kamera akışında kişileri tespit eden, takip eden ve bir kameradan çıkıp başka bir kamerada görünen kişiyi **aynı kişi olarak yeniden tanıyan** (Re-Identification) bir bilgisayarlı görü sistemidir. Amaç kişinin kim olduğunu bulmak değil, anonim bir numara (P1, P2, ...) ile kameralar arasında takip etmektir. YOLOv8 ve OSNet kullanır; GPU olmadan, CPU üzerinde çalışacak şekilde optimize edilmiştir.

> Bu proje, Mustafa Aykut'un geliştirdiği [person-recognition-and-tracking](https://github.com/MustafaAykut77/person-recognition-and-tracking) çalışmasının üzerine geliştirilmektedir.

## Temel Özellikler

*   **Çoklu Kamera Desteği:** Kamera sayısı sınırsızdır; kameralar `config/cameras.yaml` dosyasından okunur (webcam, video dosyası, IP/RTSP akışı). Her kamera kendi thread'inde okunur.
*   **Gelişmiş Tespit (Detection):** `YOLOv8` modeli ile yüksek doğrulukta insan tespiti.
*   **Kişi Takibi (Tracking):** `BoxMOT` (BoostTrack) ile hareket tabanlı (Kalman + IoU) kare kare takip.
*   **Kimlik Yeniden Tanımlama (Re-ID):** `OSNet` mimarisi ile kişilerin farklı kameralarda veya zamanlarda tekrar görüldüğünde tanınması.
*   **Vektör Arama:** Bellekte tutulan, her değişiklikte anında güncellenen tam (exact) benzerlik araması (`VectorIndex`).
*   **İsteğe Bağlı Veritabanı:** `MONGO_URI` tanımlıysa kimlik vektörleri `MongoDB`'de saklanır; tanımlı değilse veya MongoDB'ye ulaşılamazsa sistem bellek modunda çalışır.
*   **Akıllı Galeri Yönetimi:** Her kişi için birden fazla açıdan (ön, arka, yan) özellik vektörü saklayarak tanıma başarımını artırır (Multi-Vector Gallery).
*   **Test-Time Augmentation (TTA):** Re-ID çıkarımı sırasında görüntü iyileştirme teknikleri ile daha sağlam özellik vektörleri.
*   **Mekansal Eşleştirme (Spatial Matching):** Homografi matrisleri kullanarak kameralar arası geçişlerde fiziksel konuma dayalı doğrulama.
*   **CPU Optimizasyonu:** Kamera başına ayarlanabilir işleme hızı, tek paylaşılan ReID modeli, ihtiyaç halinde ReID hesaplama, TorchScript ve denormal düzeltmesi.
*   **Değerlendirme Aracı:** `tools/eval_reid.py` ile Market-1501 üzerinde Rank-1 / mAP ölçümü.

## Sistem Mimarisi

Sistem modüler bir yapıya sahiptir:
1.  **Detector Worker (Process):** Ana işlemden bağımsız çalışır. Görüntü karelerini alır ve YOLOv8 ile insanları tespit eder.
2.  **Camera Nodes:** Her bir kamera akışını yönetir, görüntüleri okur ve takip (tracking) işlemini başlatır.
3.  **ReID Extractor:** Tespit edilen kişilerin görüntülerini alır, ön işlemden geçirir (CLAHE, Resize) ve OSNet modeli ile 512 boyutlu öznitelik vektörüne çevirir.
4.  **Person Manager & VectorIndex:** Çıkarılan vektörleri veritabanındaki mevcut kişilerle karşılaştırır. Eşleşme varsa kimliği doğrular, yoksa yeni bir kişi kaydı oluşturur.
5.  **MongoDB & Async Writer:** Veri yazma işlemleri ana döngüyü bloklamamak için arka planda asenkron olarak yapılır.

## Gereksinimler

*   **İşletim Sistemi:** Windows / Linux
*   **Python:** 3.8+
*   **Donanım:**
    *   CUDA destekli NVIDIA GPU (Önerilen: 6GB+ VRAM)
    *   CPU (Yalnızca CPU modunda çalışabilir ancak yavaştır)
*   **Veritabanı (isteğe bağlı):** MongoDB (Yerel veya Bulut)

## Kurulum

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/kullaniciadi/person_recognition_and_tracking.git
    cd person_recognition_and_tracking
    ```

2.  **Sanal Ortam Oluşturun (Önerilen):**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```

3.  **Bağımlılıkları Yükleyin:**
    Projenin kök dizinindeki `requirements.txt` dosyasını kullanın.
    ```bash
    pip install -r requirements.txt
    ```
    *Not: PyTorch sürümünüzün CUDA sürümünüzle uyumlu olduğundan emin olun. Gerekirse `pytorch.org` adresinden uygun komutu alın.*

4.  **Modelleri İndirin:**
    Proje `models/` klasörü altında aşağıdaki ağırlık dosyalarını bekler:
    *   `yolov8l.pt` (Otomatik inebilir veya manuel ekleyin)
    *   `osnet_x1_0_msmt17.pt` (ReID modeli)

5.  **Çevresel Değişkenleri Ayarlayın (.env):**
    Proje kök dizininde `.env` dosyası oluşturun ve aşağıdaki değişkenleri tanımlayın:
    ```env
    MONGO_URI=mongodb://localhost:27017/
    DB_NAME=person_tracking_db
    COLLECTION=persons
    ```
    `MONGO_URI` boş bırakılırsa sistem veritabanı olmadan (bellek modunda) çalışır.

## Yapılandırma

Kameralar `config/cameras.yaml` dosyasında tanımlanır (örnek: `config/cameras.epfl-room.yaml`):

```yaml
process_fps: 6          # her kamera saniyede kaç kare işlensin
cameras:
  - {id: 0, name: laptop,  source: 0}
  - {id: 1, name: telefon, source: http://192.168.1.20:8080/video}
  - {id: 2, name: koridor, source: videos/koridor.mp4}
```

Diğer ayarlar `src/utils/config.py` dosyasındadır. Önemli parametreler:

*   **DETECTION_SCORE_THRESHOLD:** Tespit için minimum güven skoru (Örn: 0.55).
*   **MATCH_SIMILARITY_THRESHOLD:** İki kişinin aynı sayılması için gereken minimum benzerlik (Örn: 0.85).
*   **REID_MODEL_PATH:** Kullanılacak ReID modelinin yolu.
*   **GALLERY_SIZE:** Bir kişi için saklanacak maksimum vektör sayısı.

## Kullanım

Sistemi başlatmak için ana modülü çalıştırın:

```bash
python src/__main__.py                                             # config/cameras.yaml
python src/__main__.py --cameras config/cameras.epfl-room.yaml     # başka bir kamera dosyası
python src/__main__.py --headless --max-seconds 60                 # ekransız test çalıştırması
```

ReID doğruluğunu ölçmek için (Market-1501 veri seti gerekir):

```bash
python tools/eval_reid.py --data <Market-1501 klasörü> --num-ids 200
```

### Klavye Kısayolları (Çalışma Esnasında)
*   `q`: Çıkış yap.
*   `p`: Akışı duraklat/devam ettir.
*   `d`: Mekansal hata ayıklama (spatial debug) modunu aç/kapat.
*   `r`: **DİKKAT!** Veritabanını ve ID sayacını sıfırlar (Reset).
*   `t`: Test-Time Augmentation (TTA) özelliğini aç/kapat.
*   `u` / `j`: Tespit eşiğini (Detection Threshold) artır/azalt.
*   `y` / `h`: Eşleşme eşiğini (Match Threshold) artır/azalt.

## Proje Yapısı

```
person_recognition_and_tracking/
├── models/                  # YOLO ve ReID model ağırlıkları
├── logs/                    # Sistem logları
├── src/
│   ├── __main__.py          # Ana giriş noktası (Orchestrator)
│   ├── core/                # Çekirdek Yapay Zeka Modülleri
│   │   ├── camera_stream.py # Kamera okuyucu (kamera başına thread)
│   │   ├── detector.py      # YOLOv8 Worker Process
│   │   ├── reid_extractor.py# OSNet Özellik Çıkarıcı (tek, paylaşılan)
│   │   ├── camera_node.py   # Tekil Kamera İşleyicisi
│   │   └── quality_checker.py # Görüntü Kalite Kontrolü
│   ├── database/            # Veri Katmanı
│   │   ├── vector_index.py  # Bellek içi vektör arama
│   │   ├── mongo_handler.py # MongoDB Bağlantısı (isteğe bağlı)
│   │   └── async_writer.py  # Arka Plan Yazma Kuyruğu
│   ├── tracking/            # Takip ve Mantık Katmanı
│   │   ├── person_manager.py # Kişi Yönetimi ve Eşleştirme Mantığı
│   │   ├── cross_camera_matcher.py # Kameralar Arası Eşleştirme
│   │   └── spatial_matcher.py # Konum Bazlı Eşleştirme
│   └── utils/               # Yardımcı Araçlar
│       ├── config.py        # Merkezi Konfigürasyon
│       └── logger.py        # Loglama Yapısı
├── config/
│   └── cameras.yaml         # Kamera tanımları
└── tools/
    ├── calibrate_homography.py # Kamera kalibrasyon aracı
    └── eval_reid.py         # ReID doğruluk ölçümü (Market-1501)
```

## Modül Detayları

### Core (Çekirdek)
*   **Detector:** Python `multiprocessing` kullanarak ana döngüyü yavaşlatmadan GPU üzerinde YOLOv8 çıkarımı yapar.
*   **ReidExtractor:** `BoxMOT` kütüphanesinin sağladığı modelleri kullanarak, algılanan kişilerden 512 boyutlu vektörler üretir. Parlaklık dengeleme (CLAHE) ve kırpma işlemleri burada yapılır.

### Database (Veritabanı)
*   **VectorIndex:** Tüm kişilerin galeri vektörlerini bellekte tutar ve tam (exact) kosinüs benzerliği ile k-NN oylamalı arama yapar. Her kayıt/güncelleme bir sonraki aramada hemen görünür.
*   **AsyncWriter:** Veritabanına yazma işlemleri disk I/O gerektirdiği için yavaştır. Bu modül, yazılacak verileri bir kuyruğa alır ve ayrı bir thread'de işleyerek FPS düşüşünü engeller.

### Tracking (Takip)
*   **PersonManager:** Sistemin beynidir. Yeni gelen tespitlerin veritabanındaki kişilerle eşleşip eşleşmediğine karar verir. `VectorIndex`'ten gelen adayları ve mekansal bilgileri birleştirir.
*   **SpatialMatcher:** Kameralar arasındaki perspektif farkını (Homography) kullanarak, bir kameradaki kişinin diğer kamerada nerede olması gerektiğini tahmin eder. Görsel benzerlik yetersiz kaldığında (örn. kıyafet değişimi zor, ışık kötü) konumsal doğrulama sağlar.

## Sorun Giderme

*   **MongoDB Bağlantı Hatası:** `MONGO_URI` değerinin doğru olduğundan ve MongoDB servisinin çalıştığından emin olun.
*   **Düşük FPS:**
    *   GPU kullandığınızdan emin olun (`torch.cuda.is_available()` True dönmeli).
    *   `config.py` içinde `DETECTION_SCORE_THRESHOLD` değerini artırarak gereksiz işlem yükünü azaltın.
*   **Model Yükleme Hatası:** `models/` klasöründe `.pt` dosyalarının tam ve eksiksiz olduğundan emin olun.

## Lisans

Bu proje [MIT License](LICENSE) altında lisanslanmıştır.

---
**Geliştirici Notu:** Bu proje eğitim ve araştırma amaçlı geliştirilmiştir. Canlı ortamlarda kullanmadan önce güvenlik ve performans testlerinin yapılması önerilir.