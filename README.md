# Çoklu Kamera Kişi Tanıma ve Takip Sistemi (Multi-Camera Person Recognition & Tracking System)

Bu proje, birden fazla kamera akışını kullanarak kişileri tespit eden, takip eden ve kimliklerini (Re-Identification) veritabanında saklayan gelişmiş bir bilgisayarlı görü sistemidir. YOLOv8, OSNet ve Faiss gibi modern teknolojileri kullanarak gerçek zamanlıya yakın performans sunar.

## Temel Özellikler

*   **Çoklu Kamera Desteği:** Birden fazla video kaynağından (IP kamera, video dosyası vb.) eş zamanlı görüntü işleme.
*   **Gelişmiş Tespit (Detection):** `YOLOv8` modeli ile yüksek doğrulukta insan tespiti.
*   **Kişi Takibi (Tracking):** `BoxMOT` (BoostTrack) kütüphanesi ile kararlı nesne takibi.
*   **Kimlik Yeniden Tanımlama (Re-ID):** `OSNet` mimarisi ile kişilerin farklı kameralarda veya zamanlarda tekrar görüldüğünde tanınması.
*   **Vektör Arama:** `Faiss` (Facebook AI Similarity Search) kütüphanesi ile milyonlarca vektör arasında milisaniyeler içinde benzerlik araması.
*   **Kalıcı Veritabanı:** Tespit edilen kişilerin özellikleri ve kimlik bilgileri `MongoDB` üzerinde saklanır.
*   **Akıllı Galeri Yönetimi:** Her kişi için birden fazla açıdan (ön, arka, yan) özellik vektörü saklayarak tanıma başarımını artırır (Multi-Vector Gallery).
*   **Test-Time Augmentation (TTA):** Re-ID çıkarımı sırasında görüntü iyileştirme teknikleri ile daha sağlam özellik vektörleri.
*   **Mekansal Eşleştirme (Spatial Matching):** Homografi matrisleri kullanarak kameralar arası geçişlerde fiziksel konuma dayalı doğrulama.
*   **Asenkron Mimari:** Tespit, veritabanı yazma ve görselleştirme işlemlerinin ayrıştırılmasıyla yüksek FPS.

## Sistem Mimarisi

Sistem modüler bir yapıya sahiptir:
1.  **Detector Worker (Process):** Ana işlemden bağımsız çalışır. Görüntü karelerini alır ve YOLOv8 ile insanları tespit eder.
2.  **Camera Nodes:** Her bir kamera akışını yönetir, görüntüleri okur ve takip (tracking) işlemini başlatır.
3.  **ReID Extractor:** Tespit edilen kişilerin görüntülerini alır, ön işlemden geçirir (CLAHE, Resize) ve OSNet modeli ile 512 boyutlu öznitelik vektörüne çevirir.
4.  **Person Manager & Faiss:** Çıkarılan vektörleri veritabanındaki mevcut kişilerle karşılaştırır. Eşleşme varsa kimliği doğrular, yoksa yeni bir kişi kaydı oluşturur.
5.  **MongoDB & Async Writer:** Veri yazma işlemleri ana döngüyü bloklamamak için arka planda asenkron olarak yapılır.

## Gereksinimler

*   **İşletim Sistemi:** Windows / Linux
*   **Python:** 3.8+
*   **Donanım:**
    *   CUDA destekli NVIDIA GPU (Önerilen: 6GB+ VRAM)
    *   CPU (Yalnızca CPU modunda çalışabilir ancak yavaştır)
*   **Veritabanı:** MongoDB (Yerel veya Bulut)

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
    FAISS_COLLECTION=faiss_index
    ```

## Yapılandırma

Tüm ayarlar `src/utils/config.py` dosyasında bulunur. Önemli parametreler:

*   **VIDEO_SOURCES:** İzlenecek video dosyalarının yolları veya kamera ID'leri (0, 1...).
*   **DETECTION_SCORE_THRESHOLD:** Tespit için minimum güven skoru (Örn: 0.55).
*   **MATCH_SIMILARITY_THRESHOLD:** İki kişinin aynı sayılması için gereken minimum benzerlik (Örn: 0.85).
*   **REID_MODEL_PATH:** Kullanılacak ReID modelinin yolu.
*   **GALLERY_SIZE:** Bir kişi için saklanacak maksimum vektör sayısı.

## Kullanım

Sistemi başlatmak için ana modülü çalıştırın:

```bash
python src/__main__.py
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
│   │   ├── detector.py      # YOLOv8 Worker Process
│   │   ├── reid_extractor.py# OSNet Özellik Çıkarıcı
│   │   ├── camera_node.py   # Tekil Kamera İşleyicisi
│   │   └── quality_checker.py # Görüntü Kalite Kontrolü
│   ├── database/            # Veri Katmanı
│   │   ├── mongo_handler.py # MongoDB Bağlantısı
│   │   ├── faiss_manager.py # Faiss Vektör İndeksi Yönetimi
│   │   └── async_writer.py  # Arka Plan Yazma Kuyruğu
│   ├── tracking/            # Takip ve Mantık Katmanı
│   │   ├── person_manager.py # Kişi Yönetimi ve Eşleştirme Mantığı
│   │   ├── cross_camera_matcher.py # Kameralar Arası Eşleştirme
│   │   └── spatial_matcher.py # Konum Bazlı Eşleştirme
│   └── utils/               # Yardımcı Araçlar
│       ├── config.py        # Merkezi Konfigürasyon
│       └── logger.py        # Loglama Yapısı
└── tools/
    └── calibrate_homography.py # Kamera kalibrasyon aracı
```

## Modül Detayları

### Core (Çekirdek)
*   **Detector:** Python `multiprocessing` kullanarak ana döngüyü yavaşlatmadan GPU üzerinde YOLOv8 çıkarımı yapar.
*   **ReidExtractor:** `BoxMOT` kütüphanesinin sağladığı modelleri kullanarak, algılanan kişilerden 512 boyutlu vektörler üretir. Parlaklık dengeleme (CLAHE) ve kırpma işlemleri burada yapılır.

### Database (Veritabanı)
*   **FaissManager:** `HNSW` (Hierarchical Navigable Small World) indeksi kullanır. Bu, büyük veri setlerinde bile çok hızlı "en yakın komşu" araması sağlar. İndeks bellekte tutulur ve periyodik olarak diske/DB'ye yedeklenir.
*   **AsyncWriter:** Veritabanına yazma işlemleri disk I/O gerektirdiği için yavaştır. Bu modül, yazılacak verileri bir kuyruğa alır ve ayrı bir thread'de işleyerek FPS düşüşünü engeller.

### Tracking (Takip)
*   **PersonManager:** Sistemin beynidir. Yeni gelen tespitlerin veritabanındaki kişilerle eşleşip eşleşmediğine karar verir. `Faiss`'ten gelen adayları ve mekansal bilgileri birleştirir.
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