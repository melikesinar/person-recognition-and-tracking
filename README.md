# Person Re-Identification and Tracking System

Advanced person re-identification system using YOLO, BoostTrack, OSNet, and Faiss.

## 📁 Project Structure
```
src/
├── main.py                # Main application entry point
├── core/                  # Core functionality
│   ├── detector.py        # YOLO detector wrapper
│   ├── tracker.py         # BoostTrack wrapper
│   ├── reid_extractor.py  # Feature extraction logic
│   └── quality_checker.py # Quality assessment
├── database/              # Database management
│   ├── mongo_handler.py   # MongoDB operations
│   ├── faiss_manager.py   # Faiss index management
│   └── async_writer.py    # Async database writer
├── tracking/              # Tracking logic
│   ├── person_manager.py  # Person enrollment/update
│   ├── spatial_matcher.py # Spatial re-acquisition
│   └── buffer_manager.py  # Buffer cleanup
└── utils/                 # Utilities
    ├── config.py          # Configuration management
    └── visualizer.py      # Visualization utilities
```

## 🚀 Features

- ✅ **Dual-Index Pattern**: Non-blocking Faiss index rebuilds
- ✅ **Async Database Operations**: MongoDB writes don't block main thread
- ✅ **Advanced Quality Check**: 6-criteria enrollment assessment
- ✅ **Spatial Re-acquisition**: IoU + motion prediction for re-identification
- ✅ **Test-Time Augmentation**: Horizontal flip for better accuracy
- ✅ **Live Threshold Tuning**: Adjust detection/match thresholds on-the-fly

## ⌨️ Keyboard Controls

| Key | Function |
|-----|----------|
| `q` | Quit application |
| `u` / `j` | Detection threshold up/down |
| `y` / `h` | Match threshold up/down |
| `t` | Toggle TTA (Test-Time Augmentation) |
| `d` | Show database status |
| `k` | Delete last person |
| `r` | Reset entire database |

## 🔧 Configuration

Edit `.env` file:
```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=your_db_name
COLLECTION=persons
FAISS_COLLECTION=faiss_index

MIN_ENROLL_FRAMES=30
MIN_BBOX_AREA_ENROLL=10000
MIN_SHARPNESS_ENROLL=50.0
DETECTION_SCORE_THRESHOLD=0.7
MATCH_SIMILARITY_THRESHOLD=0.85
```

## 📊 System Architecture
```
┌─────────────┐
│   Camera    │
└──────┬──────┘
       │
       v
┌─────────────┐     ┌──────────────┐
│  Detection  │────>│   Tracking   │
│  (YOLOv8)   │     │ (BoostTrack) │
└─────────────┘     └──────┬───────┘
                           │
                           v
                    ┌──────────────┐
                    │  Re-ID Model │
                    │   (OSNet)    │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           v               v               v
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Spatial │   │  Faiss   │   │  Quality │
    │ Matching │   │  Search  │   │  Check   │
    └──────────┘   └──────────┘   └──────────┘
           │               │               │
           └───────────────┼───────────────┘
                           v
                    ┌──────────────┐
                    │  Enrollment  │
                    │   or Match   │
                    └──────┬───────┘
                           │
                           v
                    ┌──────────────┐
                    │   MongoDB    │
                    └──────────────┘
```

## 🎯 Performance Metrics

- **FPS**: 20-30 (depends on hardware)
- **Enrollment Quality**: >90% (6-criteria assessment)
- **Re-acquisition Accuracy**: ~85-90% (IoU + spatial matching)
- **Index Build Time**: <1s for 100 persons (atomic swap)

## 📝 TODO / Future Improvements

- [ ] Replace OSNet with BoT (Bag of Tricks) model
- [ ] Add multi-camera support
- [ ] Implement trajectory analysis
- [ ] Add attribute-based search
- [ ] Privacy features (face blurring)
- [ ] Web UI dashboard

## 📄 License

MIT License
