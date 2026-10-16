# RSNA Knee Abnormality Detection

Production-ready Multimodal Deep Learning pipeline for detecting 12 clinically significant knee abnormalities from MRI scans (DICOM series) and radiology reports.

---

## 🎯 Target Findings (12 Clinical Multi-Labels)

The system predicts the presence of 12 conditions evaluated during a comprehensive knee MRI:

| Category | Targets |
|---|---|
| **Ligaments** | `ACL` (Anterior Cruciate Ligament), `MCL` (Medial Collateral Ligament) |
| **Menisci** | `Medial Meniscus`, `Lateral Meniscus` |
| **Osteoarthritis (OA)** | `Medial OA`, `Lateral OA`, `PF OA` (Patellofemoral) |
| **Joint Pathology** | `Effusion`, `Synovitis`, `Baker's` (Cyst), `Contusion`, `Fracture` |

---

## 📊 Dataset Links

This repository uses the official challenge dataset:

- **Official Competition:** [RSNA Knee Abnormality Detection on Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- **Data Download Page:** [Kaggle Competition Data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)

### Download via Kaggle CLI:
```bash
# Accept the competition rules on Kaggle first, then:
kaggle competitions download -c rsna-knee-abnormality-detection
unzip rsna-knee-abnormality-detection.zip -d data/raw/
```

### Alternative Benchmark:
- [Stanford MRNet Dataset](https://stanfordmlgroup.github.io/competitions/mrnet/)

For detailed directory structures, see [`data/README.md`](file:///data/README.md).

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run Inference Server (FastAPI)
```bash
docker-compose up rsna-knee-api
```
Or locally:
```bash
uvicorn rsna_knee.inference.server:app --host 0.0.0.0 --port 8000
```

### 3. API Endpoints
- `GET /health` — Health check & model status
- `POST /predict` — Multilabel classification for knee MRI slices
- `GET /metrics` — Prometheus telemetry
