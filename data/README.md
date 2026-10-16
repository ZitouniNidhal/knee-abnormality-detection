# Dataset Guide — RSNA Knee Abnormality Detection

This repository is built for the **RSNA Knee Abnormality Detection** challenge.

## 1. Official Dataset (Primary)

- **Competition Link:** [RSNA Knee Abnormality Detection on Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- **Data Page:** [https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)

### Download via Kaggle CLI
Make sure you have accepted the competition rules on Kaggle, then run:

```bash
# Download the dataset archive
kaggle competitions download -c rsna-knee-abnormality-detection

# Unzip into data/raw/
unzip rsna-knee-abnormality-detection.zip -d data/raw/
```

### Expected Folder Structure in `data/raw/`
```text
data/raw/
├── train.csv                # StudyInstanceUID, reports, targets
├── test.csv                 # Test study identifiers
├── sample_submission.csv    # Submission template with the 12 targets
├── train/                   # Training MRI DICOM/image series organized by study
│   └── <StudyInstanceUID>/
│       └── <SeriesInstanceUID>/
│           └── <SOPInstanceUID>.dcm
└── test/                    # Inference/evaluation MRI series
```

---

## 2. Alternative / Complementary Benchmarks

- **Stanford MRNet Dataset:**
  - URL: [https://stanfordmlgroup.github.io/competitions/mrnet/](https://stanfordmlgroup.github.io/competitions/mrnet/)
  - Focus: Knee MRI exams for ACL tears, meniscal tears, and general abnormalities.

---

## 3. Target Findings (12 Clinical Multi-Labels)

The pipeline detects the following 12 conditions (`TARGETS` in `src/rsna_knee/constants.py`):

1. `ACL` (Anterior Cruciate Ligament tear)
2. `MCL` (Medial Collateral Ligament tear)
3. `Medial Meniscus` (Medial meniscus tear/degeneration)
4. `Lateral Meniscus` (Lateral meniscus tear/degeneration)
5. `Medial OA` (Medial compartment Osteoarthritis)
6. `Lateral OA` (Lateral compartment Osteoarthritis)
7. `PF OA` (Patellofemoral Osteoarthritis)
8. `Effusion` (Joint effusion / excess fluid)
9. `Synovitis` (Synovial inflammation)
10. `Baker's` (Baker's / popliteal cyst)
11. `Contusion` (Bone bruise / contusion)
12. `Fracture` (Occult / acute bone fracture)
