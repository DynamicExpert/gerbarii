# Network Anomaly Detection

Binary classification of network traffic as **normal** or **attack** using classical ML and an unsupervised autoencoder. Full pipeline from raw NSL-KDD files to metrics, ROC/PR curves and feature importance.

## Why this matters

Intrusion detection systems (IDS) remain a core layer of network defence. Signature-based rules miss zero-day and polymorphic attacks; supervised models catch known patterns, while reconstruction-based autoencoders flag traffic that simply does not look like the normal baseline. A reproducible, well-documented pipeline is the foundation for any production IDS experiment.

## Dataset

**NSL-KDD** (improved version of KDD Cup 1999)

| Item | Value |
|------|-------|
| Source | [UNB CIC](https://www.unb.ca/cic/datasets/nsl.html) |
| Train file | `KDDTrain+.txt` (~125 973 records) |
| Test file | `KDDTest+.txt` (~22 544 records) |
| Features | 41 (3 categorical, 38 numeric) + label |
| Task | Binary: `normal` (0) vs any attack (1) |
| Attack families | DoS, Probe, R2L, U2R |

Download the two `.txt` files and place them in `data/`.

Alternative (not wired by default): [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).

## Methods

| Model | Role | Why |
|-------|------|-----|
| **Random Forest** | Strong supervised baseline | Handles mixed feature types, built-in feature importance, robust to scale |
| **XGBoost** | Gradient-boosted trees | Usually best tabular performance, fast inference |
| **Autoencoder (PyTorch)** | Unsupervised anomaly detector | Trained only on normal traffic → high reconstruction error signals novel attacks |

Class imbalance is handled with **SMOTE** on the training set (configurable to `class_weight` or `none`).

## Results (expected after training on NSL-KDD)

Typical metrics on `KDDTest+` (exact numbers depend on random seed and hyper-parameters):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|-----|---------|--------|
| Random Forest | ~0.97 | ~0.72 | ~0.83 | ~0.96 | ~0.95 |
| XGBoost | ~0.97 | ~0.75 | ~0.85 | ~0.97 | ~0.96 |
| Autoencoder | ~0.90 | ~0.65 | ~0.75 | ~0.88 | ~0.85 |

> On NSL-KDD the supervised models dominate because most test attacks belong to families seen in training. The autoencoder becomes more valuable when the test set contains truly novel attack types.

## Project structure

```
network-anomaly-detection/
├── README.md
├── requirements.txt
├── data/                  # put KDDTrain+.txt & KDDTest+.txt here
│   └── .gitkeep
├── notebooks/
│   └── 01_eda.ipynb       # exploratory analysis
├── src/
│   ├── data_loader.py     # load NSL-KDD, integrity checks
│   ├── preprocessing.py   # one-hot, scaling, SMOTE
│   ├── models.py          # RF, XGBoost, Autoencoder
│   ├── train.py           # full training pipeline
│   └── evaluate.py        # metrics + figures
├── configs/
│   └── config.yaml        # paths & hyper-parameters
├── results/
│   ├── figures/           # confusion matrices, ROC, PR, importance
│   ├── models/            # saved .joblib / .pt files
│   └── metrics.json
└── tests/
    └── test_preprocessing.py
```

## How to run

```bash
# 1. Clone & install
git clone <your-repo-url>
cd network-anomaly-detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download NSL-KDD
# Visit https://www.unb.ca/cic/datasets/nsl.html
# Place KDDTrain+.txt and KDDTest+.txt into data/

# 3. Train all models
python -m src.train --config configs/config.yaml

# 4. Evaluate & generate figures
python -m src.evaluate --config configs/config.yaml

# 5. (optional) unit tests
pytest tests/ -v

# 6. (optional) EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

## Generated figures

After `evaluate.py` you will find in `results/figures/`:

1. **01_class_distribution.png** — bar charts of normal vs attack in train/test  
2. **02_cm_*.png** — confusion matrices for each model + best model  
3. **03_roc_curves.png** — ROC curves of all models on one plot  
4. **04_pr_curves.png** — Precision-Recall curves  
5. **05_feature_importance_*.png** — top-20 features for RF and XGBoost  

Metrics are written to `results/metrics.json`.

## Limitations

- NSL-KDD is an old, heavily pre-processed dataset; real traffic has concept drift, encrypted payloads and far higher volume.
- Binary labels collapse all attack types; a multi-class or hierarchical detector is more useful operationally.
- The autoencoder threshold is set on the training normal distribution — it needs recalibration when the normal baseline changes.
- SMOTE synthesises minority samples in feature space; it does not create realistic packet sequences.
- No online / streaming evaluation; the pipeline is offline batch only.

## Future work

- Add CICIDS2017 / UNSW-NB15 loaders and multi-class labels.
- Experiment with isolation forest, LOF and variational autoencoders.
- Feature selection / SHAP explanations for analyst interpretability.
- Export ONNX models and a minimal FastAPI inference service.
- Drift detection and scheduled re-training hooks.

## References

1. Tavallaee et al. — *A Detailed Analysis of the KDD CUP 99 Data Set* (NSL-KDD paper).  
2. Sharafaldin et al. — *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization* (CICIDS2017).  
3. Sommer & Paxson — *Outside the Closed World: On Using Machine Learning for Network Intrusion Detection* (IEEE S&P 2010).  
4. NSL-KDD download: https://www.unb.ca/cic/datasets/nsl.html  
5. CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html  

## License

MIT
