# Differential Gene Expression Classification using Machine Learning

A reproducible ML pipeline for classifying disease state using gene expression profiles from a GEO dataset.

## Project Summary
This project performs exploratory data analysis (EDA), preprocessing, and supervised machine learning on a GEO gene expression dataset. The objective is to predict disease vs. healthy state using expression features and basic clinical covariates.

## Repository structure
```
GEO-ML-Differential-Expression/
├── README.md
├── data/
│   └── GEO_dataset.csv            # (add your dataset here)
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── 02_Modeling_and_Evaluation.ipynb
├── src/
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── visualize.py
├── results/
│   ├── figures/
│   └── metrics.json
├── requirements.txt
└── LICENSE
```

## Quick start

1. Place your dataset CSV in `data/GEO_dataset.csv`. Expected format:
   - rows = samples
   - columns = features (gene expression) and metadata columns (`Samples`, `age`, `gender`, `target`)
   - `target` is the label column (0 = control, 1 = disease)

2. Create a virtual environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run preprocessing & training:
```bash
python src/preprocess.py --input data/GEO_dataset.csv --output results/processed.csv
python src/train_models.py --input results/processed.csv --output results/metrics.json --figdir results/figures
```

4. Visualize:
```bash
python src/visualize.py --metrics results/metrics.json --outdir results/figures
```

## What to improve
- Add cross-validation & hyperparameter tuning (GridSearchCV)
- Add ROC curves and class probability calibration
- Use feature selection or regularization for high-dimensional gene sets
- Add unit tests and CI (GitHub Actions)

## License
MIT
