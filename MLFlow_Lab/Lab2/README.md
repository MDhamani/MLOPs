# MLflow Lab 2: Breast Cancer XGBoost Tracking

## Changes Made for MLflow Lab 2

- Changed dataset to Breast Cancer (`sklearn.datasets`)
- Replaced baseline classifier with XGBoost (`XGBClassifier`)
- Added end-to-end notebook workflow in `starter.ipynb` (EDA, training, evaluation, artifact logging)
- Standardized outputs to `models/`, `metrics/`, and `mlruns/`
- Added MLflow tracking, model logging, and local serving instructions
- Added troubleshooting notes for tracking URI mismatch, Windows placeholder issues, and missing serving dependencies
- Updated requirements and documentation

## Current Structure

```text
MLFlow_Lab/Lab2/
  README.md
  requirements.txt
  starter.ipynb
  data/                      
  models/                    
  metrics/                   
  mlruns/                  
```

## Environment Setup

Run from `MLFlow_Lab/Lab2`:

```bash
uv venv
uv pip install -r requirements.txt
```

## Workflow 1: Notebook (`starter.ipynb`)

`starter.ipynb` is the active notebook flow and includes:

- Reproducible setup (`SEED`, output folders)
- Data loading and quick EDA
- Train/test split + XGBoost training
- Metrics: `accuracy`, `f1`, `roc_auc`
- Extra evaluation outputs:
  - classification report
  - feature importance CSV
  - confusion matrix image
  - ROC curve image
  - top-feature importance image
- MLflow logging of params/metrics/model/artifacts
- Optional local serving smoke test (`/invocations`)

Artifacts created by notebook runs:

- `models/model_<timestamp>_xgb_model.joblib`
- `metrics/<timestamp>_metrics.json`
- `metrics/feature_importance_latest.csv`
- `metrics/confusion_matrix_latest.png`
- `metrics/roc_curve_latest.png`
- `metrics/feature_importance_latest.png`

## MLflow UI

Start the tracking UI from `MLFlow_Lab/Lab2`:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5001
```

Open:

- `http://127.0.0.1:5001`

## Serve the Model Locally

```bash
mlflow models serve --env-manager=local -m models:/Breast_Cancer_XGBoost@latest -h 0.0.0.0 -p 5001
```

Then test:

```bash
curl -X POST http://localhost:5001/invocations -H "Content-Type: application/json" -d "{\"dataframe_split\": {\"columns\": [\"mean radius\"], \"data\": [[14.0]]}}"
```

Notebook already has a Python `requests` smoke-test cell that posts to `http://localhost:5001/invocations`.