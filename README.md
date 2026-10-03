# FinTrust ML Pipeline

Machine Learning Engineering Intern: submission for the AnalystLab Africa
FinTrust Digital Bank Experience Lab (Weeks 1–3).

## Project Purpose

This repository contains a reproducible machine-learning pipeline that
predicts whether a FinTrust transaction requires risk review, using the
synthetic `Risk_Review_Flag` label. The ML Engineering track's role is
the workflow, testing, and service layer around a model, not final model
selection a development model (logistic regression) is used here behind
a documented, swappable interface. The `Risk_Review_Flag` is a synthetic
educational label and is not a real fraud determination.

## Repository Structure

| Path | Purpose |
|---|---|
| `data/raw/` | Original, unmodified FinTrust source files |
| `data/processed/` | Reserved for saved intermediate data (not currently used) |
| `notebooks/` | Exploratory work, weekly development notebooks |
| `src/` | Reusable pipeline code |
| `src/data_validation.py` | Checks that stop the pipeline on bad data, plus reporting functions |
| `src/preprocessing.py` | Cleaning, feature building, merging customer and transaction data |
| `src/model.py` | Model pipeline definition, training split, save/load |
| `src/predict.py` | Turns raw transactions into scored predictions |
| `src/pipeline.py` | Entry points: `run_training`, `run_prediction` |
| `src/api.py` | FastAPI service exposing the pipeline over HTTP |
| `tests/` | `pytest` suite covering validation, model loading, prediction, output format, reproducibility |
| `models/` | Saved trained model (`risk_model.joblib`) |
| `reports/` | Evaluation results and test output |
| `requirements.txt` | Pinned dependencies |

## Setup

```bash
pip install -r requirements.txt
```

Place the three FinTrust source files in `data/raw/`:
`FinTrust_Customer_Data.xlsx`, `FinTrust_Transaction_Data.xlsx`,
`FinTrust_Data_Dictionary.xlsx`.

## Running the Pipeline

All commands below are run from the repository root.

**Train the model and generate an evaluation report:**
```python
from src.pipeline import load_raw_data, run_training

transactions, customers = load_raw_data()
model, report, warnings = run_training(transactions, customers)
print(report)
```
This validates the data, builds features, splits by time (not randomly,
to avoid leaking future transactions into training), fits the model,
saves it to `models/risk_model.joblib`, and writes the evaluation report
to `reports/`.

**Score new transactions:**
```python
from src.pipeline import run_prediction

scores, warnings = run_prediction(transactions, customers)
print(scores.head())
```
Returns a table with one row per transaction: `Transaction_ID`,
`Risk_Score` (0–1), and `Risk_Review_Prediction` (Yes/No at a 0.5
threshold). `warnings` lists any non-fatal issues found, such as an
unexpected category.

## Running the API

```bash
uvicorn src.api:app --reload
```

- `POST /predict` — accepts a list of transactions, returns risk scores
- `GET /health` — confirms the service is running

Interactive docs are available at `http://127.0.0.1:8000/docs`. Example
request using Python:
```python
import requests
response = requests.post("http://127.0.0.1:8000/predict", json=[{
    "Transaction_ID": "FT-T000001",
    "Customer_ID": "FT-C01135",
    "Transaction_DateTime": "2026-01-01T00:00:00",
    "Transaction_Type": "Card Purchase",
    "Amount_NGN": 5923.80,
    "Channel": "Mobile App",
    "Device_Type": "Android",
    "International_Transaction": "No"
}])
print(response.json())
```

## Running the Tests

```bash
pytest tests/ -v
```
9 tests covering valid input, missing values, unexpected categories,
invalid data types, empty input, model loading, prediction generation,
output format, and reproducibility.

## Known Limitations

- The baseline logistic regression model has low recall (~4%) on the
  risk-review class due to class imbalance in the data (19.6% positive
  rate). Addressing this is a Data Science track concern; the pipeline
  itself is correct and reproducible regardless of model quality.
- `Location` is excluded from modelling pending a fairness/bias
  discussion.
- `Transaction_Status` is excluded as a feature due to leakage risk —
  it may not be known at the time a prediction is needed.
- No Data Science model was available during Weeks 2–3. The model
  interface (a fitted scikit-learn `Pipeline` with `predict_proba`,
  saved/loaded via `save_model`/`load_model`) is designed to accept
  one if provided later.
- The API runs locally for development only; it has not been deployed.
## Status

Week 3 complete, pipeline logic moved from notebooks into `src/`
modules, model wrapped in a single scikit-learn `Pipeline` (fixing a
column-mismatch bug present in the Week 2 notebook version), model
save/load implemented, strict input validation added, a `pytest` suite
of 9 tests added, and a FastAPI service built around the pipeline.