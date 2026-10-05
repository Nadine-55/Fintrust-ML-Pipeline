Machine Learning Fintrust Project 
## Repository Structure

| Path | Purpose |
|---|---|
| `data/raw/` | Original, unmodified FinTrust source files |
| `data/processed/` | Reserved for saved intermediate data |
| `notebooks/` | Development and validation notebooks |
| `src/data_validation.py` | Validation checks that can stop the pipeline on bad data |
| `src/preprocessing.py` | Cleaning, feature building, merging customer and transaction data |
| `src/model.py` | Model pipeline definition (scaler + encoder + classifier), training split, save/load |
| `src/predict.py` | Turns raw transactions into scored predictions |
| `src/pipeline.py` | Entry points: `run_training`, `run_prediction` |
| `src/api.py` | FastAPI service exposing the pipeline over HTTP |
| `tests/` | pytest suite, validation, model loading, prediction, output format, reproducibility |
| `models/` | Saved trained model (`risk_model.joblib`) |
| `reports/` | Evaluation results and test output |
| `requirements.txt` | Project dependencies |

## Setup

```bash
pip install -r requirements.txt
```

Place the FinTrust source files in `data/raw/`: `FinTrust_Customer_Data.xlsx`, `FinTrust_Transaction_Data.xlsx`, `FinTrust_Data_Dictionary.xlsx`.

## Usage

**Train and evaluate:**
```python
from src.pipeline import load_raw_data, run_training

transactions, customers = load_raw_data()
model, report, warnings = run_training(transactions, customers)
print(report)
```
Validates the data, builds features, splits chronologically (not randomly, to avoid leaking future transactions into training), fits the model, saves it to `models/risk_model.joblib`, and writes an evaluation report to `reports/`.

**Score transactions:**
```python
from src.pipeline import run_prediction

scores, warnings = run_prediction(transactions, customers)
print(scores.head())
```
Returns `Transaction_ID`, `Risk_Score` (0 to 1), and `Risk_Review_Prediction` (Yes/No). `warnings` surfaces any non-fatal issues, such as an unexpected category value.

**Run the API:**
```bash
uvicorn src.api:app --reload
```
- `POST /predict`: accepts a list of transactions, returns risk scores
- `GET /health`: confirms the service is running
- Interactive docs: `http://127.0.0.1:8000/docs`

```python
import requests
response = requests.post("http://127.0.0.1:8000/predict", json=[{
    "Transaction_ID": "FT-T000001", "Customer_ID": "FT-C01135",
    "Transaction_DateTime": "2026-01-01T00:00:00", "Transaction_Type": "Card Purchase",
    "Amount_NGN": 5923.80, "Channel": "Mobile App",
    "Device_Type": "Android", "International_Transaction": "No"
}])
print(response.json())
```

**Run the tests:**
```bash
pytest tests/ -v
```
9 tests covering valid input, missing values, unexpected categories, invalid data types, empty input, model loading, prediction generation, output format, and reproducibility.

## Design decisions worth knowing

- Encoding lives inside the model pipeline, not preprocessing. A `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` ensures a single live transaction produces the same feature columns as the full training set. An earlier version using `pd.get_dummies()` directly on the dataframe did not have this guarantee, and that gap was caught and fixed during testing.
- Chronological train/test split, not random, chosen specifically because the data is time stamped, to avoid training on information that would not exist yet in a live setting.
- `Transaction_Status` is deliberately excluded as a model feature. It is an outcome resolved after a transaction occurs, so using it risks relying on information unavailable at prediction time.
- The model is swappable by design. A fitted scikit-learn `Pipeline` exposing `predict_proba`, loaded via `load_model()`, is the only contract required, demonstrated by substituting a `RandomForestClassifier` into the same pipeline shape with no changes to the rest of the codebase.

## Known limitations

- The current model has low recall (around 4%) on the risk review class, driven by class imbalance in the data (around 19.6% positive rate). The pipeline, validation, and testing are correct and reproducible independent of this; improving model performance was treated as a modelling concern separate from the engineering work here.
- `Location` is excluded from the model pending an unresolved fairness question. Geography should not influence a risk score without deliberate justification, and none was established within the project's scope.
- The API is a local development service and has not been deployed.
- This repository was built and tested as an individual contribution within a simulated multi track team project; it was not integrated with other tracks' outputs (for example a separately developed predictive model) during the project timeline.

## Stack

Python, pandas, scikit-learn, FastAPI, pytest, Git/GitHub