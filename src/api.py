from typing import Optional, List
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import load_raw_data, run_prediction
from src.data_validation import DataValidationError

app = FastAPI(title="FinTrust Risk Review API")

_, customers = load_raw_data()

class TransactionIn(BaseModel):
    Transaction_ID: str
    Customer_ID: str
    Transaction_DateTime: datetime
    Transaction_Type: str
    Amount_NGN: float
    Channel: str
    Device_Type: Optional[str] = None
    International_Transaction: str


class PredictionOut(BaseModel):
    Transaction_ID: str
    Risk_Score: float
    Risk_Review_Prediction: str


@app.post("/predict", response_model=List[PredictionOut])
def predict(transactions: List[TransactionIn]):
    df = pd.DataFrame([t.dict() for t in transactions])
    try:
        scores, warnings = run_prediction(df, customers)
    except DataValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return scores.to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok"}