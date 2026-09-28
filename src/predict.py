import pandas as pd
from src.preprocessing import build_features
from src.model import load_model, FEATURE_COLUMNS

DEFAULT_THRESHOLD = 0.5


def predict_transactions(transactions, customers, model=None,
                         threshold=DEFAULT_THRESHOLD):
    if model is None:
        model = load_model()

    features = build_features(transactions, customers, mode='inference')
    scores = model.predict_proba(features[FEATURE_COLUMNS])[:, 1]

    return pd.DataFrame({
        'Transaction_ID': features['Transaction_ID'],
        'Risk_Score': scores.round(4),
        'Risk_Review_Prediction': ['Yes' if s >= threshold else 'No' for s in scores],
    })