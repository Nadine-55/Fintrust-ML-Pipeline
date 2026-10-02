import numpy as np
import pandas as pd
import pytest

from src.pipeline import load_raw_data, run_training, run_prediction
from src.data_validation import DataValidationError
from src.model import load_model, FEATURE_COLUMNS
from src.preprocessing import build_features


@pytest.fixture(scope="module")
def data():
    return load_raw_data()


@pytest.fixture(scope="module")
def trained(data):
    transactions, customers = data
    return run_training(transactions, customers)


# --- input validation (from Week 2, expanded) ---

def test_valid_input_passes(data):
    transactions, customers = data
    _, _, warnings = run_training(transactions, customers, save=False)
    assert warnings == []


def test_missing_values_allowed_at_inference(data):
    transactions, customers = data
    scores, warnings = run_prediction(transactions.head(5), customers)
    assert len(scores) == 5


def test_unexpected_category_warns_not_raises(data):
    transactions, customers = data
    odd = transactions.copy()
    odd.loc[0, 'Channel'] = 'Carrier Pigeon'
    scores, warnings = run_prediction(odd.head(5), customers)
    assert any('Carrier Pigeon' in w for w in warnings)


def test_invalid_data_type_raises(data):
    transactions, customers = data
    bad = transactions.copy()
    bad['Amount_NGN'] = bad['Amount_NGN'].astype(str)
    with pytest.raises(DataValidationError):
        run_prediction(bad, customers)


def test_empty_input_raises(data):
    transactions, customers = data
    with pytest.raises(DataValidationError):
        run_prediction(transactions.iloc[0:0], customers)


# --- new for Week 3 ---

def test_model_loading():
    model = load_model()
    assert hasattr(model, 'predict_proba')


def test_prediction_generation(data):
    transactions, customers = data
    scores, _ = run_prediction(transactions.head(5), customers)
    assert len(scores) == 5
    assert scores['Risk_Score'].between(0, 1).all()


def test_output_format(data):
    transactions, customers = data
    scores, _ = run_prediction(transactions.head(5), customers)
    assert list(scores.columns) == ['Transaction_ID', 'Risk_Score', 'Risk_Review_Prediction']
    assert scores['Risk_Review_Prediction'].isin(['Yes', 'No']).all()


def test_reproducibility(data):
    transactions, customers = data
    scores_1, _ = run_prediction(transactions.head(20), customers)
    scores_2, _ = run_prediction(transactions.head(20), customers)
    assert np.array_equal(scores_1['Risk_Score'].values, scores_2['Risk_Score'].values)