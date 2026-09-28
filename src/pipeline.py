from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

from src.data_validation import check_transactions, check_customers, check_join
from src.preprocessing import build_features
from src.model import (time_based_split, build_pipeline, get_target,
                       FEATURE_COLUMNS, save_model)
from src.predict import predict_transactions

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
REPORTS_DIR = ROOT / 'reports'


def load_raw_data():
    customers = pd.read_excel(RAW_DIR / 'FinTrust_Customer_Data.xlsx')
    transactions = pd.read_excel(RAW_DIR / 'FinTrust_Transaction_Data.xlsx')
    return transactions, customers


def run_training(transactions=None, customers=None, save=True):
    if transactions is None or customers is None:
        transactions, customers = load_raw_data()

    # Validation: stops here if the data is unusable
    check_customers(customers)
    warnings = check_transactions(transactions, mode='train')
    check_join(transactions, customers)

    # Preprocessing, feature preparation, time-based split
    features = build_features(transactions, customers, mode='train')
    train_part, test_part = time_based_split(features)

    # Model
    pipe = build_pipeline()
    pipe.fit(train_part[FEATURE_COLUMNS], get_target(train_part))

    # Evaluation
    y_test = get_target(test_part)
    report = classification_report(y_test, pipe.predict(test_part[FEATURE_COLUMNS]))

    if save:
        save_model(pipe)
        REPORTS_DIR.mkdir(exist_ok=True)
        (REPORTS_DIR / 'week3_model_evaluation.txt').write_text(report)

    return pipe, report, warnings


def run_prediction(transactions, customers, model=None, threshold=0.5):
    check_customers(customers)
    warnings = check_transactions(transactions, mode='inference')
    check_join(transactions, customers)

    scores = predict_transactions(transactions, customers,
                                  model=model, threshold=threshold)
    return scores, warnings