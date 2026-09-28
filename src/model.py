from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.preprocessing import select_model_columns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib
from pathlib import Path

NUMERIC_COLS = ['Amount_NGN', 'Age', 'Tenure_Months', 'Digital_Engagement_Score',
                'Monthly_Income_Band', 'transaction_hour', 'transaction_dayofweek']


def time_based_split(merged, train_fraction=0.8):
    merged_sorted = merged.sort_values('Transaction_DateTime').reset_index(drop=True)
    split_index = int(len(merged_sorted) * train_fraction)
    train_data = merged_sorted.iloc[:split_index]
    test_data = merged_sorted.iloc[split_index:]
    return train_data, test_data


def prepare_xy(data):
    data = select_model_columns(data)
    X = data.drop(columns=['Risk_Review_Flag'])
    y = data['Risk_Review_Flag']
    return X, y


def train_model(X_train, y_train):
    X_train = X_train.copy()
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, scaler


def predict_labels(model, scaler, X):
    X = X.copy()
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    return model.predict(X)

BINARY_FEATURES = ['International_Transaction']
CATEGORICAL_FEATURES = ['Channel', 'Transaction_Type', 'Device_Type', 'Gender',
                        'City', 'Customer_Segment', 'Account_Type',
                        'Preferred_Channel', 'Account_Status']
FEATURE_COLUMNS = NUMERIC_COLS + BINARY_FEATURES + CATEGORICAL_FEATURES


def get_target(df):
    return df['Risk_Review_Flag'].map({'Yes': 1, 'No': 0})


def build_pipeline():
    preprocessor = ColumnTransformer([
        ('scale', StandardScaler(), NUMERIC_COLS),
        ('binary', 'passthrough', BINARY_FEATURES),
        ('encode', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ], remainder='drop')

    return Pipeline([
        ('preprocess', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000)),
    ])

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'risk_model.joblib'


def save_model(pipe, path=MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    return path


def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file at {path}. Train and save a model first.")
    return joblib.load(path)