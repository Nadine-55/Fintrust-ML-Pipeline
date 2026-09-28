from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.preprocessing import select_model_columns

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