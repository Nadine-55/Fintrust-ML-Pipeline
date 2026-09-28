import pandas as pd


def preprocess_transactions(df, mode='train'):
    df = df.copy()

    if mode == 'train':
        df = df.dropna(subset=['Device_Type', 'Location'])
    elif mode == 'inference':
        df['Device_Type'] = df['Device_Type'].fillna('Unknown')
        df['Location'] = df['Location'].fillna('Unknown')

    df['International_Transaction'] = df['International_Transaction'].map({'Yes': 1, 'No': 0})
    df['Risk_Review_Flag'] = df['Risk_Review_Flag'].map({'Yes': 1, 'No': 0})

    df['transaction_hour'] = df['Transaction_DateTime'].dt.hour
    df['transaction_dayofweek'] = df['Transaction_DateTime'].dt.dayofweek

    df = pd.get_dummies(df, columns=['Channel', 'Transaction_Type', 'Device_Type'], drop_first=True)

    return df


def preprocess_customers(df):
    df = df.copy()

    income_map = {
        'Below 100k': 0,
        '100k-249k': 1,
        '250k-499k': 2,
        '500k-999k': 3,
        '1m+': 4
    }
    df['Monthly_Income_Band'] = df['Monthly_Income_Band'].map(income_map)

    df = pd.get_dummies(
        df,
        columns=['Gender', 'City', 'Customer_Segment', 'Account_Type', 'Preferred_Channel', 'Account_Status'],
        drop_first=True
    )

    return df

def merge_data(processed_transactions, processed_customers):
    return processed_transactions.merge(processed_customers, on='Customer_ID', how='left')


def select_model_columns(merged):
    drop_cols = ['Transaction_ID', 'Customer_ID', 'Customer_Name',
                 'Transaction_DateTime', 'Transaction_Status', 'Location']
    return merged.drop(columns=drop_cols)

INCOME_MAP = {'Below 100k': 0, '100k-249k': 1, '250k-499k': 2,
              '500k-999k': 3, '1m+': 4}


def build_features(transactions, customers, mode='train'):
    df = transactions.copy()

    if mode == 'train':
        df = df.dropna(subset=['Device_Type', 'Location'])
    else: 
        df['Device_Type'] = df['Device_Type'].fillna('Unknown')

    df['International_Transaction'] = df['International_Transaction'].map({'Yes': 1, 'No': 0})
    df['transaction_hour'] = df['Transaction_DateTime'].dt.hour
    df['transaction_dayofweek'] = df['Transaction_DateTime'].dt.dayofweek

    cust = customers.copy()
    cust['Monthly_Income_Band'] = cust['Monthly_Income_Band'].map(INCOME_MAP)

    df = df.merge(cust, on='Customer_ID', how='left', validate='many_to_one')
    return df.reset_index(drop=True)