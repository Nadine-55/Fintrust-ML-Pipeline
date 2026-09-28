import pandas as pd

def validate_transactions(df):
    issues = {}
    issues['missing_device_type'] = df['Device_Type'].isna().sum()
    issues['missing_location'] = df['Location'].isna().sum()
    issues['duplicate_transaction_ids'] = df['Transaction_ID'].duplicated().sum()
    issues['unexpected_amount'] = (df['Amount_NGN'] <= 0).sum()
    issues['unexpected_risk_values'] = (~df['Risk_Review_Flag'].isin(['Yes', 'No'])).sum()
    return issues


def validate_customer(df):
    issues = {}
    issues['missing_values'] = df.isnull().sum().sum()
    issues['duplicate_customer_ids'] = df['Customer_ID'].duplicated().sum()
    issues['unexpected_gender'] = (~df['Gender'].isin(['Male', 'Female', 'Prefer not to say'])).sum()
    expected_bands = ['Below 100k', '100k-249k', '250k-499k', '500k-999k', '1m+']
    issues['unexpected_income_band'] = (~df['Monthly_Income_Band'].isin(expected_bands)).sum()
    return issues

class DataValidationError(ValueError):
    """Raised when input data is too broken to process safely."""


REQUIRED_TRANSACTION_COLUMNS = [
    'Transaction_ID', 'Customer_ID', 'Transaction_DateTime', 'Transaction_Type',
    'Amount_NGN', 'Channel', 'Device_Type', 'International_Transaction']

NOT_NULL_TRANSACTION_COLUMNS = [
    'Transaction_ID', 'Customer_ID', 'Transaction_DateTime', 'Transaction_Type',
    'Amount_NGN', 'Channel', 'International_Transaction']

REQUIRED_CUSTOMER_COLUMNS = [
    'Customer_ID', 'Age', 'Gender', 'City', 'Customer_Segment', 'Account_Type',
    'Tenure_Months', 'Digital_Engagement_Score', 'Monthly_Income_Band',
    'Preferred_Channel', 'Account_Status']

EXPECTED_TRANSACTION_CATEGORIES = {
    'Transaction_Type': ['Card Purchase', 'Cash Withdrawal', 'Transfer',
                         'Deposit', 'Airtime/Data', 'Bill Payment'],
    'Channel': ['Mobile App', 'ATM', 'POS', 'Web', 'USSD'],
}

VALID_INCOME_BANDS = ['Below 100k', '100k-249k', '250k-499k', '500k-999k', '1m+']


def check_transactions(df, mode='inference'):
    """Raise on structural problems; return a list of non-fatal warnings."""
    if df is None or len(df) == 0:
        raise DataValidationError('Input is empty: no transactions to process.')

    required = list(REQUIRED_TRANSACTION_COLUMNS)
    if mode == 'train':
        required += ['Location', 'Risk_Review_Flag']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f'Missing required columns: {missing}')

    nulls = [c for c in NOT_NULL_TRANSACTION_COLUMNS if df[c].isna().any()]
    if nulls:
        raise DataValidationError(f'Nulls found in required fields: {nulls}')

    if not pd.api.types.is_numeric_dtype(df['Amount_NGN']):
        raise DataValidationError('Amount_NGN must be numeric.')
    if not pd.api.types.is_datetime64_any_dtype(df['Transaction_DateTime']):
        raise DataValidationError('Transaction_DateTime must be a datetime.')
    if (df['Amount_NGN'] <= 0).any():
        raise DataValidationError('Amount_NGN must be greater than 0.')
    if (~df['International_Transaction'].isin(['Yes', 'No'])).any():
        raise DataValidationError("International_Transaction must be 'Yes' or 'No'.")
    if mode == 'train' and (~df['Risk_Review_Flag'].isin(['Yes', 'No'])).any():
        raise DataValidationError("Risk_Review_Flag must be 'Yes' or 'No'.")

    warnings = []
    for col, allowed in EXPECTED_TRANSACTION_CATEGORIES.items():
        unexpected = set(df[col].unique()) - set(allowed)
        if unexpected:
            warnings.append(f'Unexpected {col} values (encoder will ignore them): '
                            f'{sorted(map(str, unexpected))}')
    return warnings


def check_customers(df):
    if df is None or len(df) == 0:
        raise DataValidationError('Customer table is empty.')
    missing = [c for c in REQUIRED_CUSTOMER_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f'Missing required customer columns: {missing}')
    nulls = [c for c in REQUIRED_CUSTOMER_COLUMNS if df[c].isna().any()]
    if nulls:
        raise DataValidationError(f'Nulls found in customer fields: {nulls}')
    if df['Customer_ID'].duplicated().any():
        raise DataValidationError('Duplicate Customer_ID values in customer table.')
    if (~df['Monthly_Income_Band'].isin(VALID_INCOME_BANDS)).any():
        raise DataValidationError('Unexpected Monthly_Income_Band values.')


def check_join(transactions, customers):
    orphans = set(transactions['Customer_ID']) - set(customers['Customer_ID'])
    if orphans:
        raise DataValidationError(
            f'{len(orphans)} transaction Customer_IDs have no customer record.')