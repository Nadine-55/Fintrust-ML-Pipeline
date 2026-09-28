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