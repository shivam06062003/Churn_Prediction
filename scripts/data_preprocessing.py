import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop('Churn_Yes', axis=1)
    y = df_encoded['Churn_Yes']

    return X, y