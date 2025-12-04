import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df, categorical_cols=[]):
    df = df.fillna("Unknown")
    for col in categorical_cols:
        le = LabelEncoder()
        df[col+'_encoded'] = le.fit_transform(df[col])
    return df
