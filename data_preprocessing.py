import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()
    
    X = df.drop("Weight", axis=1)
    y = df["Weight"]
    
    return X, y
