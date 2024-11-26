import pandas as pd

def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    return df.values