import os
import pandas as pd

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_data():
    df = pd.read_csv('aapl_msi_sbux.csv')
    return df.values