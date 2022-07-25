import pandas as pd
import numpy as np

def data_loader(filepath, filter=False):
    with open(filepath, "r") as file:
        df = pd.read_csv(file, sep=";", header=0)
    if filter:
        pass
    features = df.columns.to_list()
    df = df.to_numpy()
    
    complexities = df[-1, :-1]
    complexities[np.where(np.isnan(complexities))] = 1.

    return df[:-1, :-1], df[:-1, -1], features[:-1], complexities


