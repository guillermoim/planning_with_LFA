import pandas as pd
import numpy as np

def data_loader(filepath, max_complexity):
    
    with open(filepath, "r") as file:
        df = pd.read_csv(file, sep=";", header=0)
    
    
    complexities = df.iloc[-1, :-1].to_numpy()
    complexities[np.where(np.isnan(complexities))] = 1.
    
    df.iloc[-1, :-1] = pd.Series(complexities)

    # FIlter out features more complex than the max_complexity 
    filter = np.where(complexities > max_complexity)[0].tolist()
    
    df = df.drop(columns = df.columns[filter])

    features = df.columns.to_list()
    X = df.to_numpy()

    return X[:-1, :-1], X[:-1, -1], features[:-1], X[-1, :-1]



if __name__ == '__main__':
    X, y, names, c = data_loader('../results/blocksworld/p-clear-3blocks-0-flag/feat_matrix_extended.csv', 4)

    print(X.shape, y.shape, c.max())