import pandas as pd


def data_loader(filepath, filter=False):
    with open(filepath, "r") as file:
        df = pd.read_csv(file, sep=";", header=0)
    if filter:
        pass
    features = df.columns.to_list()
    df = df.to_numpy()
    
    # Returns feature valuations and target
    # (features, V*, names, complexities)
    return df[:-1, :-1], df[:-1, -1], features[:-1], df[-1, :-1]


