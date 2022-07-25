import sys
sys.path.append('..')
from src import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

def _remove_constant_features(df, y, to_keep):

    train_features, test_features, train_labels, test_labels=train_test_split(df, y, test_size=0.1, random_state=41)
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(train_features)

    to_drop = [col for col in train_features.columns
                    if col not in train_features.columns[constant_filter.get_support()] and col not in to_keep]


    df.drop(to_drop, axis=1, inplace=True)

    return df


def _remove_duplicated_features(df, y, to_keep):
    
    train_features, test_features, train_labels, test_labels=train_test_split(df, y, test_size=0.1, random_state=41)
    train_features_T = train_features.T
    unique_features = train_features_T.drop_duplicates(keep='first').T
    
    to_drop = [col for col in train_features.columns if col not in unique_features.columns]
    to_drop = [col for col in to_drop if col not in to_keep]

    df.drop(to_drop, axis=1, inplace=True)


    return df

def _remove_completely_correlated_features(df, to_keep):
    
    corr_matrix = df.corr().abs()

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool_))

    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] == 1) and column not in to_keep]

    df.drop(to_drop, axis=1, inplace=True)

    return df

def preprocess_data(X, y, names, complexities, base_feat):
    
    df = pd.DataFrame(X, columns=names)
    names = list(map(lambda x:names[x], sorted(range(len(names)), key=lambda x: complexities[x])))
    
    df = df[names]  

    _, p = df.shape

    df = _remove_constant_features(df, y, base_feat)
    df = _remove_duplicated_features(df, y, base_feat)
    df = _remove_completely_correlated_features(df, base_feat)

    _, p1 = df.shape

    print(f'{p - p1} features have been eliminated.')

    new_complexities = complexities.sort()
    new_complexities = list(map(lambda x: complexities[names.index(x)], df.columns))

    return df.to_numpy(), y, df.columns.to_list(), complexities

def plot_cov(X, names, filename, title='default'):
    
    df = pd.DataFrame(data=X)
    f = plt.figure(figsize=(19, 15))
    
    X = df.corr()
    X = np.tril(X)
    
    print('correlation done')
    
    plt.matshow(X, fignum=f.number)
    print('matshow done')
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), names, fontsize=10, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), names, fontsize=10)
    
    print('ticks done')
    for (i, j), z in np.ndenumerate(df.corr()):
        if i >= j:
            plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    
    print('final step')
    plt.title(title, fontsize=16)
    print('final checj)')
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', dpi=500)
    
    
if __name__ == '__main__':
    
    path = "../results/blocksworld/p-on-4blocks-0-flag"
    path_matrix = f"{path}/feat_matrix_extended.csv"
    X, y, names, complexities = data_loader(path_matrix)
    base_feat, _, _ = get_key_features("blocksworld-on", regression=True)
    
    complexities[np.where(np.isnan(complexities))] = 1.
    complexities = pd.Series(complexities)
    df = pd.DataFrame(X, columns=names)
    names = list(map(lambda x:names[x], sorted(range(len(names)), key=lambda x: complexities[x])))
    df = df[names]  

    print('Original shape', df.shape)

    df = remove_constant_features(df, y, base_feat)
    df = remove_duplicated_features(df, y, base_feat)
    df = remove_completely_correlated_features(df, base_feat)

    print('Processed shape', df.shape)
    print(list(map(lambda x: x in df.columns, base_feat)))
    print(print(base_feat))