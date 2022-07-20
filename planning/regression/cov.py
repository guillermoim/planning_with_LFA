from fileinput import filename
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from data_loader import data_loader
import os

def plot_cov(X, names, filename, title='default'):
    
    df = pd.DataFrame(data=X)
    f = plt.figure(figsize=(19, 15))
    
    X = df.corr()
    X = np.tril(X)
    plt.matshow(X, fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), names, fontsize=10, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), names, fontsize=10)

    for (i, j), z in np.ndenumerate(df.corr()):
        if i >= j:
            plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16)
    
    plt.savefig(f'corr_plots/{filename}.pdf', bbox_inches='tight', dpi=500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XXXXX')
    parser.add_argument('--filepath', type=str)


    args = parser.parse_args()

    filepath = args.filepath
    filename = os.path.basename(filepath).split('.')[0]

    X, y, names, complexities = data_loader(filepath, filter=False)

    plot_cov(X, names, filename, title='default')