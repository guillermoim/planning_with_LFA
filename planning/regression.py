import argparse
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd

from regression import *
from src.data_loader import data_loader
from src.key_features import get_key_features

warnings.filterwarnings("ignore")


def filter_list(the_list, mask):
    flist = list()
    for i in range(mask.size):
        if mask[i]:
            flist.append(the_list[i])
    return flist


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear Regression")
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--method", type=str, default='L0learn')

    args = parser.parse_args()

    filepath = args.filepath
    bins = args.bins
    method = args.method

    basepath = os.path.dirname(filepath)

    X, y, names, complexities = data_loader(filepath)
    y = np.abs(y)

    n, p = X.shape
    step = p // bins

    # names, complexities, X = eliminate(X, names, complexities, tol=1)

    # Select seed for reproducibility
    np.random.seed(10)
    random.seed(10)

    # MAX COMPLEXITY FEATURES
    # c_max = 8

    omp_interval = np.arange(1, 10)
    lasso_positive = False
    lasso_complexities = False
    lasso_nlambda = 20
   

    n, p = np.shape(X)

    # n = number of states, p = number of features
    # This mask should be in the data
    # TODO: increasing number of features available
    # Create a permuation of the non ground truth features
    mask = np.empty(p, dtype=bool)

    filename = os.path.basename(filepath)
    filename = filename.split(".")[0]

    # edit the stdout, to write out in a more appropiate buffer
    old_stdout = sys.stdout

    simulations = []

    # This loop increases one feature at a time from the nfs pool
    # (by setting the mask to True)

    base_features, weights, bias = get_key_features(filepath, regression=1)
    base_features = list(map(names.index, base_features))

    if len(base_features) == 0:
        base_features = [0]

    for i in range(p):
        if i in base_features:
            mask[i] = True
        else:
            mask[i] = False

    # Adjust base features back to 0. This applies to base features
    # defined as concept_distance that can take value of \infty
    # but from a more abstract level they should be 0.
    idxs = np.where(X > 500)
    X[idxs] = 0

    fs = []
    nfs = []

    # fs contains ground truth feats if provided, nfs contains the rest
    fs = []
    nfs = []
    for i in range(p):
        if mask[i]:
            fs.append(i)
        else:
            nfs.append(i)

    perm = np.random.permutation(nfs)

    for i, fi in enumerate(range(0, len(nfs), step)):
        if i == 0:
            N = np.sum(mask)
            # print(f"Considering {N} features.")
            mX = X[:, mask]
        else:
            mask[perm[fi : fi + step]] = True
            N = np.sum(mask)
            # print(f"Considering {N} features.")
            mX = X[:, mask]

        fil_names = filter_list(names, mask)
        fil_complexities = filter_list(complexities, mask)
        fil_idx = filter_list(range(0, p), mask)

        oo = os.path.basename(basepath)

        if "L0learn" == method:
            os.makedirs(f"outputs/{oo}/l0learn", exist_ok=True)
            path = f"outputs/{oo}/l0learn"
            with open(f"{path}/{N}-features.out", "w") as f:
                sys.stdout = f
                print(f"Considering {N} features")
                print("L0Learn Method\n----------\n")
                sim = do_l0learn(mX, y, fil_idx, fil_complexities, fil_names)
                simulations.extend(sim)

        sys.stdout = old_stdout
    df = pd.DataFrame.from_records(simulations)
    df.to_csv(os.path.join(basepath, f"regression.csv"), index=False)
