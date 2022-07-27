import argparse
import os
import random
import string
import sys
import warnings

import numpy as np
import pandas as pd
from src.data_loader import data_loader
from src.key_features import get_key_features

from regression import *

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
    parser.add_argument("--outpath", type=str)
    parser.add_argument("--method", type=str, default='l0learn')
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--max_complexity", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=1.)

    args = parser.parse_args()

    filepath = args.filepath
    outpath = args.outpath
    bins = args.bins
    method = args.method
    task = args.task
    max_complexity = args.max_complexity
    threshold = args.threshold

    X, y, names, complexities = None, None, None, None

    with open(f"{filepath}","r") as file:
        
        for i, subfile in enumerate(file.readlines()):
            subfile = subfile.strip('\n')
            if subfile == '':
                break
            if i < 1:
                X, y, names, complexities = data_loader(subfile, max_complexity)
            else:
                X_, y_, _, _ = data_loader(subfile, max_complexity)

                X, y = np.vstack([X, X_]), np.concatenate([y, y_])

    # Preprocess data: remove constant, duplicate and perfectly correlated features
    base_features, _, _ = get_key_features(task, 1)
    X, y, names, complexities = preprocess_data(X, y, names, complexities, base_features)

    # names, complexities, X = eliminate(X, names, complexities, tol=0.9)
    n, p = X.shape
    step = p // bins
    # Select seed for reproducibility

    omp_interval = np.arange(1, 10)
    lasso_positive = False
    lasso_complexities = False
    lasso_nlambda = 20

    # n = number of states, p = number of features
    # This mask should be in the data
    # TODO: increasing number of features available
    # Create a permuation of the non ground truth features
    mask = np.empty(p, dtype=bool)

    # This loop increases one feature at a time from the nfs pool
    # (by setting the mask to True)


    base_features = list(map(names.index, base_features))

    for i in range(p):
        if i in base_features:
            mask[i] = True
        else:
            mask[i] = False

    # Adjust base features back to 0. This applies to base features
    # defined as concept_distance that can take value of \infty
    # but from a more abstract level they should be 0.
    idxs = np.where(X[:, base_features] > 500)
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


    # create tmp-buffer
    tmp = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))
    res = []

    for i, fi in enumerate(range(0, len(nfs), step)):
        if i == 0:
            N = np.sum(mask)
            print(f"Considering {N} features.")
            mX = X[:, mask]
        else:
            mask[perm[fi : fi + step]] = True
            N = np.sum(mask)
            print(f"Considering {N} features.")
            mX = X[:, mask]

        fil_names = filter_list(names, mask)
        fil_complexities = filter_list(complexities, mask)
        fil_idx = filter_list(range(0, p), mask)

        if method == "omp":
            os.makedirs(f"outputs/{tmp}/omp", exist_ok=True)
            path = f"outputs/{tmp}/omp"
            with open(f"{path}/{p}-features.out", "w") as f:
                print(f"Considering {N} features")
                print("----------\nomp Method\n----------\n")
                res1 = do_OMP(mX, y, omp_interval, fil_idx, fil_complexities, fil_names)
                res.extend(res1)

        if method == "lasso":
            os.makedirs(f"outputs/{tmp}/lasso", exist_ok=True)
            path = f"outputs/{tmp}/lasso"
            with open(f"{path}/{p}-features.out", "w") as f:
                print(f"Considering {N} features")
                print("Lasso Method\n----------\n")
                res2 = do_lasso(
                    mX,
                    y,
                    lasso_nlambda,
                    lasso_positive,
                    lasso_complexities,
                    fil_idx,
                    fil_complexities,
                    fil_names,
                )
                res.extend(res2)

        if method == "l0learn":
            os.makedirs(f"outputs/{tmp}/l0learn", exist_ok=True)
            path = f"outputs/{tmp}/l0learn"
            with open(f"{path}/{p}-features.out", "w") as f:
                print(f"Considering {N} features")
                print("L0Learn Method\n----------\n")
                res3 = do_l0learn(mX, y, fil_idx, fil_complexities, fil_names)
                res.extend(res3)

    df = pd.DataFrame.from_records(res)
    df.to_csv(f"{outpath}-k{max_complexity}-th{threshold}.csv", index=False)
