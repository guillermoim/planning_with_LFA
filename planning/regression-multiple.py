import argparse
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
from src.data_loader import data_loader
from src.key_features import get_key_features
#from test_eliminate_correlated import eliminate

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
    parser.add_argument("--step", type=int, default=5)

    args = parser.parse_args()

    filepath = args.filepath
    step = args.step

    X, y, names, complexities = None, None, None, None

    with open(f"{filepath}","r") as file:
        
        for i, subfile in enumerate(file.readlines()):
            if i == 0:
                X, y, names, complexities = data_loader(subfile)
            else:
                X_, y_, _, _ = data_loader(subfile)
                X = 
                continue


    X, y, names, complexities = data_loader(filepath)
    y = np.abs(y)
    # names, complexities, X = eliminate(X, names, complexities, tol=0.9)

    # Select seed for reproducibility
    np.random.seed(10)
    random.seed(10)

    # MAX COMPLEXITY FEATURES
    # c_max = 8

    omp_interval = np.arange(1, 10)
    lasso_positive = False
    lasso_complexities = False
    lasso_nlambda = 20

    # methods = ["l0learn", "lasso", "omp"]
    methods = ["l0learn"]

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

    base_features = get_key_features(filepath)
    base_features = list(map(names.index, base_features))

    for i in range(p):
        if i in base_features:
            mask[i] = True
        else:
            mask[i] = False

    # Adjust base features back to 0. This applies to base features
    # defined as concept_distance that can take value of \infty
    # but from a more abstract level they should be 0.

    idxs = np.where(X[:, base_features] > 20)
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

    for fi in range(0, len(nfs), step):

        mask[perm[fi : fi + step]] = True
        N = np.sum(mask)
        print(f"Considering {N} features.")
        mX = X[:, mask]

        fil_names = filter_list(names, mask)
        fil_complexities = filter_list(complexities, mask)
        fil_idx = filter_list(range(0, p), mask)

        if "omp" in methods:
            os.makedirs(f"outputs/{filename}/omp", exist_ok=True)
            path = f"outputs/{filename}/omp"
            with open(f"{path}/{p}-features.out", "w") as f:
                sys.stdout = f
                print(f"Considering {N} features")
                print("----------\nomp Method\n----------\n")
                sim1 = do_OMP(mX, y, omp_interval, fil_idx, fil_complexities, fil_names)
                simulations.extend(sim1)

        if "lasso" in methods:
            os.makedirs(f"outputs/{filename}/lasso", exist_ok=True)
            path = f"outputs/{filename}/lasso"
            with open(f"{path}/{p}-features.out", "w") as f:
                sys.stdout = f
                print(f"Considering {N} features")
                print("Lasso Method\n----------\n")
                sim2 = do_lasso(
                    mX,
                    y,
                    lasso_nlambda,
                    lasso_positive,
                    lasso_complexities,
                    fil_idx,
                    fil_complexities,
                    fil_names,
                )
                simulations.extend(sim2)

        if "l0learn" in methods:
            os.makedirs(f"outputs/{filename}/l0learn", exist_ok=True)
            path = f"outputs/{filename}/l0learn"
            with open(f"{path}/{p}-features.out", "w") as f:
                sys.stdout = f
                print(f"Considering {N} features")
                print("L0Learn Method\n----------\n")
                sim3 = do_l0learn(mX, y, fil_idx, fil_complexities, fil_names)
                simulations.extend(sim3)

        sys.stdout = old_stdout

    df = pd.DataFrame.from_records(simulations)
    df.to_csv(f"runs/{filename}.csv", index=False)
