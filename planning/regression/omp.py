from pyexpat import features
import numpy as np

from .score import score
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error, mean_absolute_error


def do_OMP(X, y, omp_interval, idx, complexities, names):
    # Check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
    _, p = np.shape(X)

    res = []

    for s in omp_interval:
        if s <= p:

            print(f"find {s} best features out of {p} features")

            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s, fit_intercept=False)
            omp.fit(X, y)
            coef = omp.coef_
            (idx_r,) = coef.nonzero()

            yp = omp.predict(X)

            feat_weights = []
            feat_complexities = []
            feat_names = []

            for i in idx_r:
                print(
                    f"\tFeat {idx[i]}\t(c={complexities[i]})\tweight {coef[i]:.2}:\t{names[i]}"
                )
                feat_weights.append(coef[i])
                feat_complexities.append(complexities[i])
                feat_names.append(names[i])

            print(f"Score {omp.score(X, y)}")

            entry = {
                "method": "omp",
                "parameter": s,
                "total_features": p,
                "score": omp.score(X, y),
                "mae": mean_absolute_error(y, yp),
                "mse": mean_squared_error(y, yp),
                "features_names": feat_names,
                "feature_weights": feat_weights,
                "feature_complexities": feat_complexities,
                "support_size": 0
                }

            res.append(entry)

    return res
