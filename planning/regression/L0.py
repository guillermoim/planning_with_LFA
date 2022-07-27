import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import numpy as np
from .score import score
from sklearn.metrics import mean_squared_error, mean_absolute_error


def quickL0(X, y, names, complexities, maxSupport=8):
    
    n, p = np.shape(X)
    l0learn = rpackages.importr("L0Learn")
    # Next lines: convert X to R matrix.
    Xf = X.flatten("F")
    Xr = robjects.IntVector(Xf)
    XR = robjects.r["matrix"](Xr, nrow=X.shape[0])
    yr = robjects.FloatVector(y)
    rlearn = robjects.r["L0Learn.fit"]

    l0fit = rlearn(XR, yr, algorithm="CDPSI", penalty="L0",
                    maxSuppSize=maxSupport,)  # lambdaGrid=lambdas

    rprint = robjects.r["print"]
    res = rprint(l0fit)
    vlambda = res[0]
    # vgamma = res[1]
    vsupp = res[2]
    yt = np.squeeze(np.array(y))

    solution_weights = np.zeros((1, p+1))

    max_score = -1
    min_supp = 10

    solution = None

    for l in range(len(vlambda)):
        
        # First fit the model
        rfunc = robjects.r["coef"]
        res = rfunc(l0fit, vlambda[l], gamma=0)
        # Then, get the prediction over training data
        rfunc = robjects.r["predict"]
        res1 = rfunc(l0fit, XR, vlambda[l], gamma=0)
        yp = np.squeeze(np.array(res1.slots["x"]))
        # and get the current score
        #   (vidx represents the non-zero entries of the features)
        vidx = res.slots["i"]
        current_score = score(yt, yp)

        if not (current_score >= max_score and 0 < vsupp[l] < min_supp):
            continue
        
        asnum = robjects.r["as.numeric"]
        weights = asnum(res)

        solution_weights[0, -1] = weights[0]
        feat_complexities = []
        feat_names = []

        for i in range(1, len(vidx)):

            solution_weights[0, vidx[i]-1] = weights[i]
            feat_complexities.append(complexities[vidx[i] - 1])
            feat_names.append(names[vidx[i] - 1])
        
            solution = (solution_weights, feat_complexities, feat_names)
            max_score = current_score
            min_supp = vsupp[l]
    

    return solution
        

def do_l0learn(X, y, idx, complexities, names):
    # Check https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html
    # Check https://github.com/rpy2/rpy2 (https://rpy2.github.io/doc/v3.4.x/html/introduction.html#r-packages)
    n, p = np.shape(X)
    l0learn = rpackages.importr("L0Learn")
    # print(X.shape)
    # print('Converting features to R matrix')
    Xf = X.flatten("F")
    # print('\tflatten done')
    Xr = robjects.IntVector(Xf)
    # print('\tvector created')
    XR = robjects.r["matrix"](Xr, nrow=X.shape[0])
    # print('\tmatrix created')
    yr = robjects.FloatVector(y)
    rlearn = robjects.r["L0Learn.fit"]

    # laux = robjects.FloatVector([1.5e-9, 4e-20, 3e-38])
    # lambdas = robjects.r("list()")
    # lambdas.rx2[1] = laux
    l0fit = rlearn(
        XR,
        yr,
        algorithm="CDPSI",
        penalty="L0",
        maxSuppSize=8,  # lambdaGrid=lambdas
    )
    rprint = robjects.r["print"]
    res = rprint(l0fit)
    vlambda = res[0]
    # vgamma = res[1]
    vsupp = res[2]
    yt = np.squeeze(np.array(y))

    res_ = []

    for l in range(len(vlambda)):
        
        if vlambda[l] == 0:
            continue

        print(f"Lambda={vlambda[l]}")
        rfunc = robjects.r["coef"]
        res = rfunc(l0fit, vlambda[l], gamma=0)
        weights = res.slots["x"]
        vidx = res.slots["i"]

        asnum = robjects.r["as.numeric"]
        weights = asnum(res)

        print(f"\tIntercept\t{weights[0]}")

        feat_weights = [weights[0]]
        feat_complexities = []
        feat_names = []

        for i in range(0, len(vidx)):
            if vidx[i] == 0:
                continue
            print(
                f"\tFeat {idx[vidx[i]-1]}\t(c={complexities[vidx[i]-1]})\tweight {weights[vidx[i]]:.4}:\t{names[vidx[i]-1]}"
            )

            feat_weights.append(round(weights[vidx[i]], 4))
            feat_complexities.append(complexities[vidx[i] - 1])
            feat_names.append(names[vidx[i] - 1])

        rfunc = robjects.r["predict"]
        res = rfunc(l0fit, XR, vlambda[l], gamma=0)
        yp = np.squeeze(np.array(res.slots["x"]))
        print(f"Score {score(yt, yp)}")
        supp_size = vsupp[l]

        entry = {
            "method": "L0",
            "parameter": vlambda[l],
            "max_support_size": vsupp[l],
            "total_features": p,
            "score": score(yp, y),
            "mae": mean_absolute_error(y, yp),
            "mse": mean_squared_error(y, yp),
            "features_names": feat_names,
            "feature_weights": feat_weights,
            "feature_complexities": feat_complexities,
            "support_size": supp_size
        }

        res_.append(entry)

    return res_
