import numpy as np
import pandas as pd
from scipy import stats, signal, optimize


### inputs:
# mtx: pandas dataframe of crisp value matrix
# numFuzzySets: number of fuzzy sets
# fuzzyBy: whether to perform feature-wise ("feature") or sample-wise ("sample") fuzzification or overall ("matrix") (default: "feature")
# widthFactor: factor for width scaling (default: 1)
# slopeFactor: percent of slope width in each trapezoidal membership function (default: 0.5)
# bwFactor: band width factor (default: 1)
# useOptimize: whether to use scipy optimization for Gaussian curve fitting (default: False)
def estimatorByDefault (mtx, numFuzzySets, fuzzyBy = "feature", widthFactor = 1, slopeFactor = 0.5, bwFactor = 1, useOptimize = False):
    if fuzzyBy == "feature":
        finite_data = mtx.mask (~np.isfinite (mtx))
    elif fuzzyBy == "sample":
        finite_data = mtx.mask (~np.isfinite (mtx)).T
    elif fuzzyBy == "matrix":
        finite_data = mtx.mask (~np.isfinite (mtx)).melt ()[["value"]].T
    else:
        raise ValueError
    centerIdx = int (numFuzzySets / 2); fuzzyConcepts = dict ()
    for feature in finite_data.index:
        values = finite_data.loc[feature].dropna (); mu = values.mean ()
        if np.isnan (mu) or len (values) < 2:
            mu = 0; sigma = 1
        else:
            try:
                kernel = stats.gaussian_kde (values); kernel.set_bandwidth (bw_method = bwFactor * kernel.factor)
                density = pd.DataFrame ({"value": values, "density": kernel (values)}).sort_values ("value").drop_duplicates ()
                modes = density.iloc[signal.argrelmax (density["density"].to_numpy ())[0]].drop_duplicates ()
                modes.loc["mean"] = {"value": mu, "density": kernel ([mu])[0]}; modes = modes.sort_values ("value")
            except np.linalg.LinAlgError:
                modes = pd.DataFrame ({"value": mu, "density": 0}, index = ["mean"])
            meanIdx = list (modes.index).index ("mean")
            if modes.shape[0] == 1:
                modeIdx = 0
            elif meanIdx == 0:
                modeIdx = 1
            elif meanIdx == modes.shape[0] - 1:
                modeIdx = modes.shape[0] - 2
            else:
                modeIdx = modes.reset_index (drop = True)["density"].idxmax ()
                if np.abs (meanIdx - modeIdx) > 1:
                    modeIdx = modes.reset_index (drop = True).loc[[meanIdx - 1, meanIdx + 1]].sort_values ("density").index[1]
            if useOptimize:
                lb = np.floor (modes.iloc[modeIdx, 0] * 1e3) / 1e3; ub = np.ceil (modes.iloc[modeIdx, 0] * 1e3) / 1e3
                ub = ub + 1e-3 if lb == ub else ub
                res, _ = optimize.curve_fit (lambda x, m, s: stats.norm.pdf (x, loc = m, scale = s), density["value"], density["density"],
                                             bounds = [(lb, -np.inf), (ub, np.inf)])
                mu = res[0]; sigma = res[1]
            else:
                mu = modes.iloc[modeIdx, 0]
                sigma1 = values[values < mu].std (); sigma1 = 0 if np.isnan (sigma1) else sigma1
                sigma2 = values[values > mu].std (); sigma2 = 0 if np.isnan (sigma2) else sigma2
                sigma = np.sqrt (sigma1 ** 2 + sigma2 ** 2)
            if np.isnan (sigma) or round (sigma, 3) == 0:
                sigma = 1
        coords = [mu + widthFactor * (i + overlap) * sigma for i in np.linspace (-numFuzzySets, numFuzzySets, numFuzzySets + 1)
                  for overlap in [-slopeFactor, slopeFactor]]
        concept = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFuzzySets + 1)], 3).tolist ()
        left = np.floor (min (values.min (), concept[0][2])) - 1; left = -6 if np.isnan (left) else left
        right = np.ceil (max (values.max (), concept[-1][1])) + 1; right = 6 if np.isnan (right) else right
        concept[0][0] = left; concept[0][1] = left; concept[-1][2] = right; concept[-1][3] = right
        concept[centerIdx] = [round (mu, 3), round (sigma, 3)]; fuzzyConcepts[feature] = concept
    if fuzzyBy == "matrix":
        fuzzyConcepts = {feature: fuzzyConcepts["value"] for feature in mtx.index}
    return fuzzyConcepts


