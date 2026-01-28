import numpy as np
import pandas as pd


### inputs:
# mtx: pandas dataframe of crisp value matrix
# functionType: type of membership functions (trapezoidal or Gaussian)
# functionParams: list of lists or numpy array of fixed function parameters or percents for derivation of function parameters
# fuzzyBy: whether to perform feature-wise ("feature") or sample-wise ("sample") fuzzification or overall ("matrix") (default: "feature")
# paramBy: whether to define function parameters by given values ("fix") or proportion ("percentile") per fuzzy set (default: "fix")
# labelValues: list of values to be labelled in the downstream analysis (default: list ())
def estimatorByParameter (mtx, functionType, functionParams, fuzzyBy = "feature", paramBy = "fix", labelValues = list ()):
    if isinstance (functionParams, list) and all ([isinstance (x, list) for x in functionParams]):
        params = np.array (functionParams)
    elif type (functionParams).__module__ == np.__name__:
        params = functionParams.copy ()
    else:
        raise TypeError
    match fuzzyBy:
        case "feature":
            matrix = mtx.replace (labelValues, np.nan)
        case "sample":
            matrix = mtx.replace (labelValues, np.nan).T
        case "matrix":
            matrix = pd.DataFrame (mtx.replace (labelValues, np.nan).melt ()["value"]).T
        case _:
            raise ValueError
    if functionType == "trapezoidal":
        if params.shape[1] != 4:
            raise ValueError
        if paramBy == "fix":
            allFC = dict (); paramsMin = params[0, 2]; paramsMax = params[-1, 1]
            for feature in matrix.index:
                minimum = min (paramsMin, np.floor (matrix.loc[feature].min (skipna = True)) - 1)
                maximum = max (paramsMax, np.ceil (matrix.loc[feature].max (skipna = True)) + 1)
                params[0, 0] = minimum; params[0, 1] = minimum; params[-1, 2] = maximum; params[-1, 3] = maximum
                allFC[feature] = params.tolist ()
        elif paramBy == "percentile":
            allFC = {feature: np.round ([matrix.loc[feature].dropna ().quantile (p).tolist () for p in params], 3)
                     for feature in matrix.index}
            for feature in matrix.index:
                if np.isnan (allFC[feature]).all ():
                    allFC[feature] = np.zeros (params.shape)
                else:
                    minimum = np.floor (matrix.loc[feature].min (skipna = True)) - 1
                    maximum = np.ceil (matrix.loc[feature].max (skipna = True)) + 1
                    allFC[feature][0, 0], allFC[feature][0, 1] = minimum, minimum
                    allFC[feature][-1, -2], allFC[feature][-1, -1] = maximum, maximum
        else:
            raise ValueError
    elif functionType == "gauss":
        if params.shape[1] != 2:
            raise ValueError
        if paramBy == "fix":
            allFC = {feature: params.tolist () for feature in matrix.index}
        elif paramBy == "percentile":
            allFC = dict (); fct = 2 * np.sqrt (2 * np.log (2))
            missIdx = np.where ([not isinstance (p, (int, float)) or np.isnan (p) or not np.isfinite (p) for p in params[:, 1]])[0]
            for feature in matrix.index:
                center = np.round (matrix.loc[feature].dropna ().quantile (params[:, 0]).tolist (), 3)
                if np.isnan (center).all ():
                    allFC[feature] = np.zeros (params.shape)
                else:
                    std = params[:, 1].copy ()
                    for idx in missIdx:
                        if idx == 0:
                            std[idx] = (center[idx + 1] - center[idx]) / fct
                        else:
                            std[idx] = (center[idx] - center[idx - 1]) / fct
                    allFC[feature] = np.round ([center, std.astype (float)], 3).T
        else:
            raise ValueError
    else:
        raise ValueError
    if fuzzyBy == "matrix":
        allFC = {feature: allFC["value"] for feature in mtx.index}
    return allFC


