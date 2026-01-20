import warnings
import numpy as np
import pandas as pd
from scipy import stats, signal


def fixOverlapCutoff (cutoff):
    newCutoff = cutoff.copy ()
    for feature in cutoff.index:
        cVal = newCutoff.loc[feature]
        overlapIdx = np.where (cVal.diff () == 0)[0]
        if len (overlapIdx) > 0:
            overlapIdx = np.insert (overlapIdx, 0, overlapIdx[0] - 1)
            nonOverlapCutoff = np.linspace (cVal.iloc[overlapIdx[0] - 1], cVal.iloc[overlapIdx[-1] + 1], len (overlapIdx) + 2)
            newCutoff.loc[feature, newCutoff.columns[overlapIdx]] = nonOverlapCutoff[1:-1]
    return newCutoff



def estimateCutoff (mtx, percents):
    numFuzzySets = len (percents)
    valueRange = pd.DataFrame ({"min": np.floor (mtx.min (axis = 1, skipna = True)) - 1,
                                "max": np.ceil (mtx.max (axis = 1, skipna = True)) + 1})
    q = np.array (percents).cumsum ()
    if round (q[-1], 3) != 1:
        raise ValueError
    q = [int (1000 * i) for i in q[:-1]]
    cutoff = round (pd.DataFrame ([np.linspace (valueRange.loc[idx, "min"] + 1, valueRange.loc[idx, "max"] - 1, 1001)[q]
                                   for idx in mtx.index],
                                  index = mtx.index, columns = [f"C{idx}" for idx in range (1, numFuzzySets)]), 3)
    cutoff.insert (0, "C0", valueRange["min"]); cutoff[f"C{numFuzzySets}"] = valueRange["max"]
    cutoff = fixOverlapCutoff (cutoff)
    return cutoff



def estimateSigma (mean, valueRange):
    center = [valueRange[0]] + mean + [valueRange[1]]; width = list ()
    fct1 = np.sqrt (2 * np.log (2)); fct2 = np.sqrt (6 * np.log (10))
    for idx in range (len (mean)):
        sigma = min (center[idx + 2] - center[idx + 1], center[idx + 1] - center[idx]) / fct1
        if len (mean) > 2:
            if idx < 2:
                sigma = min (sigma, (center[idx + 3] - center[idx + 1]) / fct2)
            elif idx + 4 > len (center):
                sigma = min (sigma, (center[idx + 1] - center[idx - 1]) / fct2)
            else:
                sigma = min (sigma, (center[idx + 3] - center[idx + 1]) / fct2,
                             (center[idx + 1] - center[idx - 1]) / fct2)
        width.append (round (sigma, 3))
    return width



def getFullConcept (partial, functionType, valueRange):
    if functionType == "trap":
        center = [-np.inf] + partial[:, 0].tolist () + [np.inf]
        slope = [0] + partial[:, 1].tolist () + [0]
        finalFC = np.round ([[center[i] - slope[i], center[i] + slope[i],
                              center[i + 1] - slope[i + 1], center[i + 1] + slope[i + 1]]
                             for i in range (partial.shape[0] + 1)], 3)
        finalFC[0, 0] = valueRange[0]; finalFC[0, 1] = valueRange[0]
        finalFC[-1, 2] = valueRange[1]; finalFC[-1, 3] = valueRange[1]
    elif functionType == "gauss":
        cutoff = [valueRange[0]] + partial.tolist () + [valueRange[1]]
        center = cutoff[1:] - np.diff (cutoff) / 2
        finalFC = np.round ([center, estimateSigma (center.tolist (), valueRange)], 3).T
    else:
        raise ValueError
    return finalFC



def estimateTrapezoidalConcept (ticks, numFuzzySets, percents = list (), slope = list ()):
    if len (percents) != numFuzzySets:
        if len (percents) > 0:
            warnings.warn ("Unequal number of proportions and fuzzy sets, returning to default (equal) mode.")
        percents = [1 / numFuzzySets] * numFuzzySets
    cutoff = estimateCutoff (pd.DataFrame ({"percents": range (1001)}).T, [i for i in percents]).loc["percents"]
    if len (slope) != numFuzzySets - 1:
        if len (slope) > 0:
            warnings.warn ("Unequal number of slopes and cutoffs, returning to default (equal) mode.")
        slope = [cutoff.diff ().iloc[1:].min () / 4] * (numFuzzySets - 1)
    partial = np.round ([cutoff.tolist ()[1:-1], [round (1000 * i) for i in slope]]).T
    concept = getFullConcept (partial, "trap", [0, 1000])
    concept = {feature: np.array ([[ticks.loc[feature, int (i)] for i in params] for params in concept])
               for feature in ticks.index}
    return concept



def estimateGaussianConcept (ticks, numFuzzySets, percents = list ()):
    if len (percents) != numFuzzySets:
        if len (percents) > 0:
            warnings.warn ("Unequal number of proportions and fuzzy sets, returning to default (equal) mode.")
        percents = [1 / numFuzzySets] * numFuzzySets
    cutoff = estimateCutoff (pd.DataFrame ({"percents": range (1001)}).T, percents).loc["percents"]
    concept = {feature: getFullConcept (np.array (ticks.loc[feature], [int (10 * C) for C in cutoff]), "gauss",
                                        [ticks.loc[feature, 0], ticks.loc[feature, 1000]])
               for feature in ticks.index}
    return concept



def estimatorByCutoff (mtx, numFuzzySets, functionType, fuzzyBy = "feature", cutoffBy = "proportion",
                       percents = list (), slope = list (), labelValues = list ()):
    match fuzzyBy:
        case "feature":
            matrix = mtx.replace (labelValues, np.nan)
        case "sample":
            matrix = mtx.replace (labelValues, np.nan).T
        case "matrix":
            matrix = pd.DataFrame (mtx.replace (labelValues, np.nan).melt ()["value"]).T
        case _:
            raise ValueError
    valueRange = [np.floor (matrix.min (axis = None, skipna = True)) - 1, np.ceil (matrix.max (axis = None, skipna = True)) + 1]
    match cutoffBy:
        case "proportion":
            ticks = matrix.quantile (np.linspace (0, 1, 1001), axis = 1, numeric_only = True).T
            ticks = ticks.round (3).rename (columns = {ticks.columns[i]: i for i in range (1001)})
            ticks[0] = np.floor (ticks[0]) - 1; ticks[1000] + np.ceil (ticks[1000]) + 1
        case "width":
            ticks = pd.DataFrame ({"min": np.floor (matrix.min (axis = 1, skipna = True)) - 1,
                                   "max": np.ceil (matrix.max (axis = 1, skipna = True)) + 1})
            ticks.loc[np.isnan (ticks["min"]), "min"] = valueRange[0]
            ticks.loc[np.isnan (ticks["max"]), "max"] = valueRange[1]
            ticks = ticks.apply (lambda x: np.linspace (x["min"], x["max"], 1001), axis = 1, result_type = "expand")
            ticks = ticks.round (3).rename (columns = {ticks.columns[i]: i for i in range (1001)})
        case _:
            raise ValueError
    match functionType:
        case "trapezoidal":
            allFC = estimateTrapezoidalConcept (ticks, numFuzzySets, percents = percents, slope = slope)
        case "gauss":
            allFC = estimateGaussianConcept (ticks, numFuzzySets, percents = percents)
    return allFC



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



def estimatorByDefault (mtx, numFuzzySets, widthFactor = 1, slopeFactor = 0.5, bwFactor = 1):
    finite_data = mtx.mask (~np.isfinite (mtx)); centerIdx = int (numFuzzySets / 2)
    fuzzyConcepts = dict ()
    for feature in mtx.index:
        values = finite_data.loc[feature].dropna (); mu = values.mean ()
        if np.isnan (mu):
            mu = 0
        if len (values) < 2:
            continue
        else:
            try:
                kernel = stats.gaussian_kde (values); kernel.set_bandwidth (bw_method = bwFactor * kernel.factor)
                density = pd.DataFrame ({"value": values, "density": kernel (values)}).sort_values ("value").drop_duplicates ()
                modes = density.iloc[signal.argrelmax (density["density"].to_numpy ())[0]].drop_duplicates ()
            except np.linalg.LinAlgError:
                modes = pd.DataFrame (columns = ["value", "density"], dtype = float)
            modes.loc["mean"] = {"value": mu, "density": kernel ([mu])[0]}; modes = modes.sort_values ("value")
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
            mu = modes.iloc[modeIdx, 0]
            sigma1 = values[values < mu].std (); sigma1 = 0 if np.isnan (sigma1) else sigma1
            sigma2 = values[values > mu].std (); sigma2 = 0 if np.isnan (sigma2) else sigma2
            sigma = np.sqrt (sigma1 ** 2 + sigma2 ** 2)
        if np.isnan (sigma) or sigma == 0:
            continue
        xRange = [np.floor (values.min ()) - 1, np.ceil (values.max ()) + 1]
        coords = [mu + widthFactor * (i + overlap) * sigma
                  for i in np.linspace (-numFuzzySets, numFuzzySets, numFuzzySets + 1)
                  for overlap in [-slopeFactor, slopeFactor]]
        concept = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFuzzySets + 1)], 3).tolist ()
        left = min (xRange[0], concept[0][3]); right = max (xRange[1], concept[-1][1])
        concept[0][0] = left; concept[0][1] = left; concept[-1][2] = right; concept[-1][3] = right
        concept[centerIdx] = [round (mu, 3), round (sigma, 3)]; fuzzyConcepts[feature] = concept
    return fuzzyConcepts


