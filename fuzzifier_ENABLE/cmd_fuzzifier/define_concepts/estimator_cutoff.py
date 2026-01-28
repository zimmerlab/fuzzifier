import warnings
import numpy as np
import pandas as pd


### inputs:
# cutoff: pandas dataframe of cutoff values between fuzzy sets per feature/sample
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



### inputs:
# mtx: pandas dataframe of crisp value matrix
# percents: list of expected proportion of samples per fuzzy set
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



### inputs:
# mean: list of mean values for Gaussian curves
# valueRange: [minimum, maximum] of matrix
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



### inputs:
# partial: partial fuzzy concept consisting of cutoffs and slopes (only for trapezoidal functions)
# functionType: type of membership functions (trapezoidal or Gaussian)
# valueRange: [minimum, maximum] of matrix
def getFullConcept (partial, functionType, valueRange):
    if functionType == "trap":
        center = [-np.inf] + partial[:, 0].tolist () + [np.inf]
        slope = [0] + partial[:, 1].tolist () + [0]
        finalFC = np.round ([[center[i] - slope[i], center[i] + slope[i], center[i + 1] - slope[i + 1], center[i + 1] + slope[i + 1]]
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



### inputs:
# ticks: pandas dataframe of width or value percentiles
# numFuzzySets: number of fuzzy sets
# percents: list of expectation proportion per fuzzy set (default: list ())
# slope: list of slope values per intersection between trapezoidal membership functions (default: list ())
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
    concept = {feature: np.array ([[ticks.loc[feature, int (i)] for i in params] for params in concept]) for feature in ticks.index}
    return concept



### inputs:
# ticks: pandas dataframe of width or value percentiles
# numFuzzySets: number of fuzzy sets
# percents: list of expectation proportion per fuzzy set (default: list ())
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



### inputs:
# mtx: pandas dataframe of crisp value matrix
# numFuzzySets: number of fuzzy sets
# functionType: type of membership functions (trapezoidal or Gaussian)
# fuzzyBy: whether to perform feature-wise ("feature") or sample-wise ("sample") fuzzification or overall ("matrix") (default: "feature")
# cutoffBy: whether to define cutoffs by proportion ("proportion") or width ("width") per fuzzy set (default: "proportion")
# percents: list of expected percents of samples with highest membership in each fuzzy set or expected proportion in the value range per row  (default: list ())
# slope: list of slope values per cutoff (default: list ())
# labelValues: list of values to be labelled in the downstream analysis (default: list ())
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


