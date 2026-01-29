import numpy as np
import pandas as pd
from scipy import stats, signal


def getMtxSummary (mtx, labels = list (), noiseRep = None):
    if mtx.empty or noiseRep is None:
        nRow = 0; nCol = 0; mtxSize = 1; minimum = 0; maximum = 0
        numNoise = 0; numNA = 0; numNegInf = 0; numInf = 0; numZero = 0
    else:
        nRow = mtx.shape[0]; nCol = mtx.shape[1]; mtxSize = nRow * nCol
        minimum = round (mtx.replace ([-np.inf] + noiseRep + labels, np.nan).min (axis = None, skipna = True), 3)
        maximum = round (mtx.replace ([np.inf] + noiseRep + labels, np.nan).max (axis = None, skipna = True), 3)
        numNoise = ((mtx == noiseRep[0]) | (mtx == noiseRep[1])).sum ().sum ()
        numNA = np.isnan (mtx).sum ().sum ()
        numNegInf = (~np.isfinite (mtx) & (mtx < 0)).sum ().sum ()
        numInf = (~np.isfinite (mtx) & (mtx > 0)).sum ().sum ()
        numZero = (mtx == 0).sum ().sum ()
    summary = pd.DataFrame ({"statement": ["features/rows", "samples/columns", "minimum", "maximum",
                                           "noise", "NaN", "-inf", "+inf", "zero"],
                             "number": [nRow, nCol, minimum, maximum,
                                        numNoise, numNA, numNegInf, numInf, numZero],
                             "percentage": ["/", "/", "/", "/",
                                            "{:.1%}".format (numNoise / mtxSize),
                                            "{:.1%}".format (numNA / mtxSize),
                                            "{:.1%}".format (numNegInf / mtxSize),
                                            "{:.1%}".format (numInf / mtxSize),
                                            "{:.1%}".format (numZero / mtxSize)]})
    return summary


def estimateStep (minimum, maximum):
    return 10 ** (round (np.log10 (maximum - minimum)) - 2)


def getIntersection (concept, typeFS, valueRange):
    if typeFS == "trap":
        intersection = concept[:-1, -2:].mean (axis = 1).tolist ()
    elif typeFS == "gauss":
        intersection = [(concept[i, 0] * concept[i + 1, 1] + concept[i + 1, 0] * concept[i, 1]) / (concept[i, 1] + concept[i + 1, 1])
                        for i in range (concept.shape[0] - 1)]
    else:
        raise ValueError
    intersection = [valueRange[0]] + intersection + [valueRange[1]]
    return intersection


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
    q = [int (100 * i) for i in q[:-1]]
    cutoff = round (pd.DataFrame ([np.linspace (valueRange.loc[idx, "min"] + 1, valueRange.loc[idx, "max"] - 1, 101)[q]
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


def getFinalConcept (concept, typeFS, valueRange):
    if typeFS == "trap":
        center = [-np.inf] + concept[:, 0].tolist () + [np.inf]
        slope = [0] + concept[:, 1].tolist () + [0]
        finalFC = np.round ([[center[i] - slope[i], center[i] + slope[i], center[i + 1] - slope[i + 1], center[i + 1] + slope[i + 1]]
                             for i in range (concept.shape[0] + 1)], 3)
        finalFC[0, 0] = valueRange[0]; finalFC[0, 1] = valueRange[0]
        finalFC[-1, 2] = valueRange[1]; finalFC[-1, 3] = valueRange[1]
    elif typeFS == "gauss":
        cutoff = [valueRange[0]] + concept.tolist () + [valueRange[1]]
        center = cutoff[1:] - np.diff (cutoff) / 2
        finalFC = np.round ([center, estimateSigma (center.tolist (), valueRange)], 3).T
    else:
        raise ValueError
    return finalFC


def fitMode (values, bwFct = 1, useFit = True):
    finite_values = values[np.isfinite (values)]; mu = finite_values.mean ()
    if useFit:
        if np.isnan (mu) or len (values) < 2:
            return np.nan, np.nan
        try:
            kernel = stats.gaussian_kde (finite_values); kernel.set_bandwidth (bw_method = bwFct * kernel.factor)
            density = pd.DataFrame ({"value": finite_values, "density": kernel (finite_values)}).sort_values ("value").drop_duplicates ()
            modes = density.iloc[signal.argrelmax (density["density"].to_numpy ())[0]].drop_duplicates ()
            modes.loc["mean"] = {"value": mu, "density": kernel ([mu])[0]}; modes = modes.sort_values ("value")
        except (ValueError, np.linalg.LinAlgError):
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
        mu = modes.iloc[modeIdx, 0]
        sigma1 = finite_values[finite_values < mu].std (); sigma1 = 0 if np.isnan (sigma1) else sigma1
        sigma2 = finite_values[finite_values > mu].std (); sigma2 = 0 if np.isnan (sigma2) else sigma2
        sigma = np.sqrt (sigma1 ** 2 + sigma2 ** 2)
    else:
        sigma = finite_values.std ()
    if sigma != 0 and round (sigma, 3) == 0:
        sigma = 1e-3
    return round (mu, 3), round (sigma, 3)


def getDefaultConcept (numFS_side):
    numFS = 2 * numFS_side + 1
    coords = [i + overlap for i in np.linspace (-numFS, numFS, numFS + 1) for overlap in [-0.5, 0.5]]
    trap = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFS + 1)], 3).tolist ()
    trap[0][0] = trap[0][1]; trap[-1][3] = trap[-1][2]; trap = np.round (trap, 3)
    gauss = trap[:, [1, 2]].mean (axis = 1)
    return trap, gauss
    

