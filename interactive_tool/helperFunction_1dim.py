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



def _getIntersection (concept):
    intersection = list ()
    for idx in range (1, len (concept)):
        prev_typeFS = "trap" if len (concept[idx - 1]) == 4 else "gauss"
        next_typeFS = "trap" if len (concept[idx]) == 4 else "gauss"
        if prev_typeFS == next_typeFS:
            if prev_typeFS == "trap":
                a, b = concept[idx][:2]; c, d = concept[idx - 1][2:]
                try:
                    coord = (a ** 2 - d ** 2) / (a - b + c - d)
                except ZeroDivisionError:
                    coord = (a + b + c + d) / 4
            else:
                mu1, sigma1 = concept[idx - 1]; mu2, sigma2 = concept[idx]
                try:
                    coord = (mu1 * sigma2 + mu2 * sigma1) / (sigma1 + sigma2)
                except ZeroDivisionError:
                    coord = (mu1 + mu2) / 2
        else:
            if prev_typeFS == "trap":
                p = concept[idx - 1][2:]; mu, sigma = concept[idx]
                values = pd.DataFrame ({"x": np.linspace (*p, int ((p[1] - p[0]) / 1e-3))})
                values["trap"] = p[0] if p[0] == p[1] else (values["x"] - p[1]) / (p[0] - p[1])
            else:
                p = concept[idx][:2]; mu, sigma = concept[idx - 1]
                values = pd.DataFrame ({"x": np.linspace (*p, int ((p[1] - p[0]) / 1e-3))})
                values["trap"] = p[0] if p[0] == p[1] else (values["x"] - p[0]) / (p[1] - p[0])
            values["gauss"] = mu if sigma == 0 else np.exp (-(values["x"] - mu) ** 2 / (2 * sigma ** 2))
            diff = values["trap"] - values["gauss"]
            if diff.empty:
                coord = p[0]
            elif diff[0] == 0:
                coord = values.loc[0, "x"]
            else:
                i = (diff < 0).sum () if diff[0] < 0 else (diff > 0).sum ()
                coord = values.loc[[i - 1, i], "x"].mean ()
        intersection.append (coord)
    return intersection



def getPercentage (values, concept, labels = list (), minLevel = -np.inf, maxLevel = np.inf):
    raw = values.replace (labels, np.nan); raw = raw.mask ((raw <= minLevel) | (raw >= maxLevel))
    cutoffs = [-np.inf] + _getIntersection (concept) + [np.inf]
    percent = [((raw >= cutoffs[idx]) & (raw < cutoffs[idx + 1])).mean () for idx in range (len (cutoffs) - 1)]
    return percent



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
    newCutoff = cutoff.copy ()
    for feature in cutoff.index:
        cVal = newCutoff.loc[feature]
        overlapIdx = np.where (cVal.diff () == 0)[0]
        if len (overlapIdx) > 0:
            overlapIdx = np.insert (overlapIdx, 0, overlapIdx[0] - 1)
            nonOverlapCutoff = np.linspace (cVal.iloc[overlapIdx[0] - 1], cVal.iloc[overlapIdx[-1] + 1], len (overlapIdx) + 2)
            newCutoff.loc[feature, newCutoff.columns[overlapIdx]] = nonOverlapCutoff[1:-1]
    return newCutoff



def _estimateSigma (mean, valueRange):
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
        finalFC = np.round ([center, _estimateSigma (center.tolist (), valueRange)], 3).T
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



def getLines (fuzzyConcept, cutoffs, colors):
    lines = list (); curves = list (); numFuzzySets = len (fuzzyConcept)
    if cutoffs[0] >= cutoffs[1]:
        return lines, curves
    if len (colors) == 0:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
                  "blue", "orange", "green", "red", "purple",
                  "brown", "pink", "gray", "olive", "cyan"]
    for idx in range (numFuzzySets):
        params = fuzzyConcept[idx]
        if len (params) == 2:
            if params[1] > 0:
                xValues = np.linspace (*cutoffs, 1000)
                yValues = np.exp (-(xValues - params[0]) ** 2 / (2 * params[1] ** 2))
                curves.append ([xValues, yValues, colors[idx]])
            else:
                continue
        elif len (params) == 4:
            if cutoffs[1] <= params[0]:
                continue
            elif cutoffs[1] > params[0] and cutoffs[1] < params[1]:
                with np.errstate (divide = "ignore", invalid = "ignore"):
                    y_cutoffs = [(cutoffs[0] - params[0]) / (params[1] - params[0]), (cutoffs[1] - params[0]) / (params[1] - params[0])]
                lines += [(max (cutoffs[0], params[0]), cutoffs[1]), (max (y_cutoffs[0], 0), y_cutoffs[1]), colors[idx]]
            elif cutoffs[1] >= params[1] and cutoffs[1] <= params[2]:
                with np.errstate (divide = "ignore", invalid = "ignore"):
                    y_cutoffs = [(cutoffs[0] - params[0]) / (params[1] - params[0]), 1]
                if cutoffs[0] < params[1]:
                    lines += [(max (cutoffs[0], params[0]), params[1]), (max (y_cutoffs[0], 0), 1), colors[idx],
                              (params[1], cutoffs[1]), (1, 1), colors[idx]]
                else:
                    lines += [(cutoffs[0], cutoffs[1]), (1, 1), colors[idx]]
            else:
                if cutoffs[0] < params[1]:
                    with np.errstate (divide = "ignore", invalid = "ignore"):
                        y_cutoffs = [(cutoffs[0] - params[0]) / (params[1] - params[0]), (cutoffs[1] - params[3]) / (params[2] - params[3])]
                    lines += [(max (cutoffs[0], params[0]), params[1]), (max (y_cutoffs[0], 0), 1), colors[idx],
                              (params[1], params[2]), (1, 1), colors[idx],
                              (params[2], min (cutoffs[1], params[3])), (1, max (y_cutoffs[1], 0)), colors[idx]]
                elif cutoffs[0] >= params[1] and cutoffs[0] <= params[2]:
                    with np.errstate (divide = "ignore", invalid = "ignore"):
                        y_cutoffs = [1, (cutoffs[1] - params[3]) / (params[2] - params[3])]
                    lines += [(cutoffs[0], params[2]), (1, 1), colors[idx],
                              (params[2], min (cutoffs[1], params[3])), (1, max (y_cutoffs[1], 0)), colors[idx]]
                else:
                    with np.errstate (divide = "ignore", invalid = "ignore"):
                        y_cutoffs = [(cutoffs[0] - params[3]) / (params[2] - params[3]), (cutoffs[1] - params[3]) / (params[2] - params[3])]
                    lines += [(cutoffs[0], min (cutoffs[1], params[3])), (y_cutoffs[0], max (y_cutoffs[1], 0)), colors[idx]]
        else:
            raise ValueError
    return lines, curves



def generateOutputFromConstraint (featureList, pctConcept, ticks, widths, direction, basicInfo, typeList, names, colors):
    num = len (typeList); output = dict ()
    featureMap = dict (zip (featureList, featureList if direction == "feature" else ["ALL"] * len (featureList)))
    for feature in featureList:
        params = list (); concept = list (); featureInfo = basicInfo.copy (); useFeature = featureMap[feature]
        minLevel = basicInfo.get ("MIN-NOISE", -np.inf); maxLevel = basicInfo.get ("MAX-NOISE", np.inf)
        xMin = np.floor (ticks.loc[feature, 0]) - 1; xMax = np.ceil (ticks.loc[feature, 1000]) + 1
        for i in range (num):
            if typeList[i] == "trap":
                coords = ticks.loc[useFeature, pctConcept[i]].round (3).tolist ()
                if i == 0:
                    coords[0] = xMin; coords[1] = xMin
                elif i == num - 1:
                    coords[2] = xMax; coords[3] = xMax
                if any ([~np.isfinite (x) for x in coords]):
                    featureInfo["number_fuzzy_sets"] = 0
                    break
                params.append (coords); concept.append ([coords, "trapezoidal", colors[i]])
            else:
                center = round (ticks.loc[useFeature, int (pctConcept[i][0])], 3)
                if not np.isfinite (center):
                    featureInfo["number_fuzzy_sets"] = 0
                    break
                params.append ([center, round (pctConcept[i][1] * widths[useFeature], 3)])
                concept.append ([params[-1], "Gaussian", colors[i]])
        featureInfo.update (dict (zip (names, concept)))
        percent = dict (zip (names, getPercentage (ticks.loc[useFeature].copy (), params, labels = list (), minLevel = minLevel, maxLevel = maxLevel)))
        for name in names:
            featureInfo[name].append (round (percent[name], 5))
        output[feature] = featureInfo.copy ()
    return output



def generateOutputFromDefault (featureList, zConcept, fit, allRanges, ticks, direction, basicInfo, typeList, names, colors):
    num = len (typeList); output = dict ()
    featureMap = dict (zip (featureList, featureList if direction == "feature" else ["ALL"] * len (featureList)))
    for feature in featureList:
        mu, sigma = fit.loc[feature]
        params = list (); concept = list (); featureInfo = basicInfo.copy (); useFeature = featureMap[feature]
        minLevel = basicInfo.get ("MIN-NOISE", -np.inf); maxLevel = basicInfo.get ("MAX-NOISE", np.inf)
        for i in range (num):
            if typeList[i] == "trap":
                coords = [round (mu + sigma * zConcept[i][0], 3), round (mu + sigma * zConcept[i][1], 3),
                          round (mu + sigma * zConcept[i][2], 3), round (mu + sigma * zConcept[i][3], 3)]
                if i == 0:
                    xMin = np.floor (min (allRanges.loc[useFeature, "min"], coords[2])) - 1
                    coords[0] = xMin; coords[1] = xMin
                elif i == num - 1:
                    xMax = np.ceil (max (allRanges.loc[useFeature, "max"], coords[1])) + 1
                    coords[2] = xMax; coords[3] = xMax
                if any ([~np.isfinite (x) for x in coords]):
                    featureInfo["number_fuzzy_sets"] = 0
                    break
                params.append (coords); concept.append ([coords, "trapezoidal", colors[i]])
            else:
                center = round (mu + sigma * zConcept[i][0], 3)
                if not np.isfinite (center):
                    featureInfo["number_fuzzy_sets"] = 0
                    break
                params.append ([center, round (zConcept[i][1] * sigma, 3)]); concept.append ([params[-1], "Gaussian", colors[i]])
        featureInfo.update (dict (zip (names, concept)))
        percent = dict (zip (names, getPercentage (ticks.loc[useFeature].copy (), params, labels = list (), minLevel = minLevel, maxLevel = maxLevel)))
        for name in names:
            featureInfo[name].append (round (percent[name], 5))
        output[feature] = featureInfo.copy ()
    return output
    

