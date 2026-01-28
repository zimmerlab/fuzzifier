import numpy as np
import pandas as pd
from scipy import stats, signal, optimize


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



def getSegments (mtx, labels, valueRange):
    tmp = mtx.replace (labels + [-np.inf, np.inf], np.nan)
    widthTicks = pd.DataFrame ({"min": np.floor (tmp.min (axis = 1, skipna = True)) - 1,
                                 "max": np.ceil (tmp.max (axis = 1, skipna = True)) + 1})
    widthTicks.loc["ALL"] = {"min": valueRange[0], "max": valueRange[1]}
    widthTicks.loc[np.isnan (widthTicks["min"]), "min"] = valueRange[0]
    widthTicks.loc[np.isnan (widthTicks["max"]), "max"] = valueRange[1]
    widthTicks = widthTicks.apply (lambda x: np.linspace (x["min"], x["max"], 1001),
                                   axis = 1, result_type = "expand").round (3)
    widthTicks = widthTicks.rename (columns = {widthTicks.columns[i]: i for i in range (1001)})
    propTicks = tmp.quantile (np.linspace (0, 1, 1001), axis = 1, numeric_only = True).T
    propTicks.loc["ALL"] = tmp.melt ()["value"].dropna ().quantile (np.linspace (0, 1, 1001))
    propTicks = propTicks.rename (columns = {propTicks.columns[i]: i for i in range (1001)})
    propTicks = round (propTicks, 3)
    return widthTicks, propTicks



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



def getDefaultConcepts (mtx, numFS, labels):
    concepts = {"fixed": dict (), "width": dict (), "prop": dict ()}; percents = [1 / numFS] * numFS
    dummy = pd.DataFrame (mtx.melt ()["value"].replace (labels + [-np.inf, np.inf], np.nan).dropna ()).T
    cutoff = estimateCutoff (dummy, percents).loc["value"]
    slope = cutoff.diff ().iloc[1:].min () / 4
    concepts["fixed"]["trap"] = np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)], 3).T
    concepts["fixed"]["gauss"] = np.round (cutoff.tolist ()[1:-1], 3)
    dummy = pd.DataFrame ({"percents": range (101)}).T
    cutoff = estimateCutoff (dummy, percents).loc["percents"]
    slope = cutoff.diff ().iloc[1:].min () / 4
    concepts["width"]["trap"] = np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T
    concepts["width"]["gauss"] = np.round (cutoff.tolist ()[1:-1])
    concepts["prop"]["trap"] = np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T
    concepts["prop"]["gauss"] = np.round (cutoff.tolist ()[1:-1])
    return concepts



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
        finalFC = np.round ([[center[i] - slope[i], center[i] + slope[i],
                              center[i + 1] - slope[i + 1], center[i + 1] + slope[i + 1]]
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



def getDensityMaxima (values, bwFct = 1):
    kernel = stats.gaussian_kde (values); kernel.set_bandwidth (bw_method = bwFct * kernel.factor)
    density = kernel (values)
    with np.errstate (divide = "ignore", invalid = "ignore"):
        density = (density - density.min ()) / (density.max () - density.min ())
    density = pd.DataFrame ({"value": values.values, "density": density}).sort_values ("value").drop_duplicates ()
    modes = density.iloc[signal.argrelmax (density["density"].to_numpy ())[0]].drop_duplicates ().sort_values ("value")
    return modes



def fitMode (values, bwFct = 1):
    if len (values) < 2:
        return values.mean (), np.nan
    try:
        kernel = stats.gaussian_kde (values); kernel.set_bandwidth (bw_method = bwFct * kernel.factor)
        density = pd.DataFrame ({"value": values.values, "density": kernel (values)}).sort_values ("value").drop_duplicates ()
        modes = density.iloc[signal.argrelmax (density["density"].to_numpy ())[0]].drop_duplicates ().sort_values ("value")
    except (ValueError, np.linalg.LinAlgError):
        return values.mean (), np.nan
    if modes.empty:
        return values.mean (), values.std ()
    modes.loc["mean"] = {"value": values.mean (), "density": kernel ([values.mean ()])[0]}
    modes = modes.sort_values ("value"); meanIdx = list (modes.index).index ("mean")
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
    mu = modes.iloc[modeIdx, 0]; sigma = np.sqrt (values[values < mu].std () ** 2 + values[values > mu].std () ** 2)
    #lb = np.floor (modes.iloc[modeIdx, 0] * 1e3) / 1e3; ub = np.ceil (modes.iloc[modeIdx, 0] * 1e3) / 1e3
    #ub = ub + 1e-3 if lb == ub else ub
    #res, cov = optimize.curve_fit (lambda x, m, s: stats.norm.pdf (x, loc = m, scale = s), density["value"], density["density"],
    #                                bounds = [(lb, -np.inf), (ub, np.inf)])
    return mu, sigma



def calculateOverlap (mu1, sigma1, mu2, sigma2):
    if np.isnan (sigma1) or np.isnan (sigma2) or sigma1 * sigma2 == 0:
        overlap = 2; middle = np.nan
    elif sigma1 == sigma2:
        if mu1 < mu2:
            overlap = 2 * stats.norm.cdf ((mu1 + mu2) / 2, loc = mu2, scale = sigma2)
        else:
            overlap = 2 * stats.norm.cdf ((mu1 + mu2) / 2, loc = mu1, scale = sigma1)
        overlap += (mu1 == mu2); middle = np.nan
    else:
        delta = sigma1 * sigma2 * np.sqrt ((mu1 - mu2) ** 2 + 2 * (sigma2 ** 2 - sigma1 ** 2) * np.log (sigma2 / sigma1))
        intersection = np.array ([((mu1 * sigma2 ** 2 - mu2 * sigma1 ** 2) - delta) / (sigma2 ** 2 - sigma1 ** 2),
                                  ((mu1 * sigma2 ** 2 - mu2 * sigma1 ** 2) + delta) / (sigma2 ** 2 - sigma1 ** 2)])
        middle = intersection[(intersection > min (mu1, mu2)) & (intersection < max (mu1, mu2))]
        if middle.size == 0:
            if sigma1 < sigma2:
                AUC = stats.norm.cdf (intersection, loc = mu2, scale = sigma2)
            else:
                AUC = stats.norm.cdf (intersection, loc = mu1, scale = sigma1)
            overlap = np.abs (AUC[1] - AUC[0]) + 1; middle = np.nan
        else:
            AUC1 = stats.norm.cdf (middle[0], loc = mu1, scale = sigma1)
            AUC2 = stats.norm.cdf (middle[0], loc = mu2, scale = sigma2)
            if mu1 < mu2:
                overlap = (1 - AUC1) + AUC2
            else:
                overlap = AUC1 + (1 - AUC2)
            middle = middle[0]
    return overlap, middle



def findSubcluster (values, prefix, defaultRange, bwFct = 1, maxIteration = 50):
    try:
        kernel = stats.gaussian_kde (values); kernel.set_bandwidth (bw_method = bwFct * kernel.factor)
    except (ValueError, np.linalg.LinAlgError):
        return np.array (list ()), pd.Series ("", index = values.index, dtype = str)
    density = pd.DataFrame ({"value": values.values, "density": kernel (values)}, index = values.index)
    density = density.sort_values ("value").drop_duplicates ()
    modes = density.iloc[signal.argrelmax (density["density"].to_numpy ())[0]].drop_duplicates ().sort_values ("value")["value"].to_numpy ()
    intersection = [defaultRange[0]] + (modes[:-1] + np.ediff1d (modes) / 2).tolist () + [defaultRange[1]]
    subclusters = pd.DataFrame (index = values.index, dtype = str); numIter = 0
    while numIter < maxIteration:
        parameters = list (); subclusters[f"iteration_{numIter}"] = ""
        for i in range (len (intersection) - 1):
            subsetVal = values[(values > intersection[i]) & (values <= intersection[i + 1])]
            center = [m for m in modes if m > intersection[i] and m <= intersection[i + 1]]
            if len (center) == 0 or len (subsetVal) / len (values) < 0.1:
                continue
            elif len (center) == 1:
                center = center[0]
            else:
                center = sum (center) / len (center)
            lb = np.floor (center * 1e3) / 1e3; ub = np.ceil (center * 1e3) / 1e3; ub = ub + 1e-3 if lb == ub else ub
            try:
                kernel = stats.gaussian_kde (subsetVal); kernel.set_bandwidth (bw_method = bwFct * kernel.factor)
            except (ValueError, np.linalg.LinAlgError):
                continue
            res, _ = optimize.curve_fit (lambda x, m, s: stats.norm.pdf (x, loc = m, scale = s), subsetVal, kernel (subsetVal),
                                         bounds = [(lb, -np.inf), (ub, np.inf)])
            subclusters.loc[subsetVal.index, f"iteration_{numIter}"] = f"{prefix}_{i}"; parameters.append (np.round (res, 3).tolist ())
        newIntersection = list ()
        for i in range (len (parameters) - 1):
            mu1, sigma1 = parameters[i]; mu2, sigma2 = parameters[i + 1]; overlap, middle = calculateOverlap (mu1, sigma1, mu2, sigma2)
            if np.isnan (middle):
                continue
            if overlap < 0.3:
                newIntersection.append (middle)
        if numIter > 0:
            identical = np.array ([(subclusters[f"iteration_{i}"] == subclusters[f"iteration_{numIter}"]).all () for i in range (numIter - 1)])
            if identical.any ():
                break
        intersection = np.array ([defaultRange[0]] + newIntersection + [defaultRange[1]]); numIter += 1
    parameters = np.array (parameters); subclusters = pd.Series (subclusters[f"iteration_{numIter}"], index = subclusters.index).replace (np.nan, "")
    return parameters, subclusters



def getLines (fuzzyConcept, colours):
    lines = list (); numFuzzySets = fuzzyConcept.shape[0]
    if len (colours) == 0:
        colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                   "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
                   "blue", "orange", "green", "red", "purple",
                   "brown", "pink", "gray", "olive", "cyan"]
    for idx in range (numFuzzySets):
        params = fuzzyConcept[idx]
        lines += [(params[0], params[1]), (0, 1), colours[idx],
                  (params[1], params[2]), (1, 1), colours[idx],
                  (params[2], params[3]), (1, 0), colours[idx]]
    return lines



def getCurves (fuzzyConcept, valueRange, colours = list (), setPlateau = False):
    lines = list (); numFuzzySets = fuzzyConcept.shape[0]
    if len (colours) == 0:
        colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                   "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
                   "blue", "orange", "green", "red", "purple",
                   "brown", "pink", "gray", "olive", "cyan"]
    xValues = np.linspace (*valueRange, 1000)
    func = lambda x, p: np.exp (-(x - p[0]) ** 2 / (2 * p[1] ** 2))
    for idx in range (numFuzzySets):
        if setPlateau:
            if idx == 0:
                platform = (xValues <= fuzzyConcept[idx][0]).astype (int)
                yValues = platform + (1 - platform) * func (xValues, fuzzyConcept[idx])
            elif idx == numFuzzySets - 1:
                platform = (xValues >= fuzzyConcept[idx][0]).astype (int)
                yValues = platform + (1 - platform) * func (xValues, fuzzyConcept[idx])
            else:
                yValues = func (xValues, fuzzyConcept[idx])
        else:
            yValues = func (xValues, fuzzyConcept[idx])
        lines.append ([yValues, colours[idx]])
    return lines



def findSpecificCluster (markerStats, allSets, maxNumCluster):
    levelFS = dict (zip (allSets, [str (x) for x in range (len (allSets))]))
    markerAnnot = markerStats.copy (); markerAnnot["isSpecific"] = False
    markerAnnot["mainFS"] = markerAnnot["mainFS"].replace (levelFS).astype (int)
    for feature in set (markerAnnot["feature"]):
        numLevels = markerAnnot.loc[markerAnnot["feature"] == feature].value_counts ("mainFS")
        numLevels = numLevels.sort_index (ascending = False).cumsum ()
        specificLevel = list (numLevels[numLevels <= maxNumCluster].index)
        markerAnnot.loc[(markerAnnot["feature"] == feature) & markerAnnot["mainFS"].isin (specificLevel), "isSpecific"] = True
    return markerAnnot


