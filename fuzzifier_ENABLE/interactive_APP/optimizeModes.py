import numpy as np
import pandas as pd


### inputs:
# x: 1-dimensional numpy array of crisp values
# mean: mean of Gaussian function
# std: standard deviation of Gaussian function
def calcualteGaussian (x, mean, std):
    with np.errstate (divide = "ignore", invalid = "ignore"):
        y = np.exp (-((x - mean) ** 2) / (2 * (std ** 2))) / (std * np.sqrt (2 * np.pi))
    return y


### inputs:
# rawValues: pandas series of crisp values
# functionParams: numpy array of function parameters to be optimized
# fuzzyParams: dictionary of further parameters, including "addIndicator", "indicateValue", "namePrefix", "tolerance", "multiMatches", "valueRange", etc.
# mergeOverlapFS: whether fuzzy sets that strongly overlap with neighboring fuzzy sets are to be merged
# maxIteration: maximal number of iterations
def optimizeGaussian (rawValues, functionParams, fuzzyParams = dict (), mergeOverlapFS = False, maxIteration = 20):
    # Generate list of all fuzzy variables, excluding indicator fuzzy set.
    namePrefix = fuzzyParams.get ("namePrefix", "FS"); numFuzzySets = functionParams.shape[0]
    allFuzzySets = [f"{namePrefix}{idx}" for idx in range (1, numFuzzySets + 1)]
    functionParamsOpt = np.copy (functionParams)
    # Get range and index of raw values from input.
    if fuzzyParams.get ("addIndicator", False):
        index = rawValues[~rawValues.isin (fuzzyParams.get ("indicateValue", [0]))].index; crispValues = rawValues[index].to_numpy ()
    else:
        index = rawValues.index; crispValues = rawValues.to_numpy ()
    # Prepare parameters for the while-loop.
    bestResult = np.round (functionParamsOpt, 3); numIteration = 0
    while numIteration < maxIteration:
        # expectation: Evaluate new parameters by probability.
        memberships = pd.DataFrame (index = index, columns = allFuzzySets, dtype = float)
        for idx in range (len (allFuzzySets)):
            mean = functionParamsOpt[idx, 0]; std = functionParamsOpt[idx, 1]
            memberships[allFuzzySets[idx]] = pd.Series (calcualteGaussian (crispValues, mean, std), index = index)
        pctMainFuzzySets = memberships.idxmax (axis = 1, skipna = True).value_counts (normalize = True).sort_index ()
        pctMainFuzzySets = pctMainFuzzySets[pctMainFuzzySets > 0.01]
        probSampleFuzzySet = memberships * pctMainFuzzySets; probSample = probSampleFuzzySet.sum (axis = 1)
        # maximization: Derive new function parameters.
        mean = list (); std = list ()
        for FS in allFuzzySets:
            with np.errstate (divide = "ignore", invalid = "ignore"):
                prob = pd.Series ([probSampleFuzzySet.loc[sample, FS] / probSample[sample] for sample in memberships.index], index = memberships.index)
                mean.append ((prob * rawValues).sum () / prob.sum ())
                std.append (np.sqrt ((prob * (rawValues - mean[-1]) ** 2).sum () / prob.sum ()))
        functionParamsOpt = np.array ([mean, std]).T; functionParamsOpt = functionParamsOpt[~np.isnan (functionParamsOpt).any (axis = 1)]
        if mergeOverlapFS:
            overlappingFS = list (); numFuzzySets = functionParamsOpt.shape[0]
            for idx in range (numFuzzySets - 1):
                mean1 = functionParamsOpt[idx, 0]; std1 = functionParamsOpt[idx, 1]
                mean2 = functionParamsOpt[idx + 1, 0]; std2 = functionParamsOpt[idx + 1, 1]
                # Calculate coordinates for intersection between curves of neighboring membership functions.
                if std1 == std2:
                    if mean1 == mean2:
                        overlappingFS += [idx, idx + 1]
                        continue
                    else:
                        intersectX = np.array ([(mean1 + mean2) / 2, (mean1 + mean2) / 2])
                else:
                    with np.errstate (divide = "ignore", invalid = "ignore"):
                        delta = std1 * std2 * np.sqrt ((mean1 - mean2) ** 2 + 2 * (std1 ** 2 - std2 ** 2) * np.log (std1 / std2))
                    intersectX = np.array (sorted ([(mean2 * std1 ** 2 - mean1 * std2 ** 2 - delta) / (std1 ** 2 - std2 ** 2),
                                                    (mean2 * std1 ** 2 - mean1 * std2 ** 2 + delta) / (std1 ** 2 - std2 ** 2)]))
                intersectY = calcualteGaussian (intersectX, mean1, std1)
                minHeight = min (calcualteGaussian (mean1, mean1, std1), calcualteGaussian (mean2, mean2, std2))
                neighboring = max (intersectY) > 0.99 * minHeight or np.abs (max (intersectX) - mean1) < std1 / 3 or np.abs (mean2 - max (intersectX)) < std2 / 3
                inside = min (intersectY) > 0.05 * minHeight and intersectX[0] < mean1 < intersectX[1] and intersectX[0] < mean2 < intersectX[1]
                if neighboring or inside:
                    overlappingFS += [idx, idx + 1]
            if len (overlappingFS) != 0:
                mergeTwoIdx = list () # overlap of exactly 2 consecutive fuzzy sets
                mergeMoreIdx = list () # overlap of 3 or more consecutive fuzzy sets
                newParams = list ()
                for idx in sorted (set (overlappingFS)):
                    idxOccurrence = overlappingFS.count (idx)
                    if idxOccurrence == 1: # first or last index for overlap between consecutive fuzzy sets
                        if len (mergeMoreIdx) == 0: # not the last index for overlap between 3 or more consecutive fuzzy sets
                            mergeTwoIdx.append (idx)
                            if len (mergeTwoIdx) != 1: # last index of overlap between 2 consecutive fuzzy sets
                                newParams.append (functionParamsOpt[mergeTwoIdx].mean (axis = 0)) # merge 2 consecutive and overlapping fuzzy sets
                                mergeTwoIdx = list ()
                        else: # last index for overlap between 3 or more consecutive fuzzy sets
                            newParams.append (functionParamsOpt[list (set (mergeMoreIdx))].mean (axis = 0)) # merge 3 or more consecutive and overlapping fuzzy sets
                            mergeTwoIdx = list (); mergeMoreIdx = list ()
                    else: # middle index for overlap between 3 or more consecutive fuzzy sets
                        mergeMoreIdx += [idx - 1, idx, idx + 1]
                functionParamsOpt = np.vstack ([np.delete (functionParamsOpt, list (set (overlappingFS)), axis = 0), np.array (newParams)])
        functionParamsOpt = np.round (functionParamsOpt[functionParamsOpt[:, 0].argsort ()], 3)
        allFuzzySets = [f"{namePrefix}{i}" for i in range (1, functionParamsOpt.shape[0] + 1)]
        # termination: Check if the result converges.
        if functionParamsOpt.shape[0] == bestResult.shape[0]: # no membership functions removed
            if (functionParamsOpt == bestResult).all ():
                break
        bestResult = functionParamsOpt.copy ()
        numIteration += 1
    return bestResult


