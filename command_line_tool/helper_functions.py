import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import optimize, signal, stats


def _fitMode (values, bwFct = 1, useFit = True, useOptimize = False):
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
        if useOptimize:
            lb = np.floor (modes.iloc[modeIdx, 0] * 1e3) / 1e3; ub = np.ceil (modes.iloc[modeIdx, 0] * 1e3) / 1e3
            ub = ub + 1e-3 if lb == ub else ub
            res, _ = optimize.curve_fit (lambda x, m, s: stats.norm.pdf (x, loc = m, scale = s), density["value"], density["density"],
                                         bounds = [(lb, -np.inf), (ub, np.inf)])
            mu = res[0]; sigma = res[1]
        else:
            mu = modes.iloc[modeIdx, 0]
            sigma1 = finite_values[finite_values < mu].std (); sigma1 = 0 if np.isnan (sigma1) else sigma1
            sigma2 = finite_values[finite_values > mu].std (); sigma2 = 0 if np.isnan (sigma2) else sigma2
            sigma = np.sqrt (sigma1 ** 2 + sigma2 ** 2)
    else:
        sigma = finite_values.std ()
    return round (mu, 3), round (sigma, 3)



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



def getSubarea (mu, sigma, concept, minLevel = -np.inf, maxLevel = np.inf):
    noisePct = np.diff (stats.norm.cdf ([-np.inf, minLevel, maxLevel, np.inf], loc = mu, scale = sigma))[[0, 2]]
    pct = np.diff (stats.norm.cdf ([-np.inf] + _getIntersection (concept) + [np.inf], loc = mu, scale = sigma)).tolist ()
    tmp = list (); tmpInv = list ()
    for idx in range (len (pct)):
        tmp.append (max (0, pct[idx] - noisePct[0])); noisePct[0] = max (0, noisePct[0] - pct[idx])
        tmpInv.append (max (0, pct[-(idx + 1)] - noisePct[1])); noisePct[1] = max (0, noisePct[1] - pct[-(idx + 1)])
    percent = [min (tmp[idx], tmpInv[-(idx + 1)]) for idx in range (len (pct))]
    return percent



def _adjustBorder (concept, minimum, maximum):
    newConcept = concept.copy ()
    if len (concept[0]) == 4:
        border = np.floor (min (concept[0][2], minimum)) - 1; newConcept[0][0] = border; newConcept[0][1] = border
    if len (concept[-1]) == 4:
        border = np.ceil (max (concept[-1][1], maximum)) + 1; newConcept[-1][2] = border; newConcept[-1][3] = border
    return newConcept



def getConcept (values, method, consType, basicInfo, numFS, renameFS, labels,
                minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                useFit = False, useOptimize = False, bwFct = 1,
                refConcept = list (), consValue = list (),
                widthFct = 1, slopeFct = 0.5, centerIdx = 0):
    masked = values.replace (labels, np.nan).dropna (); info = basicInfo.copy ()
    typeFS_dict = {2: "Gaussian", 4: "trapezoidal"}
    defaultColors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                     "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
                     "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
                     "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"]
    if masked.empty:
        if method == "constraint" and consType == "fixed":
            info["MIN-NOISE"] = round (minLevelCons, 3) if np.isfinite (minLevelCons) else "-Infinity"
            info["MAX-NOISE"] = round (maxLevelCons, 3) if np.isfinite (maxLevelCons) else "Infinity"
            for idx in range (numFS):
                info[renameFS[idx]] = [refConcept[idx], typeFS_dict[len (refConcept[idx])], defaultColors[idx], 0]
        else:
            info["number_fuzzy_sets"] = 0
    else:
        minLevel = max (minLevelCons, -np.inf if minLevelPct == 0 else masked.quantile (minLevelPct))
        maxLevel = min (maxLevelCons, np.inf if maxLevelPct == 1 else masked.quantile (maxLevelPct))
        masked = masked.mask ((masked <= minLevel) | (masked >= maxLevel)).dropna ()
        info["MIN-NOISE"] = round (minLevel, 3) if np.isfinite (minLevel) else "-Infinity"
        info["MAX-NOISE"] = round (maxLevel, 3) if np.isfinite (maxLevel) else "Infinity"
        if method == "constraint":
            if consType == "fixed":
                percent = getPercentage (values, refConcept, labels = labels, minLevel = minLevel, maxLevel = maxLevel)
                for idx in range (numFS):
                    info[renameFS[idx]] = [refConcept[idx], typeFS_dict[len (refConcept[idx])], defaultColors[idx], round (percent[idx], 5)]
            elif consType == "proportion" and not masked.empty:
                percentiles = masked.quantile (consValue).round (3); std = masked.std (); typeFS = list (); concept = list ()
                for idx in range (numFS):
                    typeFS.append (typeFS_dict[len (refConcept[idx])])
                    if typeFS[-1] == "trapezoidal":
                        concept.append ([percentiles[x] for x in refConcept[idx]])
                    else:
                        concept.append ([percentiles[refConcept[idx][0]], round (refConcept[idx][1] * std, 3)])
                concept = _adjustBorder (concept, masked.min (), masked.max ())
                percent = getPercentage (masked.quantile (np.linspace (0, 1, 1001)), concept, labels = list (),
                                         minLevel = minLevel, maxLevel = maxLevel)
                for idx in range (numFS):
                    info[renameFS[idx]] = [concept[idx], typeFS[idx], defaultColors[idx], round (percent[idx], 5)]
            elif consType == "z-score" and not masked.empty:
                mu, sigma = _fitMode (masked, bwFct = bwFct, useFit = useFit, useOptimize = useOptimize)
                if (not (np.isnan (mu) and np.isnan (sigma))) and sigma > 0:
                    typeFS = list (); concept = list (); scaleFct = list ()
                    for idx in range (numFS):
                        typeFS.append (typeFS_dict[len (refConcept[idx])])
                        if typeFS[-1] == "trapezoidal":
                            concept.append ([round (mu + sigma * z, 3) for z in refConcept[idx]])
                        else:
                            concept.append ([round (mu + sigma * refConcept[idx][0], 3), round (refConcept[idx][1] * sigma, 3)])
                            scaleFct.append (refConcept[idx][1])
                    concept = _adjustBorder (concept, masked.min (), masked.max ())
                    scaleFct = 1 if len (scaleFct) == 0 else sum (scaleFct) / len (scaleFct)
                    percent = getSubarea (mu, scaleFct * sigma, concept, minLevel = minLevel, maxLevel = maxLevel)
                    for idx in range (numFS):
                        info[renameFS[idx]] = [concept[idx], typeFS[idx], defaultColors[idx], round (percent[idx], 5)]
            else:
                info["number_fuzzy_sets"] = 0
        elif method == "default":
            mu, sigma = _fitMode (masked, bwFct = bwFct, useFit = useFit, useOptimize = useOptimize)
            if (not (np.isnan (mu) and np.isnan (sigma))) and sigma > 0:
                coords = [mu + widthFct * (i + overlap) * sigma for i in np.linspace (-numFS, numFS, numFS + 1) for overlap in [-slopeFct, slopeFct]]
                concept = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFS + 1)], 3).tolist ()
                concept[centerIdx] = [round (mu, 3), round (widthFct * sigma, 3)]
                concept = _adjustBorder (concept, masked.min (), masked.max ())
                percent = getSubarea (mu, widthFct * sigma, concept, minLevel = minLevel, maxLevel = maxLevel)
                for idx in range (numFS):
                    info[renameFS[idx]] = [concept[idx], typeFS_dict[len (concept[idx])], defaultColors[idx], round (percent[idx], 5)]
            else:
                info["number_fuzzy_sets"] = 0
        else:
            raise ValueError
    return info



def fuzzify (rawValues, concept, renameLabels = dict ()):
    if not concept:
        return pd.DataFrame (dtype = float)
    numFS = concept["number_fuzzy_sets"]; labels = concept.get ("label_values", list ())
    minLevel = concept.get ("MIN-NOISE", -np.inf); maxLevel = concept.get ("MAX-NOISE", np.inf)
    if numFS == 0:
        return pd.DataFrame (dtype = float)
    masked = rawValues.replace (labels, np.nan).to_numpy (); memberships = pd.DataFrame (index = rawValues.index, dtype = float)
    for key in concept.keys ():
        if key in ["number_fuzzy_sets", "label_values", "MIN-NOISE", "MAX-NOISE"]:
            continue
        params, typeFS, _, _ = concept[key]
        if typeFS == "trapezoidal":
            if params[0] == params[1] and params[1] == params[2] and params[2] == params[3]:
                memberships[key] = 0
            elif memberships.empty:
                if params[2] == params[3]:
                    memberships[key] = (masked <= params[2]).astype (float)
                else:
                    memberships[key] = np.clip ((params[3] - masked) / (params[3] - params[2]), a_min = 0, a_max = 1)
            elif memberships.shape[1] == numFS - 1:
                if params[0] == params[1]:
                    memberships[key] = (masked >= params[0]).astype (float)
                else:
                    memberships[key] = np.clip ((params[0] - masked) / (params[0] - params[1]), a_min = 0, a_max = 1)
            else:
                if params[0] == params[1]:
                    left = np.zeros (len (masked))
                else:
                    left = (masked < params[1]).astype (int) * np.clip ((params[0] - masked) / (params[0] - params[1]), a_min = 0, a_max = np.inf)
                middle = ((masked >= params[1]) & (masked <= params[2])).astype (float)
                if params[2] == params[3]:
                    right = np.zeros (len (masked))
                else:
                    right = (masked > params[2]).astype (int) * np.clip ((params[3] - masked) / (params[3] - params[2]), a_min = 0, a_max = np.inf)
                memberships[key] = left + middle + right
        elif typeFS == "Gaussian":
            if np.isnan (params[0]) or np.isnan (params[1]) or params[1] == 0:
                memberships[key] = 0
            else:
                if memberships.empty:
                    platform = (masked <= params[0]).astype (float)
                elif memberships.shape[1] == numFS - 1:
                    platform = (masked >= params[0]).astype (float)
                else:
                    platform = np.zeros (len (masked))
                memberships[key] = platform + (1 - platform) * np.exp (-(masked - params[0]) ** 2 / (2 * params[1] ** 2))
        else:
            raise ValueError
    masked = pd.Series (masked, index = rawValues.index)
    if np.isfinite (maxLevel):
        outliers = (masked >= maxLevel)
        memberships.loc[outliers] = 0; memberships.insert (0, "MAX-NOISE", 0); memberships.loc[outliers, "MAX-NOISE"] = 1
    if np.isfinite (minLevel):
        outliers = (masked <= minLevel)
        memberships.loc[outliers] = 0; memberships.insert (0, "MIN-NOISE", 0); memberships.loc[outliers, "MIN-NOISE"] = 1
    for val in labels[::-1]:
        if np.isnan (val):
            outliers = np.isnan (rawValues)
        else:
            outliers = (rawValues == val)
        name = renameLabels.get (val, str (val))
        memberships.loc[outliers] = 0; memberships.insert (0, name, 0); memberships.loc[outliers, name] = 1
    return memberships



def getFuzzy (mtx, direction, concepts, defaultName, renameLabels, outputDir, deriveConcepts = False,
              minLevelCons = -np.inf, minLevelPct = 0, maxLevelCons = np.inf, maxLevelPct = 1):
    const = {"-infinity": -np.inf, "-inf": -np.inf,
             "+infinity": np.inf, "+inf": np.inf, "infinity": np.inf, "inf": np.inf,
             "nan": np.nan, "na": np.nan, "zero": 0}
    if mtx.lower ().endswith ("tsv"):
        with open (mtx) as f:
            samples = f.readline ().strip ("\n").split ("\t")[1:]
            features = [line.split ("\t")[0] for line in f.readlines ()]
    elif mtx.lower ().endswith ("h5ad"):
        adata = sc.read_h5ad (mtx, backed = "r")
        features = list (adata.var_names); samples = list (adata.obs_names)
    else:
        raise TypeError
    if deriveConcepts:
        param_keys = ["value_type", "number_fuzzy_sets", "label_values",
                      "fit_Gaussian_curve", "use_scipy_optimization", "band_width_factor"]
        consType = concepts.get ("value_type", "fixed"); consValue = list (); bwFct = concepts.get ("band_width_factor", 1)
        outputLabels = concepts.get ("label_values", list ())
        labels = [const.get (x.lower ()) if isinstance (x, str) else x for x in outputLabels]
        renameFS = [key for key in concepts.keys () if key not in param_keys]
        numFS = concepts.get ("number_fuzzy_sets", len (renameFS)); concept_cons = [concepts[FS][0] for FS in renameFS]
        useFit = concepts.get ("fit_Gaussian_curve", False); useOptimize = concepts.get ("use_scipy_optimization", False)
        if consType == "proportion":
            consValue = set ()
            for FS in renameFS:
                params, typeFS, _ = concepts[FS]
                if typeFS == "trapezoidal":
                    consValue |= set (params)
                else:
                    consValue |= {params[0]}
        basicInfo = {"number_fuzzy_sets": numFS, "label_values": labels}
        if mtx.lower ().endswith ("tsv"):
            with open (mtx) as f:
                values = [[np.nan if x == "" else float (x) for x in line.strip ("\n").split ("\t")[1:]] for line in f.readlines ()[1:]]
            values = pd.Series (sum (values, list ())).round (5)
        if mtx.lower ().endswith ("h5ad"):
            values = pd.Series (np.array (adata[samples].X.data).reshape ((1, -1))[0]).round (5)
        default = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                              minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                              refConcept = concept_cons, consValue = consValue,
                              useFit = False, useOptimize = False, bwFct = bwFct)
        defaultOutput = default.copy (); defaultOutput["label_values"] = outputLabels; allConcepts = {defaultName: defaultOutput}
        del values
    else:
        default = concepts.get (defaultName, dict ())
        default["label_values"] = [const.get (x.lower ()) if isinstance (x, str) else x for x in default.get ("label_values", list ())]
    if direction == "sample":
        maxSplit = 2
        for sample in samples:
            if mtx.lower ().endswith ("tsv"):
                if sample == samples[-1]:
                    with open (mtx) as f:
                        values = pd.Series ([line.strip ("\n").split ("\t")[-1] for line in f.readlines ()[1:]], index = features)
                else:
                    with open (mtx) as f:
                        values = pd.Series ([line.strip ("\n").split ("\t", maxsplit = maxSplit)[-2] for line in f.readlines ()[1:]],
                                            index = features)
                values[values == ""] = np.nan; values = values.astype (float).round (5); maxSplit += 1
            else:
                values = adata[sample].to_df ().loc[sample].round (5)
            if deriveConcepts:
                concept = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                                      minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                      refConcept = concept_cons, consValue = consValue,
                                      useFit = useFit, useOptimize = useOptimize, bwFct = bwFct)
                if concept["number_fuzzy_sets"] == 0:
                    concept = default.copy ()
                else:
                    outputConcept = concept.copy (); outputConcept["label_values"] = outputLabels; allConcepts[sample] = outputConcept
            else:
                concept = concepts.get (sample, default).copy ()
                concept["label_values"] = [const.get (x.lower ()) if isinstance (x, str) else x for x in concept.get ("label_values", list ())]
            memberships = fuzzify (values, concept, renameLabels = renameLabels).round (3)
            if not memberships.empty:
                memberships.to_csv (os.path.join (outputDir, f"fuzzyValues_{sample}.tsv"), sep = "\t")
    else:
        if mtx.lower ().endswith ("tsv"):
            with open (mtx) as f:
                _ = f.readline ()
                for feature in features:
                    values = pd.Series ([np.nan if x == "" else float (x) for x in f.readline ().strip ("\n").split ("\t")[1:]],
                                        index = samples).round (5)
                    if direction == "feature":
                        if deriveConcepts:
                            concept = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                                                  minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                                  refConcept = concept_cons, consValue = consValue,
                                                  useFit = useFit, useOptimize = useOptimize, bwFct = bwFct)
                            if concept["number_fuzzy_sets"] == 0:
                                concept = default.copy ()
                            else:
                                outputConcept = concept.copy (); outputConcept["label_values"] = outputLabels; allConcepts[feature] = outputConcept
                        else:
                            concept = concepts.get (feature, default).copy ()
                            concept["label_values"] = [const.get (x.lower ()) if isinstance (x, str) else x for x in concept.get ("label_values", list ())]
                    else:
                        concept = default.copy ()
                    memberships = fuzzify (values, concept, renameLabels = renameLabels).round (3)
                    if not memberships.empty:
                        memberships.to_csv (os.path.join (outputDir, f"fuzzyValues_{feature}.tsv"), sep = "\t")
        if mtx.lower ().endswith ("h5ad"):
            for feature in features:
                values = adata[:, feature].to_df ()[feature].round (5)
                if direction == "feature":
                    if deriveConcepts:
                        concept = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                                              minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                              refConcept = concept_cons, consValue = consValue,
                                              useFit = useFit, useOptimize = useOptimize, bwFct = bwFct)
                        if concept["number_fuzzy_sets"] == 0:
                            concept = default.copy ()
                        else:
                            outputConcept = concept.copy (); outputConcept["label_values"] = outputLabels; allConcepts[feature] = outputConcept
                    else:
                        concept = concepts.get (feature, default).copy ()
                        concept["label_values"] = [const.get (x.lower ()) if isinstance (x, str) else x for x in concept.get ("label_values", list ())]
                else:
                    concept = default.copy ()
                memberships = fuzzify (values, concept, renameLabels = renameLabels).round (3)
                if not memberships.empty:
                    memberships.to_csv (os.path.join (outputDir, f"fuzzyValues_{feature}.tsv"), sep = "\t")
    if deriveConcepts:
        with open (os.path.join (outputDir, "concepts_detailed.json"), "w", encoding = "utf-8") as f:
            json.dump (allConcepts, f, ensure_ascii = False, indent = 4, allow_nan = True)


