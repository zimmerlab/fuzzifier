import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import optimize, signal, stats

sns.set_theme (style = "white", rc = {"axes.facecolor": (0, 0, 0, 0)})


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
            density = pd.DataFrame ({"value": finite_values, "density": 0}).sort_values ("value").drop_duplicates ()
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
            try:
                res, _ = optimize.curve_fit (lambda x, m, s: stats.norm.pdf (x, loc = m, scale = s), density["value"], density["density"],
                                             bounds = [(lb, -np.inf), (ub, np.inf)])
                mu = res[0]; sigma = res[1]
            except RuntimeError:
                mu = modes.iloc[modeIdx, 0]
                sigma1 = finite_values[finite_values < mu].std (); sigma1 = 0 if np.isnan (sigma1) else sigma1
                sigma2 = finite_values[finite_values > mu].std (); sigma2 = 0 if np.isnan (sigma2) else sigma2
                sigma = np.sqrt (sigma1 ** 2 + sigma2 ** 2)
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
                minLevelCons, minLevelPct, maxLevelCons, maxLevelPct, colorList,
                useFit = False, useOptimize = False, bwFct = 1,
                refConcept = list (), consValue = list (),
                widthFct = 1, slopeFct = 0.5, centerIdx = 0):
    masked = values.replace (labels, np.nan).dropna (); info = basicInfo.copy ()
    typeFS_dict = {2: "Gaussian", 4: "trapezoidal"}
    if masked.empty:
        if method == "constraint" and consType == "fixed":
            info["MIN-NOISE"] = round (minLevelCons, 3) if np.isfinite (minLevelCons) else "-Infinity"
            info["MAX-NOISE"] = round (maxLevelCons, 3) if np.isfinite (maxLevelCons) else "Infinity"
            for idx in range (numFS):
                info[renameFS[idx]] = [refConcept[idx], typeFS_dict[len (refConcept[idx])], colorList[idx], 0]
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
                    info[renameFS[idx]] = [refConcept[idx], typeFS_dict[len (refConcept[idx])], colorList[idx], round (percent[idx], 5)]
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
                    info[renameFS[idx]] = [concept[idx], typeFS[idx], colorList[idx], round (percent[idx], 5)]
            elif consType == "z-score" and not masked.empty:
                mu, sigma = _fitMode (masked, bwFct = bwFct, useFit = useFit, useOptimize = useOptimize)
                if (not (np.isnan (mu) and np.isnan (sigma))) and sigma > 0:
                    typeFS = list (); concept = list ()
                    for idx in range (numFS):
                        typeFS.append (typeFS_dict[len (refConcept[idx])])
                        if typeFS[-1] == "trapezoidal":
                            concept.append ([round (mu + sigma * z, 3) for z in refConcept[idx]])
                        else:
                            concept.append ([round (mu + sigma * refConcept[idx][0], 3), round (refConcept[idx][1] * sigma, 3)])
                    concept = _adjustBorder (concept, masked.min (), masked.max ())
                    percent = getSubarea (mu, sigma, concept, minLevel = minLevel, maxLevel = maxLevel)
                    for idx in range (numFS):
                        info[renameFS[idx]] = [concept[idx], typeFS[idx], colorList[idx], round (percent[idx], 5)]
            else:
                info["number_fuzzy_sets"] = 0
        elif method == "default":
            mu, sigma = _fitMode (masked, bwFct = bwFct, useFit = useFit, useOptimize = useOptimize)
            if (not (np.isnan (mu) and np.isnan (sigma))) and sigma > 0:
                coords = [mu + widthFct * (i + overlap) * sigma for i in np.linspace (-numFS, numFS, numFS + 1) for overlap in [-slopeFct, slopeFct]]
                concept = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFS + 1)], 3).tolist ()
                concept[centerIdx] = [round (mu, 3), round (widthFct * sigma, 3)]
                concept = _adjustBorder (concept, masked.min (), masked.max ())
                percent = getSubarea (mu, sigma, concept, minLevel = minLevel, maxLevel = maxLevel)
                for idx in range (numFS):
                    info[renameFS[idx]] = [concept[idx], typeFS_dict[len (concept[idx])], colorList[idx], round (percent[idx], 5)]
            else:
                info["number_fuzzy_sets"] = 0
        else:
            raise ValueError
    return info



def parseConcept (concept):
    const = {"-infinity": -np.inf, "-inf": -np.inf,
             "+infinity": np.inf, "+inf": np.inf, "infinity": np.inf, "inf": np.inf,
             "nan": np.nan, "na": np.nan, "zero": 0}
    newConcept = concept.copy ()
    newConcept["MIN-NOISE"] = concept.get ("MIN-NOISE", -np.inf); newConcept["MAX-NOISE"] = concept.get ("MAX-NOISE", np.inf)
    newConcept["MIN-NOISE"] = const.get (concept["MIN-NOISE"].lower (), -np.inf) if isinstance (concept["MIN-NOISE"], str) else concept["MIN-NOISE"]
    newConcept["MAX-NOISE"] = const.get (concept["MAX-NOISE"].lower (), np.inf) if isinstance (concept["MAX-NOISE"], str) else concept["MAX-NOISE"]
    newConcept["label_values"] = [const.get (x.lower ()) if isinstance (x, str) else x for x in concept.get ("label_values", list ())]
    return newConcept



def fuzzify (rawValues, concept, renameLabels = dict (), ignoreMinNoise = False, ignoreMaxNoise = False, scaleSum = True):
    if not concept:
        return pd.DataFrame (dtype = float), pd.DataFrame (dtype = float), np.nan
    numFS = concept["number_fuzzy_sets"]; labels = concept.get ("label_values", list ())
    minLevel = -np.inf if ignoreMinNoise else concept.get ("MIN-NOISE", -np.inf)
    maxLevel = np.inf if ignoreMaxNoise else concept.get ("MAX-NOISE", np.inf)
    if numFS == 0:
        return pd.DataFrame (dtype = float), pd.DataFrame (dtype = float), np.nan
    masked = rawValues.replace (labels, np.nan).to_numpy ()
    memberships = pd.DataFrame (index = rawValues.index, dtype = float); expectation = pd.Series (dtype = float)
    for key in concept.keys ():
        if key in ["number_fuzzy_sets", "label_values", "MIN-NOISE", "MAX-NOISE"]:
            continue
        params, typeFS, _, exp = concept[key]; expectation[key] = exp
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
                    left = (masked < params[1]).astype (int) * np.clip ((params[0] - masked) / (params[0] - params[1]),
                                                                        a_min = 0, a_max = np.inf)
                middle = ((masked >= params[1]) & (masked <= params[2])).astype (float)
                if params[2] == params[3]:
                    right = np.zeros (len (masked))
                else:
                    right = (masked > params[2]).astype (int) * np.clip ((params[3] - masked) / (params[3] - params[2]),
                                                                         a_min = 0, a_max = np.inf)
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
            memberships[key] = 0
    masked = pd.Series (masked, index = rawValues.index)
    if np.isfinite (maxLevel):
        outliers = (masked >= maxLevel); name = renameLabels.get ("MAX-NOISE", "MAX-NOISE")
        memberships.loc[outliers] = 0; memberships.insert (0, name, 0); memberships.loc[outliers, name] = 1
    if np.isfinite (minLevel):
        outliers = (masked <= minLevel); name = renameLabels.get ("MIN-NOISE", "MIN-NOISE")
        memberships.loc[outliers] = 0; memberships.insert (0, name, 0); memberships.loc[outliers, name] = 1
    for val in labels[::-1]:
        if np.isnan (val):
            outliers = np.isnan (rawValues)
        else:
            outliers = (rawValues == val)
        name = renameLabels.get (val, str (val))
        memberships.loc[outliers] = 0; memberships.insert (0, name, 0); memberships.loc[outliers, name] = 1
    if scaleSum:
        memberships = memberships.div (memberships.sum (axis = 1), axis = 0)
    observation = memberships.idxmax (axis = 1).value_counts (normalize = True)
    observation = pd.Series ([observation.get (idx, 0) for idx in expectation.index], index = expectation.index)
    deviation = np.sqrt (((observation - expectation) ** 2).mean ())
    return memberships, expectation, observation, deviation



def _getLines (params, cutoffs, colors):
    lines = list (); curves = list (); numFuzzySets = len (params)
    if cutoffs[0] >= cutoffs[1]:
        return lines, curves
    if len (colors) == 0:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
                  "blue", "orange", "green", "red", "purple",
                  "brown", "pink", "gray", "olive", "cyan"]
    for idx in range (numFuzzySets):
        p = params[idx]
        if len (p) == 2:
            if p[1] > 0:
                xValues = np.linspace (*cutoffs, 1000)
                yValues = np.exp (-(xValues - p[0]) ** 2 / (2 * p[1] ** 2))
                curves.append ([xValues, yValues, colors[idx]])
            else:
                continue
        elif len (p) == 4:
            if cutoffs[1] <= p[0] and p[0] != p[1]:
                continue
            elif cutoffs[1] > p[0] and cutoffs[1] < p[1]:
                y_cutoffs = [(cutoffs[0] - p[0]) / (p[1] - p[0]), (cutoffs[1] - p[0]) / (p[1] - p[0])]
                lines += [(max (cutoffs[0], p[0]), cutoffs[1]), (max (y_cutoffs[0], 0), y_cutoffs[1]), colors[idx]]
            elif cutoffs[1] >= p[1] and cutoffs[1] <= p[2]:
                if cutoffs[0] < p[1]:
                    y_cutoffs = [0 if p[0] == p[1] else (cutoffs[0] - p[0]) / (p[1] - p[0]), 1]
                    lines += [(max (cutoffs[0], p[0]), p[1]), (max (y_cutoffs[0], 0), 1), colors[idx],
                              (p[1], cutoffs[1]), (1, 1), colors[idx]]
                else:
                    lines += [(cutoffs[0], cutoffs[1]), (1, 1), colors[idx]]
            else:
                if cutoffs[0] < p[1]:
                    y_cutoffs = [0 if p[0] == p[1] else (cutoffs[0] - p[0]) / (p[1] - p[0]),
                                 0 if p[2] == p[3] else (cutoffs[1] - p[3]) / (p[2] - p[3])]
                    lines += [(max (cutoffs[0], p[0]), p[1]), (max (y_cutoffs[0], 0), 1), colors[idx],
                              (p[1], p[2]), (1, 1), colors[idx],
                              (p[2], min (cutoffs[1], p[3])), (1, max (y_cutoffs[1], 0)), colors[idx]]
                elif cutoffs[0] >= p[1] and cutoffs[0] <= p[2]:
                    y_cutoffs = [1, 0 if p[2] == p[3] else (cutoffs[1] - p[3]) / (p[2] - p[3])]
                    lines += [(cutoffs[0], p[2]), (1, 1), colors[idx],
                              (p[2], min (cutoffs[1], p[3])), (1, max (y_cutoffs[1], 0)), colors[idx]]
                else:
                    if p[2] == p[3]:
                        return
                    y_cutoffs = [(cutoffs[0] - p[3]) / (p[2] - p[3]), (cutoffs[1] - p[3]) / (p[2] - p[3])]
                    lines += [(cutoffs[0], min (cutoffs[1], p[3])), (y_cutoffs[0], max (y_cutoffs[1], 0)), colors[idx]]
        else:
            raise ValueError
    return lines, curves



def getReport (values, concept, expectation, observation, title = "", ignoreMinNoise = False, ignoreMaxNoise = False,
               outputPath = "report.png"):
    cutoff = 0.1; plotCutoff = 0.01 if cutoff == 0 else cutoff
    masked = values.replace (concept.get ("label_values", list ()) + [-np.inf, np.inf], np.nan).dropna ()
    minimum = np.floor (masked.min ()); maximum = np.ceil (masked.max ())
    minLevel = -np.inf if ignoreMinNoise else concept.get ("MIN-NOISE", -np.inf)
    maxLevel = np.inf if ignoreMaxNoise else concept.get ("MAX-NOISE", np.inf)
    names = list (); params = list (); colors = list ()
    for key in concept:
        if key in ["number_fuzzy_sets", "label_values", "MIN-NOISE", "MAX-NOISE"]:
            continue
        names.append (key); params.append (concept[key][0]); colors.append (concept[key][2])
    lines, curves = _getLines (params, [max (minimum, minLevel), min (maximum, maxLevel)], colors)
    handles = [Line2D ([0], [0], color = c, linewidth = 2) for c in colors]
    annData = pd.DataFrame ({"observation": observation, "expectation": expectation, "deviation": observation - expectation}).T
    pltData = annData.copy (); pltData.loc[["observation", "expectation"]] = 0
    fig = plt.figure (figsize = (8, 6), layout = "constrained"); gs = fig.add_gridspec (2, 3)
    ax = fig.add_subplot (gs[0, :3])
    ax.hist (masked, bins = 25, color = "silver"); ax.set_xlim ((minimum, maximum))
    if minLevel > minimum:
        ax.axvline (minLevel, color = "black", linestyle = "dashed")
    if maxLevel < maximum:
        ax.axvline (maxLevel, color = "black", linestyle = "dashed")
    ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of raw values", size = 10)
    ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("fuzzy value", size = 10)
    ax2.plot (*lines, linewidth = 2)
    for curve in curves:
        ax2.plot (curve[0], curve[1], c = curve[2], linewidth = 2)
    ax2.legend (handles, names, facecolor = "white")
    ax = fig.add_subplot (gs[1, :])
    sns.heatmap (pltData, vmin = -3 * cutoff, vmax = 3 * cutoff, center = 0, cmap = sns.color_palette ("vlag", 3),
                 annot = annData, fmt = ".1%", linewidth = 0.5, linecolor = "black", ax = ax)
    ax.axhline (3, color = "black", linewidth = 2); ax.axvline (pltData.shape[1], color = "black", linewidth = 2)
    ax.axhline (2, color = "red", linewidth = 3)
    ax.set_xticks (ax.get_xticks ()); ax.set_xticklabels (ax.get_xticklabels (), rotation = 0, ha = "center"); ax.xaxis.tick_bottom ()
    ax.set_yticks (ax.get_yticks ()); ax.set_yticklabels (ax.get_yticklabels (), rotation = 0, ha = "right"); ax.yaxis.tick_left ()
    ax.set_xlabel (""); ax.set_ylabel ("observation - expectation", size = 10); ax.yaxis.set_label_position ("right")
    ax.set_title ("categorial Gaussian test", size = 12)
    colorbar = ax.collections[0].colorbar; colorbar.set_ticks ([-2 * plotCutoff, -plotCutoff, 0, plotCutoff, 2 * plotCutoff])
    colorbar.set_ticklabels (["not\naccepted", f"{-cutoff:.1%}", "accepted", f"{cutoff:.1%}", "not\naccepted"], size = 9)
    if title != "":
        fig.suptitle (title, size = 15)
    plt.savefig (outputPath); plt.close ()



def getClusterMap (df, palette, axisLabel, center = None, title = "", outputPath = "clustermap.png"):
    try:
        if center is None:
            g = sns.clustermap (df, dendrogram_ratio = (0.2, 0.1), cmap = palette, yticklabels = False, figsize = (5, 6))
        else:
            g = sns.clustermap (df, dendrogram_ratio = (0.2, 0.1), cmap = palette, center = center, yticklabels = False, figsize = (5, 6))
    except RecursionError:
        return
    g.ax_heatmap.set_xticklabels (g.ax_heatmap.get_xticklabels (), size = 9, rotation = 0, ha = "center")
    g.ax_heatmap.xaxis.tick_bottom (); g.ax_heatmap.set_ylabel (axisLabel, size = 10)
    if title != "":
        g.fig.suptitle (title, size = 12.5)
    plt.savefig (outputPath); plt.close ()



def optimizeFit (rawValues, concept, exp, obs, cutoff, centerFS = "MEDIUM", ignoreMinNoise = False, ignoreMaxNoise = False, maxIteration = 100):
    numFS = concept.get ("number_fuzzy_sets", 0); diff = obs - exp
    if (not concept) or numFS == 0:
        return dict ()
    if (diff[centerFS] >= -cutoff) or (diff.drop (centerFS) <= cutoff).all ():
        return {centerFS: concept[centerFS][0]}
    diff[centerFS] = np.abs (diff[centerFS]); diff = diff[diff > cutoff]
    allSets = list (diff.index); width = concept[centerFS][0][1] / len (allSets)
    labels = concept.get ("label_values", list ())
    minLevel = -np.inf if ignoreMinNoise else concept.get ("MIN-NOISE", -np.inf)
    maxLevel = np.inf if ignoreMaxNoise else concept.get ("MAX-NOISE", np.inf)
    masked = rawValues.replace (labels, np.nan); masked = masked.mask ((masked <= minLevel) | (masked >= maxLevel)).dropna ().to_numpy ()
    params = {FS: [(concept[FS][0][1] + concept[FS][0][2]) / 2 if concept[FS][1] == "trapezoidal" else concept[FS][0][0], width]
              for FS in diff.index}
    numIter = 0; bestResult = np.round ([params[FS] for FS in allSets], 3)
    while numIter < maxIteration:
        memberships = pd.DataFrame ({FS: stats.norm.pdf (masked, loc = params[FS][0], scale = params[FS][1]) for FS in allSets})
        pctMainFS = memberships.idxmax (axis = 1, skipna = True).value_counts (normalize = True)
        pctMainFS = pd.Series ([pctMainFS.get (FS, 0) for FS in allSets], index = allSets)
        probSampleFS = memberships * pctMainFS; probSample = probSampleFS.sum (axis = 1)
        params = dict (); newSets = list ()
        for FS in allSets:
            with np.errstate (divide = "ignore", invalid = "ignore"):
                prob = pd.Series ([probSampleFS.loc[sample, FS] / probSample[sample] for sample in probSample.index])
                peak = (prob * masked).sum (skipna = True) / prob.sum (skipna = True)
                width = np.sqrt ((prob * (masked - peak) ** 2).sum (skipna = True) / prob.sum (skipna = True))
            if (not np.isnan (peak)) and (not np.isnan (width)) and width != 0:
                params[FS] = [round (peak, 3), round (width, 3)]; newSets.append (FS)
        res = np.array ([params[FS] for FS in newSets]); allSets = newSets.copy ()
        if res.shape[0] == bestResult.shape[0]:
            if (res == bestResult).all ():
                break
        elif res.shape[0] <= 1:
            return {centerFS: concept[centerFS][0]}
        bestResult = res.copy (); numIter += 1
    return params



# in Bearbeitung
def optimizeConcept (rawValues, concept, optParams, xRange, widthFct = 1, slopeFct = 0.5):
    if not optParams:
        return concept
    allSets = [key for key in concept if key not in ["number_fuzzy_sets", "label_values", "MIN-NOISE", "MAX-NOISE"]]
    optSets = list (optParams.keys ()); tmp = [-1] + [allSets.index (FS) for FS in optSets] + [len (allSets)]
    idxList = pd.DataFrame (index = optParams.keys (), columns = ["left", "right"], dtype = int)
    for idx in range (1, len (optSets) + 1):
        idxList.loc[optSets[idx - 1]] = {"left": tmp[idx] - tmp[idx - 1] - 1, "right": tmp[idx + 1] - tmp[idx] - 1}
    partial = dict (); centers = dict ()
    for FS in optSets:
        left, right = idxList.loc[FS].astype (int); mu, sigma = optParams[FS]
        centerIdx = left if FS == optSets[0] else right if FS == optSets[-1] else max ([left, right])
        numFS = 2 * centerIdx + 1; centers[FS] = centerIdx
        coords = [mu + widthFct * (i + overlap) * sigma for i in np.linspace (-numFS, numFS, numFS + 1)
                  for overlap in [-slopeFct, slopeFct]]
        coords = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFS + 1)], 3).tolist ()
        coords[centerIdx] = [mu, sigma]; partial[FS] = coords
    minimum = concept.get ("MIN-NOISE", -np.inf); minimum = minimum if np.isfinite (minimum) else xRange[0]
    maximum = concept.get ("MAX-NOISE", np.inf); maximum = maximum if np.isfinite (maximum) else xRange[1]
    currFS = optSets[0]; left = centers[currFS]; fullConcept = partial[currFS][:(left + 1)]; origin = [currFS] * (left + 1)
    for idx in range (len (optSets) - 1):
        currFS = optSets[idx]; nextFS = optSets[idx + 1]; left = centers[currFS]; right = centers[nextFS]
        overlap = int (idxList.loc[currFS, "right"]); op = int (overlap // 2)
        if overlap == 0:
            fullConcept.append (partial[nextFS][right]); origin.append (nextFS)
        elif overlap % 2 == 0:
            leftConcept = partial[currFS][(left + 1):(left + op + 1)]; rightConcept = partial[nextFS][-(right + op + 1):(right + 1)]
            opLeft = max (leftConcept[-1][1], min (leftConcept[-1][2], rightConcept[0][0]))
            opRight = min (rightConcept[0][2], max (leftConcept[-1][3], rightConcept[0][1]))
            overlap = [[leftConcept[-1][0], leftConcept[-1][1], opLeft, opRight],
                       [opLeft, opRight, rightConcept[0][2], rightConcept[0][3]]]
            fullConcept += leftConcept[:-1] + overlap + rightConcept[1:]; origin += [currFS] * op + [nextFS] * (op + 1)
        else:
            leftConcept = partial[currFS][(left + 1):(left + op + 2)]; rightConcept = partial[nextFS][-(right + op + 2):(right + 1)]
            opLeft = leftConcept[-1][1]; opRight = rightConcept[0][2]
            overlap = [[leftConcept[-1][0], opLeft, opRight, rightConcept[0][3]]]
            if leftConcept[-1][0] > rightConcept[0][3]:
                leftConcept = leftConcept[:-1]; rightConcept = rightConcept[1:]
                opLeft = max (leftConcept[-1][1], min (leftConcept[-1][2], rightConcept[0][0]))
                opRight = min (rightConcept[0][2], max (leftConcept[-1][3], rightConcept[0][1]))
                overlap = [[leftConcept[-1][0], leftConcept[-1][1], opLeft, opRight], list (),
                           [opLeft, opRight, rightConcept[0][2], rightConcept[0][3]]]
            elif opLeft > opRight:
                mid = round ((opLeft + opRight) / 2, 3); opLeft = mid; opRight = mid
                overlap = [[leftConcept[-1][0], opLeft, opRight, rightConcept[0][3]]]
            fullConcept += leftConcept[:-1] + overlap + rightConcept[1:]; origin += [currFS] * op + [""] + [nextFS] * (op + 1)
    fullConcept += partial[optSets[-1]][(centers[optSets[-1]] + 1):]; origin += [optSets[-1]] * right
    fullConcept = _adjustBorder (fullConcept, minimum, maximum)
    expPct = {FS: getSubarea (*partial[FS][centers[FS]], fullConcept, minLevel = minimum, maxLevel = maximum) for FS in optSets}
    fullConcept = dict (zip (allSets, fullConcept)); labels = concept.get ("label_values", list ())
    fct = dict (zip (optSets, getPercentage (rawValues, [fullConcept[key] for key in optSets], labels = labels,
                                             minLevel = minimum, maxLevel = maximum)))
    newConcept = concept.copy ()
    for idx in range (len (allSets)):
        FS = allSets[idx]; params = fullConcept[FS]
        if origin[idx] == "":
            exp = (fct[origin[idx - 1]] * expPct[origin[idx - 1]][idx - 1] + fct[origin[idx + 1]] * expPct[origin[idx + 1]][idx + 1]) / 2
        else:
            exp = fct[origin[idx]] * expPct[origin[idx]][idx]
        if len (params) == 4:
            newConcept[FS] = [params, "trapezoidal", concept[FS][2], round (exp, 5)]
        elif len (params) == 2:
            newConcept[FS] = [params, "Gaussian", concept[FS][2], round (exp, 5)]
        else:
            newConcept[FS] = [list (), "", concept[FS][2], 0]
    return newConcept


