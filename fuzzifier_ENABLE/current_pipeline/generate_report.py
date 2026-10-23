import os
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# python generate_report.py --dir dataDirectory --label1 firstLabel --label2 secondLabel --group1 firstGroup --group2 secondGroup --output outputDirectory


def _parseConcept (concept, allSets):
    if concept["number_fuzzy_sets"] == 0:
        params = list (); colors = list  ()
    else:
        params = [concept[FS][0] for FS in allSets]; colors = [concept[FS][2] for FS in allSets]
    return params, colors



def _getLines (params, cutoffs, colors):
    lines = list (); curves = list (); numFuzzySets = len (params)
    if cutoffs[0] >= cutoffs[1] or (~np.isfinite (cutoffs[0])) or (~np.isfinite (cutoffs[1])):
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



def generateReports (leftMtx, rightMtx, left_concepts, right_concepts, noise_cutoff = -np.inf,
                     width_factor = 1, slope_factor = 0.5, fuzzy_variables = ["LOW", "low", "MID", "high", "HIGH"],
                     left_prefix = "left", right_prefix = "right", axis_label = "expression", savePlotDir = "./"):
    numFS = len (fuzzy_variables); featureList = sorted (set (leftMtx.index) & set (rightMtx.index)); cutoff = 0.05
    minFuzzyValue = 0.5; cut = (minFuzzyValue - 0.5) / 0.5
    xRange = [min (np.floor (leftMtx.mask (~np.isfinite (leftMtx)).min (axis = None, skipna = True)) - 1,
                   np.floor (rightMtx.mask (~np.isfinite (rightMtx)).min (axis = None, skipna = True)) - 1),
              max (np.ceil (leftMtx.mask (~np.isfinite (leftMtx)).max (axis = None, skipna = True)) + 1,
                   np.ceil (rightMtx.mask (~np.isfinite (rightMtx)).max (axis = None, skipna = True)) + 1)]
    lim_xRange = [noise_cutoff - 1, xRange[1]] if np.isfinite (noise_cutoff) else xRange
    
    for idx in range (len (featureList)):
        fig = plt.figure (figsize = (12, 10)); gs = fig.add_gridspec (4, 5); feature = featureList[idx]
        ax = fig.add_subplot (gs[:2, :2])
        pltData = pd.concat ([pd.DataFrame ({"group": left_prefix, "base": "all genes", "value": leftMtx.melt ()["value"]}),
                              pd.DataFrame ({"group": right_prefix, "base": "all genes", "value": rightMtx.melt ()["value"]}),
                              pd.DataFrame ({"group": left_prefix, "base": feature, "value": leftMtx.loc[feature]}),
                              pd.DataFrame ({"group": right_prefix, "base": feature, "value": rightMtx.loc[feature]})],
                             axis = 0, ignore_index = True).replace (-np.inf, np.nan).dropna ()
        sns.violinplot (pltData, x = "value", y = "group", hue = "base", order = [left_prefix, right_prefix], hue_order = ["all genes", feature],
                        palette = {"all genes": "gray", feature: "gold"}, inner = "quart", ax = ax)
        ax.set_xlabel (axis_label, size = 10); ax.set_ylabel (""); ax.legend ().set_title (None)
        
        ax = fig.add_subplot (gs[2:, 2:])
        pltData = pd.DataFrame (index = [left_prefix, right_prefix], columns = ["NO EXP", "NOISE"] + fuzzy_variables, dtype = float)
        values = leftMtx.loc[feature].replace (-np.inf, xRange[0])
        concept, colors = _parseConcept (left_concepts[feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        intersection = [mu + width_factor * (k + cut * s) * sigma for k in np.linspace (-numFS, numFS, numFS + 1) for s in [-slope_factor, slope_factor]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = mu, scale = sigma)) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (noise_cutoff, loc = mu, scale = sigma); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [noise_cutoff] + intersection + [xRange[1]]
        observation = pd.Series ([((values != xRange[0]) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= noise_cutoff)).mean ()
        pltData.loc[left_prefix] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        values = rightMtx.loc[feature].replace (-np.inf, xRange[0])
        concept, colors = _parseConcept (right_concepts[feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        intersection = [mu + width_factor * (k + cut * s) * sigma for k in np.linspace (-numFS, numFS, numFS + 1)
                        for s in [-slope_factor, slope_factor]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = mu, scale = sigma)) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (noise_cutoff, loc = mu, scale = sigma); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [noise_cutoff] + intersection + [xRange[1]]
        observation = pd.Series ([((values != xRange[0]) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= noise_cutoff)).mean ()
        pltData.loc[right_prefix] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        sns.heatmap (pltData, vmin = -3 * cutoff, vmax = 3 * cutoff, cmap = sns.color_palette ("vlag", 3),
                     annot = True, fmt = ".2%", annot_kws = {"size": 10},
                     linecolor = "silver", linewidth = 0.5, ax = ax)
        ax.set_ylabel ("observation - expectation", size = 10); ax.yaxis.set_label_position ("right")
        ax.set_title ("fuzzy Gaussian test", size = 15); ax.set_facecolor ("darkgray")
        ax.set_yticks ([0.5, 1.5]); ax.set_yticklabels ([left_prefix, right_prefix], rotation = 0, ha = "right")
        colorbar = ax.collections[0].colorbar; colorbar.set_ticks ([-2 * cutoff, -cutoff, 0, cutoff, 2 * cutoff])
        colorbar.set_ticklabels (["not\nGaussian", f"{-cutoff:.1%}", "Gaussian", f"{cutoff:.1%}", "not\nGaussian"])
        
        ax = fig.add_subplot (gs[0, 2:])
        values = leftMtx.loc[feature]; concept, colors = _parseConcept (left_concepts[feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= noise_cutoff)).sum ()
        if np.isfinite (values).any ():
            ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver")
        ax.axvline (noise_cutoff, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel (axis_label, size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"{left_prefix} - {numZeros} no expression & {numNoise} noise in {leftMtx.shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (sigma):
            lines, curves = _getLines (concept, [noise_cutoff, xRange[1]], colors); ax2.plot (*lines, linewidth = 2)
            ax.plot (curves[0][0], curves[0][1], color = curves[0][2], linewidth = 2, label = f"mu: {mu:.2f}\nsigma: {sigma:.2f}")
            ax.legend ()
        
        ax = fig.add_subplot (gs[2, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= noise_cutoff)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = mu, scale = sigma)
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (noise_cutoff, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"{left_prefix} - {len (values)} values for fitting", size = 12)
        
        ax = fig.add_subplot (gs[1, 2:])
        values = rightMtx.loc[feature]; concept, colors = _parseConcept (right_concepts[feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= noise_cutoff)).sum ()
        if np.isfinite (values).any ():
            ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver")
        ax.axvline (noise_cutoff, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel (axis_label, size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"{right_prefix} - {numZeros} no expression & {numNoise} noise in {rightMtx.shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (sigma):
            lines, curves = _getLines (concept, [noise_cutoff, xRange[1]], colors); ax2.plot (*lines, linewidth = 2)
            ax.plot (curves[0][0], curves[0][1], color = curves[0][2], linewidth = 2, label = f"mu: {mu:.2f}\nsigma: {sigma:.2f}")
            ax.legend ()
        
        ax = fig.add_subplot (gs[3, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= noise_cutoff)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = mu, scale = sigma)
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (noise_cutoff, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"{right_prefix} - {len (values)} values for fitting", size = 12)
        fig.suptitle (feature, size = 15); fig.tight_layout ()
        plt.savefig (os.path.join (savePlotDir, f"{feature}_summary.png")); plt.close ()



def generateCombinedReports (leftMtx, rightMtx, left_concepts, right_concepts, noise_cutoff = -np.inf,
                             width_factor = 1, slope_factor = 0.5, fuzzy_variables = ["LOW", "MID", "HIGH"],
                             left_prefix = "left", right_prefix = "right", group1 = "group1", group2 = "group2",
                             axis_label = "expression", savePlotDir = "./"):
    numFS = len (fuzzy_variables); cutoff = 0.05; minFuzzyValue = 0.5; cut = (minFuzzyValue - 0.5) / 0.5
    featureList = sorted (set (leftMtx[group1].index) & set (leftMtx[group2].index) & set (rightMtx[group1].index) & set (rightMtx[group2].index))
    tmp = pd.concat ([leftMtx[group1], leftMtx[group2], rightMtx[group1], rightMtx[group2]], axis = 1).replace (-np.inf, np.nan)
    xRange = [np.floor (tmp.min (axis = None, skipna = True)) - 1, np.ceil (tmp.max (axis = None, skipna = True)) + 1]
    lim_xRange = [5, xRange[1]]
    
    for idx in range (len (featureList)):
        fig = plt.figure (figsize = (18, 16)); gs = fig.add_gridspec (8, 6); feature = featureList[idx]
        ax = fig.add_subplot (gs[:4, :2])
        pltData = pd.concat ([pd.DataFrame ({"group": left_prefix, "base": f"all genes in {group1}s", "value": leftMtx[group1].melt ()["value"]}),
                              pd.DataFrame ({"group": right_prefix, "base": f"all genes in {group1}s", "value": rightMtx[group1].melt ()["value"]}),
                              pd.DataFrame ({"group": left_prefix, "base": f"{feature} in {group1}s", "value": leftMtx[group1].loc[feature]}),
                              pd.DataFrame ({"group": right_prefix, "base": f"{feature} in {group1}s", "value": rightMtx[group1].loc[feature]}),
                              pd.DataFrame ({"group": left_prefix, "base": f"all genes in {group2}s", "value": leftMtx[group2].melt ()["value"]}),
                              pd.DataFrame ({"group": right_prefix, "base": f"all genes in {group2}s", "value": rightMtx[group2].melt ()["value"]}),
                              pd.DataFrame ({"group": left_prefix, "base": f"{feature} in {group2}s", "value": leftMtx[group2].loc[feature]}),
                              pd.DataFrame ({"group": right_prefix, "base": f"{feature} in {group2}s", "value": rightMtx[group2].loc[feature]})],
                             axis = 0, ignore_index = True).replace (-np.inf, np.nan).dropna ()
        sns.violinplot (pltData, x = "value", y = "group", hue = "base", order = [left_prefix, right_prefix],
                        hue_order = [f"all genes in {group1}s", f"{feature} in {group1}s", f"all genes in {group2}s", f"{feature} in {group2}s"],
                        palette = {f"all genes in {group1}s": "dimgray", f"{feature} in {group1}s": "gold",
                                   f"all genes in {group2}s": "lightgray", f"{feature} in {group2}s": "khaki"},
                        inner = "quart", ax = ax)
        ax.set_xlabel (axis_label, size = 10); ax.set_ylabel (""); ax.legend ().set_title (None)
        
        ax = fig.add_subplot (gs[4:, 2:])
        pltData = pd.DataFrame (index = [f"{group1} {left_prefix}", f"{group2} {left_prefix}", f"{group1} {right_prefix}", f"{group2} {right_prefix}"],
                                columns = ["NO EXP", "NOISE"] + fuzzy_variables, dtype = float)
        values = leftMtx[group1].loc[feature].replace (-np.inf, xRange[0]); concept, colors = _parseConcept (left_concepts[group1][feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        intersection = [mu + width_factor * (k + cut * s) * sigma for k in np.linspace (-numFS, numFS, numFS + 1)
                        for s in [-slope_factor, slope_factor]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc =mu, scale = sigma)) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (noise_cutoff, loc = mu, scale = sigma); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [noise_cutoff] + intersection + [xRange[1]]
        observation = pd.Series ([((values != noise_cutoff) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= noise_cutoff)).mean ()
        pltData.loc[f"{group1} {left_prefix}"] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        values = leftMtx[group2].loc[feature].replace (-np.inf, xRange[0]); concept, colors = _parseConcept (left_concepts[group2][feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        intersection = [mu + width_factor * (k + cut * s) * sigma for k in np.linspace (-numFS, numFS, numFS + 1)
                        for s in [-slope_factor, slope_factor]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = mu, scale = sigma)) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (noise_cutoff, loc = mu, scale = sigma); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [noise_cutoff] + intersection + [xRange[1]]
        observation = pd.Series ([((values != noise_cutoff) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= noise_cutoff)).mean ()
        pltData.loc[f"{group2} {left_prefix}"] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        values = rightMtx[group1].loc[feature].replace (-np.inf, xRange[0]); concept, colors = _parseConcept (right_concepts[group1][feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        intersection = [mu + width_factor * (k + cut * s) * sigma for k in np.linspace (-numFS, numFS, numFS + 1)
                        for s in [-slope_factor, slope_factor]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc =mu, scale = sigma)) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (noise_cutoff, loc =mu, scale = sigma); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [noise_cutoff] + intersection + [xRange[1]]
        observation = pd.Series ([((values != noise_cutoff) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= noise_cutoff)).mean ()
        pltData.loc[f"{group1} {right_prefix}"] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        values = rightMtx[group2].loc[feature].replace (-np.inf, xRange[0]); concept, colors = _parseConcept (right_concepts[group2][feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        intersection = [mu + width_factor * (k + cut * s) * sigma for k in np.linspace (-numFS, numFS, numFS + 1)
                        for s in [-slope_factor, slope_factor]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = mu, scale = sigma)) + [1])[::2],
                                 index = fuzzy_variables)
        noiseExp = stats.norm.cdf (noise_cutoff, loc = mu, scale = sigma); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [noise_cutoff] + intersection + [xRange[1]]
        observation = pd.Series ([((values != noise_cutoff) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= noise_cutoff)).mean ()
        pltData.loc[f"{group2} {right_prefix}"] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        sns.heatmap (pltData, vmin = -3 * cutoff, vmax = 3 * cutoff, cmap = sns.color_palette ("vlag", 3),
                     annot = True, fmt = ".2%", annot_kws = {"size": 10},
                     linecolor = "silver", linewidth = 0.5, ax = ax)
        ax.set_yticks (ax.get_yticks ()); ax.set_yticklabels (ax.get_yticklabels (), rotation = 0, ha = "right")
        ax.set_ylabel ("observation - expectation", size = 10); ax.yaxis.set_label_position ("right")
        ax.set_title ("fuzzy Gaussian test", size = 15); ax.set_facecolor ("darkgray")
        colorbar = ax.collections[0].colorbar; colorbar.set_ticks ([-2 * cutoff, -cutoff, 0, cutoff, 2 * cutoff])
        colorbar.set_ticklabels (["not\nGaussian", f"{-cutoff:.1%}", "Gaussian", f"{cutoff:.1%}", "not\nGaussian"])
        
        ax = fig.add_subplot (gs[:2, 2:4])
        values = leftMtx[group1].loc[feature]; concept, colors = _parseConcept (left_concepts[group1][feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= noise_cutoff)).sum ()
        if np.isfinite (values).any ():
            ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver")
        ax.axvline (noise_cutoff, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel ("log2 (non-zero count)", size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"{group1} {left_prefix} - {numZeros} no expression & {numNoise} noise in {leftMtx[group1].shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (sigma):
            lines, curves = _getLines (concept, [noise_cutoff, xRange[1]], colors); ax2.plot (*lines, linewidth = 2)
            ax.plot (curves[0][0], curves[0][1], color = curves[0][2], linewidth = 2, label = f"mu: {mu:.2f}\nsigma: {sigma:.2f}")
            ax.legend ()
        
        ax = fig.add_subplot (gs[4, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= noise_cutoff)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = mu, scale = sigma)
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (noise_cutoff, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"{group1} {left_prefix} - {len (values)} values for fitting", size = 12)
        
        ax = fig.add_subplot (gs[:2, 4:])
        values = leftMtx[group2].loc[feature]; concept, colors = _parseConcept (left_concepts[group2][feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= noise_cutoff)).sum ()
        if np.isfinite (values).any ():
            ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver")
        ax.axvline (noise_cutoff, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel ("log2 (non-zero count)", size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"{group2} {left_prefix} - {numZeros} no expression & {numNoise} noise in {leftMtx[group2].shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (sigma):
            lines, curves = _getLines (concept, [noise_cutoff, xRange[1]], colors); ax2.plot (*lines, linewidth = 2)
            ax.plot (curves[0][0], curves[0][1], color = curves[0][2], linewidth = 2, label = f"mu: {mu:.2f}\nsigma: {sigma:.2f}")
            ax.legend ()
        
        ax = fig.add_subplot (gs[6, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= 6)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = mu, scale = sigma)
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (6, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"{group2} {left_prefix} - {len (values)} values for fitting", size = 12)
        
        ax = fig.add_subplot (gs[2:4, 2:4])
        values = rightMtx[group1].loc[feature]; concept, colors = _parseConcept (right_concepts[group1][feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= noise_cutoff)).sum ()
        if np.isfinite (values).any ():
            ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver")
        ax.axvline (noise_cutoff, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel ("log2 (non-zero count)", size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"{group1} {right_prefix} - {numZeros} no expression & {numNoise} noise in {rightMtx[group1].shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (sigma):
            lines, curves = _getLines (concept, [noise_cutoff, xRange[1]], colors); ax2.plot (*lines, linewidth = 2)
            ax.plot (curves[0][0], curves[0][1], color = curves[0][2], linewidth = 2, label = f"mu: {mu:.2f}\nsigma: {sigma:.2f}")
            ax.legend ()
        
        ax = fig.add_subplot (gs[5, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= noise_cutoff)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = mu, scale = sigma)
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (noise_cutoff, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"{group1} {right_prefix} - {len (values)} values for fitting", size = 12)
        
        ax = fig.add_subplot (gs[2:4, 4:])
        values = rightMtx[group2].loc[feature]; concept, colors = _parseConcept (right_concepts[group2][feature], fuzzy_variables)
        if len (concept) == 0:
            mu = np.nan; sigma = np.nan
        else:
            mu = concept[2][0]; sigma = concept[2][1]
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= noise_cutoff)).sum ()
        if np.isfinite (values).any ():
            ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver")
        ax.axvline (noise_cutoff, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel ("log2 (non-zero count)", size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"{group2} {right_prefix} - {numZeros} no expression & {numNoise} noise in {rightMtx[group2].shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (sigma):
            lines, curves = _getLines (concept, [noise_cutoff, xRange[1]], colors); ax2.plot (*lines, linewidth = 2)
            ax.plot (curves[0][0], curves[0][1], color = curves[0][2], linewidth = 2, label = f"mu: {mu:.2f}\nsigma: {sigma:.2f}")
            ax.legend ()
        
        ax = fig.add_subplot (gs[7, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= noise_cutoff)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = mu, scale = sigma)
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (noise_cutoff, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"{group2} {right_prefix} - {len (values)} values for fitting", size = 12)
        fig.suptitle (feature, size = 15); fig.tight_layout ()
        plt.savefig (os.path.join (savePlotDir, f"{feature}_summary.png")); plt.close ()



def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--dir", type = str, required = True, help = "Directory for raw and fuzzy values")
    parser.add_argument ("--label1", type = str, required = False, default = "left", help = "Label for the first matrix")
    parser.add_argument ("--label2", type = str, required = False, default = "right", help = "Label for the second matrix")
    parser.add_argument ("--group1", type = str, required = False, help = "Label for the first patient group in joint reports")
    parser.add_argument ("--group2", type = str, required = False, help = "Label for the second patient group in joint reports")
    parser.add_argument ("--output", type = str, required = True, help = "Output directory for report plots")
    args = parser.parse_args ()

    os.makedirs (args.output, exist_ok = True)
    if args.group1 is None or args.group2 is None:
        leftMtx = pd.read_csv (os.path.join (args.dir, f"{args.label1}.tsv"), index_col = 0, sep = "\t")
        rightMtx = pd.read_csv (os.path.join (args.dir, f"{args.label2}.tsv"), index_col = 0, sep = "\t")
        with open (os.path.join (args.dir, f"fuzzy_{args.label1}", "concepts_detailed.json")) as f:
            left_concepts = json.load (f)
        with open (os.path.join (args.dir, f"fuzzy_{args.label2}", "concepts_detailed.json")) as f:
            right_concepts = json.load (f)
        generateReports (leftMtx, rightMtx, left_concepts, right_concepts, noise_cutoff = 6,
                         width_factor = 0.5, slope_factor = 0.5, fuzzy_variables = ["LOW", "low", "MID", "high", "HIGH"],
                         left_prefix = args.label1, right_prefix = args.label2,
                         axis_label = "log2 (non-zero count)", savePlotDir = args.output)
    else:
        leftMtx = {args.group1: pd.read_csv (os.path.join (args.dir, args.group1, f"{args.label1}.tsv"), index_col = 0, sep = "\t"),
                   args.group2: pd.read_csv (os.path.join (args.dir, args.group2, f"{args.label1}.tsv"), index_col = 0, sep = "\t")}
        rightMtx = {args.group1: pd.read_csv (os.path.join (args.dir, args.group1, f"{args.label2}.tsv"), index_col = 0, sep = "\t"),
                    args.group2: pd.read_csv (os.path.join (args.dir, args.group2, f"{args.label2}.tsv"), index_col = 0, sep = "\t")}
        leftConcepts = dict (); rightConcepts = dict ()
        with open (os.path.join (args.dir, args.group1, f"fuzzy_{args.label1}", "concepts_detailed.json")) as f:
            leftConcepts[args.group1] = json.load (f)
        with open (os.path.join (args.dir, args.group2, f"fuzzy_{args.label1}", "concepts_detailed.json")) as f:
            leftConcepts[args.group2] = json.load (f)
        with open (os.path.join (args.dir, args.group1, f"fuzzy_{args.label2}", "concepts_detailed.json")) as f:
            rightConcepts[args.group1] = json.load (f)
        with open (os.path.join (args.dir, args.group2, f"fuzzy_{args.label2}", "concepts_detailed.json")) as f:
            rightConcepts[args.group2] = json.load (f)
        generateCombinedReports (leftMtx, rightMtx, leftConcepts, rightConcepts, noise_cutoff = 6,
                                 width_factor = 0.5, slope_factor = 0.5, fuzzy_variables = ["LOW", "low", "MID", "high", "HIGH"],
                                 left_prefix = args.label1, right_prefix = args.label2, group1 = args.group1, group2 = args.group2,
                                 axis_label = "log2 (non-zero count)", savePlotDir = args.output)



if __name__ == "__main__":
    main ()


