import os, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize


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


def estimateGauss (data, noise_cutoff = -np.inf, low_value_percent = 0, bw_factor = 1):
    parameters = pd.DataFrame (index = data.index, columns = ["mu", "sigma"], dtype = float)
    for idx in range (data.shape[0]):
        feature = data.index[idx]
        values = data.loc[feature]; values = values[np.isfinite (values)]
        cutoff = values[values > noise_cutoff].quantile (low_value_percent)
        values = values[values > cutoff]; mu = values.mean ()
        if len (values) < 2:
            parameters.loc[feature] = {"mu": mu, "sigma": np.nan}
            continue
        kernel = stats.gaussian_kde (values); kernel.set_bandwidth (bw_method = bw_factor * kernel.factor)
        density = pd.DataFrame ({"value": values, "density": kernel (values)}).sort_values ("value").drop_duplicates ()
        modes = density.iloc[signal.argrelmax (density["density"].to_numpy ())[0]].drop_duplicates ()
        modes.loc["mean"] = {"value": mu, "density": kernel ([mu])[0]}; modes = modes.sort_values ("value")
        meanIdx = list (modes.index).index ("mean")
        if meanIdx == 0:
            modeIdx = 1
        elif meanIdx == modes.shape[0] - 1:
            modeIdx = modes.shape[0] - 2
        else:
            modeIdx = modes.reset_index (drop = True)["density"].idxmax ()
            if np.abs (meanIdx - modeIdx) > 1:
                modeIdx = modes.reset_index (drop = True).loc[[meanIdx - 1, meanIdx + 1]].sort_values ("density").index[1]
        lb = np.floor (modes.iloc[modeIdx, 0] * 1e3) / 1e3; ub = np.ceil (modes.iloc[modeIdx, 0] * 1e3) / 1e3; ub = ub + 1e-3 if lb == ub else ub
        res, _ = optimize.curve_fit (lambda x, m, s: stats.norm.pdf (x, loc = m, scale = s), density["value"], density["density"],
                                     bounds = [(lb, -np.inf), (ub, np.inf)])
        parameters.loc[feature] = {"mu": res[0], "sigma": res[1]}
    return parameters.round (3)


def estimateConcept (data, parameter, num_fuzzy_sets = 3, width_factor = 1, slope_factor = 0.5):
    finite_data = data.mask (~np.isfinite (data)); concepts = dict ()
    for feature in data.index:
        values = finite_data.loc[feature].dropna (); mu, sigma = parameter.loc[feature, ["mu", "sigma"]]
        if np.isnan (sigma):
            continue
        xRange = [np.floor (values.min ()) - 1, np.ceil (values.max ()) + 1]
        coords = [mu + width_factor * (intersection + overlap) * sigma
                  for intersection in np.linspace (-num_fuzzy_sets, num_fuzzy_sets, num_fuzzy_sets + 1)
                  for overlap in [-slope_factor, slope_factor]]
        concept = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, num_fuzzy_sets + 1)], 3)
        left = min (xRange[0], concept[0, 3]); right = max (xRange[1], concept[-1, 1])
        concept[0][0] = left; concept[0][1] = left; concept[-1][2] = right; concept[-1][3] = right
        concepts[feature] = concept
    return concepts


def saveConcept (concepts, file_prefix, file_dir = "./"):
    output = {key: concepts[key].tolist () for key in concepts.keys ()}
    with open (os.path.join (file_dir, f"{file_prefix}_concepts.json"), "w", encoding = "utf-8") as f:
        json.dump (output, f, indent = 4)


def plotConcept (leftMtx, rightMtx, left_params, right_params, left_concepts, right_concepts, noise_cutoff = -np.inf,
                 left_prefix = "left", right_prefix = "right", x_axis_label = "expression",
                 savePlot = False, savePlotPath = "ALL_fuzzy_concepts.png"):
    featureList = sorted (set (leftMtx.index) & set (rightMtx.index))
    xRange = [min (np.floor (leftMtx.mask (~np.isfinite (leftMtx)).min (axis = None, skipna = True)) - 1,
                   np.floor (rightMtx.mask (~np.isfinite (rightMtx)).min (axis = None, skipna = True)) - 1),
              max (np.ceil (leftMtx.mask (~np.isfinite (leftMtx)).max (axis = None, skipna = True)) + 1,
                   np.ceil (rightMtx.mask (~np.isfinite (rightMtx)).max (axis = None, skipna = True)) + 1)]
    func = lambda x, p: np.exp (-(x - p[0]) ** 2 / (2 * p[1] ** 2)); xValues = np.linspace (*(xRange), 1000)
    fig, axs = plt.subplots (len (featureList), 2, sharex = True, sharey = False, figsize = (2 * len (featureList), 10))
    for idx in range (len (featureList)):
        feature = featureList[idx]
        values = leftMtx.loc[feature].replace (-np.inf, np.nan); params = left_params.loc[feature, ["mu", "sigma"]].tolist ()
        numZeros = np.isnan (values).sum (); numNoise = ((~np.isnan (values)) & (values <= noise_cutoff)).sum ()
        axs[idx, 0].hist (values, bins = 25, alpha = 0.8, color = "silver")
        if np.isfinite (noise_cutoff):
            axs[idx, 0].axvline (noise_cutoff, color = "tab:blue", linestyle = "dashed", label = "noise")
            axs[idx, 0].legend (loc = "lower left")
        ax = axs[idx, 0].twinx (); ax.set_ylim ((0, 1.05)); ax.set_ylabel ("membership", size = 10)
        if not np.isnan (params[1]):
            ax.plot (*getLines (left_concepts[feature], ["black"] * 3), linewidth = 2)
            ax.plot (xValues, func (xValues, params), color = "red", linewidth = 2, label = f"mu: {params[0]:.2f}\nsigma: {params[1]:.2f}")
            ax.legend (loc = "upper left")
        axs[idx, 0].set_xlim (xRange); axs[idx, 0].set_ylabel ("number of values", size = 10)
        axs[idx, 0].set_title (f"{left_prefix} {feature} - {numZeros} no expression & {numNoise} noise in {leftMtx.shape[1]} samples", size = 12)
        values = rightMtx.loc[feature].replace (-np.inf, np.nan); params = right_params.loc[feature, ["mu", "sigma"]].tolist ()
        numZeros = np.isnan (values).sum (); numNoise = ((~np.isnan (values)) & (values <= noise_cutoff)).sum ()
        axs[idx, 1].hist (values, bins = 25, alpha = 0.8, color = "silver")
        if np.isfinite (noise_cutoff):
            axs[idx, 1].axvline (noise_cutoff, color = "tab:blue", linestyle = "dashed", label = "noise")
            axs[idx, 1].legend (loc = "lower left")
        ax = axs[idx, 1].twinx (); ax.set_ylim ((0, 1.05)); ax.set_ylabel ("membership", size = 10)
        if not np.isnan (params[1]):
            ax.plot (*getLines (right_concepts[feature], ["black"] * 3), linewidth = 2)
            ax.plot (xValues, func (xValues, params), color = "red", linewidth = 2, label = f"mu: {params[0]:.2f}\nsigma: {params[1]:.2f}")
            ax.legend (loc = "upper left")
        axs[idx, 1].set_xlim (xRange); axs[idx, 1].set_ylabel ("number of values", size = 10)
        axs[idx, 1].set_title (f"{right_prefix} {feature} - {numZeros} no expression & {numNoise} noise in {rightMtx.shape[1]} samples", size = 12)
    fig.supxlabel (x_axis_label, size = 10); fig.supylabel (""); fig.tight_layout ()
    if savePlot:
        plt.savefig (savePlotPath)
    else:
        plt.show ()
    plt.close ()


def generateReports (leftMtx, rightMtx, left_params, right_params, left_concepts, right_concepts, noise_cutoff = -np.inf,
                     width_factor = 1, slope_factor = 0.5, fuzzy_variables = ["LOW", "MID", "HIGH"],
                     left_prefix = "left", right_prefix = "right", axis_label = "expression", savePlotDir = "./"):
    numFS = len (fuzzy_variables); featureList = sorted (set (leftMtx.index) & set (rightMtx.index)); cutoff = 0.05
    minFuzzyValue = 0.5; cut = (minFuzzyValue - 0.5) / 0.5
    xRange = [min (np.floor (leftMtx.mask (~np.isfinite (leftMtx)).min (axis = None, skipna = True)) - 1,
                   np.floor (rightMtx.mask (~np.isfinite (rightMtx)).min (axis = None, skipna = True)) - 1),
              max (np.ceil (leftMtx.mask (~np.isfinite (leftMtx)).max (axis = None, skipna = True)) + 1,
                   np.ceil (rightMtx.mask (~np.isfinite (rightMtx)).max (axis = None, skipna = True)) + 1)]
    lim_xRange = [noise_cutoff - 1, xRange[1]] if np.isfinite (noise_cutoff) else xRange; xValues = np.linspace (*(xRange), 1000)
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
        values = leftMtx.loc[feature].replace (-np.inf, xRange[0]); params = left_params.loc[feature, ["mu", "sigma"]].tolist ()
        intersection = [params[0] + width_factor * (k + cut * s) * params[1] for k in np.linspace (-numFS, numFS, numFS + 1) for s in [-slope_factor, slope_factor]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = params[0], scale = params[1])) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (noise_cutoff, loc = params[0], scale = params[1]); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
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
        values = rightMtx.loc[feature].replace (-np.inf, xRange[0]); params = right_params.loc[feature, ["mu", "sigma"]].tolist ()
        intersection = [params[0] + width_factor * (k + cut * s) * params[1] for k in np.linspace (-numFS, numFS, numFS + 1)
                        for s in [-slope_factor, slope_factor]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = params[0], scale = params[1])) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (noise_cutoff, loc = params[0], scale = params[1]); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
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
        sns.heatmap (pltData, vmin = -3 * cutoff, vmax = 3 * cutoff, cmap = sns.color_palette ("vlag", 3), annot = True, fmt = ".2%", annot_kws = {"size": 12},
                     linecolor = "silver", linewidth = 0.5, ax = ax)
        ax.set_ylabel ("observation - expectation", size = 10); ax.yaxis.set_label_position ("right")
        ax.set_title ("fuzzy Gaussian test", size = 15); ax.set_facecolor ("darkgray")
        ax.set_yticks ([0.5, 1.5]); ax.set_yticklabels ([left_prefix, right_prefix], rotation = 0, ha = "right")
        colorbar = ax.collections[0].colorbar; colorbar.set_ticks ([-2 * cutoff, -cutoff, 0, cutoff, 2 * cutoff])
        colorbar.set_ticklabels (["not\nGaussian", f"{-cutoff:.1%}", "Gaussian", f"{cutoff:.1%}", "not\nGaussian"])
        ax = fig.add_subplot (gs[0, 2:])
        values = leftMtx.loc[feature]; params = left_params.loc[feature, ["mu", "sigma"]].to_numpy ()
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= noise_cutoff)).sum ()
        ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver")
        ax.axvline (noise_cutoff, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel (axis_label, size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"{left_prefix} - {numZeros} no expression & {numNoise} noise in {leftMtx.shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (params[1]):
            ax2.plot (*getLines (left_concepts[feature], ["tab:blue", "black", "tab:red"]), linewidth = 2)
            ax.plot (xValues, stats.norm.pdf (xValues, loc = params[0], scale = params[1]), color = "tab:green", linewidth = 2,
                     label = f"mu: {params[0]:.2f}\nsigma: {params[1]:.2f}")
            ax.legend ()
        ax = fig.add_subplot (gs[2, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= noise_cutoff)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = params[0], scale = params[1])
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (noise_cutoff, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"{left_prefix} - {len (values)} values for fitting", size = 12)
        ax = fig.add_subplot (gs[1, 2:])
        values = rightMtx.loc[feature]; params = right_params.loc[feature, ["mu", "sigma"]].to_numpy ()
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= noise_cutoff)).sum ()
        ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver")
        ax.axvline (noise_cutoff, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel (axis_label, size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"{right_prefix} - {numZeros} no expression & {numNoise} noise in {rightMtx.shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (params[1]):
            ax2.plot (*getLines (right_concepts[feature], ["tab:blue", "black", "tab:red"]), linewidth = 2)
            ax.plot (xValues, stats.norm.pdf (xValues, loc = params[0], scale = params[1]), color = "tab:green",
                     linewidth = 2, label = f"mu: {params[0]:.2f}\nsigma: {params[1]:.2f}")
            ax.legend ()
        ax = fig.add_subplot (gs[3, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= noise_cutoff)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = params[0], scale = params[1])
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (noise_cutoff, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"{right_prefix} - {len (values)} values for fitting", size = 12)
        fig.suptitle (feature, size = 15); fig.tight_layout ()
        plt.savefig (os.path.join (savePlotDir, f"{feature}_summary.png")); plt.close ()


def getAnalysis (leftMtx, rightMtx, noise_cutoff = -np.inf, low_value_percent = 0, bw_factor = 1,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "expression", left_prefix = "left", right_prefix = "right", output_directory = "./"):
    leftParams = estimateGauss (leftMtx, noise_cutoff = noise_cutoff, low_value_percent = low_value_percent, bw_factor = bw_factor)
    rightParams = estimateGauss (rightMtx, noise_cutoff = noise_cutoff, low_value_percent = low_value_percent, bw_factor = bw_factor)
    leftConcepts = estimateConcept (leftMtx, leftParams, num_fuzzy_sets = num_fuzzy_sets, width_factor = width_factor, slope_factor = slope_factor)
    rightConcepts = estimateConcept (rightMtx, rightParams, num_fuzzy_sets = num_fuzzy_sets, width_factor = width_factor, slope_factor = slope_factor)
    saveConcept (leftConcepts, left_prefix, file_dir = output_directory)
    saveConcept (rightConcepts, right_prefix, file_dir = output_directory)
    plotConcept (leftMtx, rightMtx, leftParams, rightParams, leftConcepts, rightConcepts, noise_cutoff = noise_cutoff,
                 left_prefix = left_prefix, right_prefix = right_prefix, x_axis_label = axis_label,
                 savePlot = True, savePlotPath = os.path.join (output_directory, "ALL_fuzzy_concepts.png"))
    generateReports (leftMtx, rightMtx, leftParams, rightParams, leftConcepts, rightConcepts, noise_cutoff = noise_cutoff,
                     width_factor = width_factor, slope_factor = slope_factor, fuzzy_variables = fuzzy_variables,
                     left_prefix = left_prefix, right_prefix = right_prefix, axis_label = axis_label, savePlotDir = output_directory)
    

def getAnalysis_combinedMtx (adultData, adultMetadata, kidData, kidMetadata, outputDir):
    featureList = sorted (set (adultData.index) & set (kidData.index))
    with np.errstate (divide = "ignore"):
        leftMtx = {"adult": np.log2 (adultData[adultMetadata.loc[adultMetadata["category"].isin (["overweight", "obesity"])].index]),
                   "kid": np.log2 (kidData[kidMetadata.loc[kidMetadata["category"].isin (["kUmA-N", "kUmNA-N"])].index])}
    leftParams = {"adult": estimateGauss (leftMtx["adult"], noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75),
                  "kid": estimateGauss (leftMtx["kid"], noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75)}
    leftConcepts = {"adult": estimateConcept (leftMtx["adult"], leftParams["adult"], num_fuzzy_sets = 3, width_factor = 1, slope_factor = 0.5),
                    "kid": estimateConcept (leftMtx["kid"], leftParams["kid"], num_fuzzy_sets = 3, width_factor = 1, slope_factor = 0.5)}
    saveConcept (leftConcepts["adult"], "adult-overweight", file_dir = outputDir)
    saveConcept (leftConcepts["kid"], "kid-overweight", file_dir = outputDir)
    with np.errstate (divide = "ignore"):
        rightMtx = {"adult": np.log2 (adultData[adultMetadata.loc[adultMetadata["category"].isin (["underweight", "normalweight"])].index]),
                    "kid": np.log2 (kidData[kidMetadata.loc[kidMetadata["category"].isin (["kNmA-N", "kNmNA-N"])].index])}
    rightParams = {"adult": estimateGauss (rightMtx["adult"], noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75),
                   "kid": estimateGauss (rightMtx["kid"], noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75)}
    rightConcepts = {"adult": estimateConcept (rightMtx["adult"], rightParams["adult"], num_fuzzy_sets = 3, width_factor = 1, slope_factor = 0.5),
                     "kid": estimateConcept (rightMtx["kid"], rightParams["kid"], num_fuzzy_sets = 3, width_factor = 1, slope_factor = 0.5)}
    saveConcept (rightConcepts["adult"], "adult-normalweight", file_dir = outputDir)
    saveConcept (rightConcepts["kid"], "kid-normalweight", file_dir = outputDir)
    plotConcept (leftMtx["adult"], rightMtx["adult"], leftParams["adult"], rightParams["adult"], leftConcepts["adult"], rightConcepts["adult"], noise_cutoff = 6,
                 left_prefix = "adult overweight", right_prefix = "adult normalweight", x_axis_label = "log2 (non-zero count)",
                 savePlot = True, savePlotPath = os.path.join (outputDir, "ALL_adult_fuzzy_concepts.png"))
    plotConcept (leftMtx["kid"], rightMtx["kid"], leftParams["kid"], rightParams["kid"], leftConcepts["kid"], rightConcepts["kid"], noise_cutoff = 6,
                 left_prefix = "kid overweight", right_prefix = "kid normalweight", x_axis_label = "log2 (non-zero count)",
                 savePlot = True, savePlotPath = os.path.join (outputDir, "ALL_kid_fuzzy_concepts.png"))
    fuzzy_variables = ["LOW", "MID", "HIGH"]; numFS = len (fuzzy_variables); cutoff = 0.05
    minFuzzyValue = 0.5; cut = (minFuzzyValue - 0.5) / 0.5
    tmp = pd.concat ([leftMtx["adult"], leftMtx["kid"], rightMtx["adult"], rightMtx["kid"]], axis = 1).replace (-np.inf, np.nan)
    xRange = [np.floor (tmp.min (axis = None, skipna = True)) - 1, np.ceil (tmp.max (axis = None, skipna = True)) + 1]
    lim_xRange = [5, xRange[1]]; xValues = np.linspace (*(xRange), 1000)
    for idx in range (len (featureList)):
        fig = plt.figure (figsize = (18, 16)); gs = fig.add_gridspec (8, 6); feature = featureList[idx]
        ax = fig.add_subplot (gs[:4, :2])
        pltData = pd.concat ([pd.DataFrame ({"group": "overweight", "base": "all genes in adults", "value": leftMtx["adult"].melt ()["value"]}),
                              pd.DataFrame ({"group": "normalweight", "base": "all genes in adults", "value": rightMtx["adult"].melt ()["value"]}),
                              pd.DataFrame ({"group": "overweight", "base": f"{feature} in adults", "value": leftMtx["adult"].loc[feature]}),
                              pd.DataFrame ({"group": "normalweight", "base": f"{feature} in adults", "value": rightMtx["adult"].loc[feature]}),
                              pd.DataFrame ({"group": "overweight", "base": "all genes in kids", "value": leftMtx["kid"].melt ()["value"]}),
                              pd.DataFrame ({"group": "normalweight", "base": "all genes in kids", "value": rightMtx["kid"].melt ()["value"]}),
                              pd.DataFrame ({"group": "overweight", "base": f"{feature} in kids", "value": leftMtx["kid"].loc[feature]}),
                              pd.DataFrame ({"group": "normalweight", "base": f"{feature} in kids", "value": rightMtx["kid"].loc[feature]})],
                             axis = 0, ignore_index = True).replace (-np.inf, np.nan).dropna ()
        sns.violinplot (pltData, x = "value", y = "group", hue = "base", order = ["overweight", "normalweight"],
                        hue_order = ["all genes in adults", f"{feature} in adults", "all genes in kids", f"{feature} in kids"],
                        palette = {"all genes in adults": "dimgray", f"{feature} in adults": "gold",
                                   "all genes in kids": "lightgray", f"{feature} in kids": "khaki"},
                        inner = "quart", ax = ax)
        ax.set_xlabel ("log2 (non-zero count)", size = 10); ax.set_ylabel (""); ax.legend ().set_title (None)
        ax = fig.add_subplot (gs[4:, 2:])
        pltData = pd.DataFrame (index = ["adult overweight", "kid overweight", "adult normalweight", "kid normalweight"],
                                columns = ["NO EXP", "NOISE"] + fuzzy_variables, dtype = float)
        values = leftMtx["adult"].loc[feature].replace (-np.inf, xRange[0]); params = leftParams["adult"].loc[feature, ["mu", "sigma"]].tolist ()
        intersection = [params[0] + 1 * (k + cut * s) * params[1] for k in np.linspace (-numFS, numFS, numFS + 1) for s in [-0.5, 0.5]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = params[0], scale = params[1])) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (6, loc = params[0], scale = params[1]); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [6] + intersection + [xRange[1]]
        observation = pd.Series ([((values != 6) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= 6)).mean ()
        pltData.loc["adult overweight"] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        values = leftMtx["kid"].loc[feature].replace (-np.inf, xRange[0]); params = leftParams["kid"].loc[feature, ["mu", "sigma"]].tolist ()
        intersection = [params[0] + 1 * (k + cut * s) * params[1] for k in np.linspace (-numFS, numFS, numFS + 1) for s in [-0.5, 0.5]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = params[0], scale = params[1])) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (6, loc = params[0], scale = params[1]); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [6] + intersection + [xRange[1]]
        observation = pd.Series ([((values != 6) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= 6)).mean ()
        pltData.loc["kid overweight"] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        values = rightMtx["adult"].loc[feature].replace (-np.inf, xRange[0]); params = rightParams["adult"].loc[feature, ["mu", "sigma"]].tolist ()
        intersection = [params[0] + 1 * (k + cut * s) * params[1] for k in np.linspace (-numFS, numFS, numFS + 1) for s in [-0.5, 0.5]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = params[0], scale = params[1])) + [1])[::2], index = fuzzy_variables)
        noiseExp = stats.norm.cdf (6, loc = params[0], scale = params[1]); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [6] + intersection + [xRange[1]]
        observation = pd.Series ([((values != 6) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= 6)).mean ()
        pltData.loc["adult normalweight"] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        values = rightMtx["kid"].loc[feature].replace (-np.inf, xRange[0]); params = rightParams["kid"].loc[feature, ["mu", "sigma"]].tolist ()
        intersection = [params[0] + 1 * (k + cut * s) * params[1] for k in np.linspace (-numFS, numFS, numFS + 1) for s in [-0.5, 0.5]][2:-2]
        expectation = pd.Series (np.diff ([0] + list (stats.norm.cdf (intersection, loc = params[0], scale = params[1])) + [1])[::2],
                                 index = fuzzy_variables)
        noiseExp = stats.norm.cdf (6, loc = params[0], scale = params[1]); expectation["NO EXP"] = 0; expectation["NOISE"] = noiseExp
        for FS in fuzzy_variables:
            x = min (expectation[FS], noiseExp); expectation[FS] -= x; noiseExp -= x
            if noiseExp <= 0:
                break
        intersection = [6] + intersection + [xRange[1]]
        observation = pd.Series ([((values != 6) & (values >= intersection[i]) & (values < intersection[i + 1])).mean ()
                                  for i in range (len (intersection) - 1)][::2],
                                 index = fuzzy_variables)
        observation["NO EXP"] = (values == xRange[0]).mean (); observation["NOISE"] = ((values != xRange[0]) & (values <= 6)).mean ()
        pltData.loc["kid normalweight"] = observation[["NO EXP", "NOISE"] + fuzzy_variables] - expectation[["NO EXP", "NOISE"] + fuzzy_variables]
        sns.heatmap (pltData, vmin = -3 * cutoff, vmax = 3 * cutoff, cmap = sns.color_palette ("vlag", 3), annot = True, fmt = ".2%", annot_kws = {"size": 12},
                     linecolor = "silver", linewidth = 0.5, ax = ax)
        ax.set_yticks (ax.get_yticks ()); ax.set_yticklabels (ax.get_yticklabels (), rotation = 0, ha = "right")
        ax.set_ylabel ("observation - expectation", size = 10); ax.yaxis.set_label_position ("right")
        ax.set_title ("fuzzy Gaussian test", size = 15); ax.set_facecolor ("darkgray")
        colorbar = ax.collections[0].colorbar; colorbar.set_ticks ([-2 * cutoff, -cutoff, 0, cutoff, 2 * cutoff])
        colorbar.set_ticklabels (["not\nGaussian", f"{-cutoff:.1%}", "Gaussian", f"{cutoff:.1%}", "not\nGaussian"])
        ax = fig.add_subplot (gs[:2, 2:4])
        values = leftMtx["adult"].loc[feature]; params = leftParams["adult"].loc[feature, ["mu", "sigma"]].tolist ()
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= 6)).sum ()
        ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver"); ax.axvline (6, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel ("log2 (non-zero count)", size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"adult overweight - {numZeros} no expression & {numNoise} noise in {leftMtx['adult'].shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (params[1]):
            ax2.plot (*getLines (leftConcepts["adult"][feature], ["tab:blue", "black", "tab:red"]), linewidth = 2)
            ax.plot (xValues, stats.norm.pdf (xValues, loc = params[0], scale = params[1]), color = "tab:green", linewidth = 2,
                     label = f"mu: {params[0]:.2f}\nsigma: {params[1]:.2f}")
            ax.legend ()
        ax = fig.add_subplot (gs[4, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= 6)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = params[0], scale = params[1])
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (6, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"adult overweight - {len (values)} values for fitting", size = 12)
        ax = fig.add_subplot (gs[:2, 4:])
        values = leftMtx["kid"].loc[feature]; params = leftParams["kid"].loc[feature, ["mu", "sigma"]].tolist ()
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= 6)).sum ()
        ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver"); ax.axvline (6, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel ("log2 (non-zero count)", size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"kid overweight - {numZeros} no expression & {numNoise} noise in {leftMtx['kid'].shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (params[1]):
            ax2.plot (*getLines (leftConcepts["kid"][feature], ["tab:blue", "black", "tab:red"]), linewidth = 2)
            ax.plot (xValues, stats.norm.pdf (xValues, loc = params[0], scale = params[1]), color = "tab:green", linewidth = 2,
                     label = f"mu: {params[0]:.2f}\nsigma: {params[1]:.2f}")
            ax.legend ()
        ax = fig.add_subplot (gs[6, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= 6)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = params[0], scale = params[1])
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (6, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"kid overweight - {len (values)} values for fitting", size = 12)
        ax = fig.add_subplot (gs[2:4, 2:4])
        values = rightMtx["adult"].loc[feature]; params = rightParams["adult"].loc[feature, ["mu", "sigma"]].tolist ()
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= 6)).sum ()
        ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver"); ax.axvline (6, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel ("log2 (non-zero count)", size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"adult normalweight - {numZeros} no expression & {numNoise} noise in {rightMtx['adult'].shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (params[1]):
            ax2.plot (*getLines (rightConcepts["adult"][feature], ["tab:blue", "black", "tab:red"]), linewidth = 2)
            ax.plot (xValues, stats.norm.pdf (xValues, loc = params[0], scale = params[1]), color = "tab:green",
                     linewidth = 2, label = f"mu: {params[0]:.2f}\nsigma: {params[1]:.2f}")
            ax.legend ()
        ax = fig.add_subplot (gs[5, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= 6)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = params[0], scale = params[1])
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (6, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"adult normalweight - {len (values)} values for fitting", size = 12)
        ax = fig.add_subplot (gs[2:4, 4:])
        values = rightMtx["kid"].loc[feature]; params = rightParams["kid"].loc[feature, ["mu", "sigma"]].tolist ()
        numZeros = (~np.isfinite (values)).sum (); numNoise = ((np.isfinite (values)) & (values <= 6)).sum ()
        ax.hist (values[np.isfinite (values)], bins = 25, density = True, color = "silver"); ax.axvline (6, color = "darkgray", linestyle = "dashed")
        ax.set_xlim (xRange); ax.set_xlabel ("log2 (non-zero count)", size = 10); ax.set_ylabel ("density", size = 10)
        ax.set_title (f"kid normalweight - {numZeros} no expression & {numNoise} noise in {rightMtx['kid'].shape[1]} samples", size = 12)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("membership", size = 10)
        if not np.isnan (params[1]):
            ax2.plot (*getLines (rightConcepts["kid"][feature], ["tab:blue", "black", "tab:red"]), linewidth = 2)
            ax.plot (xValues, stats.norm.pdf (xValues, loc = params[0], scale = params[1]), color = "tab:green", linewidth = 2,
                     label = f"mu: {params[0]:.2f}\nsigma: {params[1]:.2f}")
            ax.legend ()
        ax = fig.add_subplot (gs[7, :2])
        values = values.mask ((~np.isfinite (values)) | (values <= 6)).dropna ()
        expected = stats.norm.ppf (np.linspace (0.01, 0.99, 99), loc = params[0], scale = params[1])
        observed = values.quantile (np.linspace (0.01, 0.99, 99)).to_numpy (); line = stats.linregress (expected, observed)
        ax.scatter (expected, observed, s = 10); ax.plot (lim_xRange, lim_xRange, color = "silver", linestyle = "dashed")
        if np.isfinite (line.slope) and np.isfinite (line.intercept):
            ax.plot (expected, line.slope * expected + line.intercept, "red", linewidth = 1)
        ax.set_xlim (lim_xRange); ax.set_ylim (lim_xRange); ax.set_yticks (np.arange (6, xRange[1], 2))
        ax.set_xlabel ("theoretical percentile", size = 10); ax.set_ylabel ("observed percentile", size = 10)
        ax.set_title (f"kid normalweight - {len (values)} values for fitting", size = 12)
        fig.suptitle (feature, size = 15); fig.tight_layout ()
        plt.savefig (os.path.join (outputDir, f"{feature}_summary.png")); plt.close ()


