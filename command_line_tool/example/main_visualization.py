import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# python main_visualization.py --baseDir baseDirectory --params parameterConfigPrefix --config config --output outputDirectory


def parseConfig (config):
    const = {"-Infinity": -np.inf, "-infinity": -np.inf, "-Inf": -np.inf, "-inf": -np.inf,
             "+Infinity": np.inf, "+infinity": np.inf, "+Inf": np.inf, "+inf": np.inf,
             "Infinity": np.inf, "infinity": np.inf, "Inf": np.inf, "inf": np.inf,
             "NaN": np.nan, "NAN": np.nan, "nan": np.nan, "NA": np.nan, "na": np.nan}
    if config.get ("define_concept_by", "default") == "default":
        numFS = 2 * config["parameters_default"]["number_fuzzy_sets_per_side"] + 1
        params = [config["parameters_default"].get ("width_scale_factor", 1), config["parameters_default"].get ("slope_percentage", 0.5)]
    else:
        numFS = len (config["parameters_constraint"]["constraints"])
        params = list ()
    savedBy = config.get ("fuzzify_per", "feature")
    labelSets = config.get ("rename_labels", dict ()); labelSets = [labelSets[val] for val in labelSets.keys () if isinstance (val, str)]
    cutoffCons = config.get ("left_noise_cutoff_constant", -np.inf); cutoffPct = config.get ("left_noise_cutoff_percent", 0)
    cutoffCons = const.get (cutoffCons, cutoffCons) if isinstance (cutoffCons, str) else cutoffCons
    if np.isfinite (cutoffCons) or cutoffPct > 0:
        labelSets.append ("MIN-NOISE")
    cutoffCons = config.get ("right_noise_cutoff_constant", np.inf); cutoffPct = config.get ("right_noise_cutoff_percent", 0)
    cutoffCons = const.get (cutoffCons, cutoffCons) if isinstance (cutoffCons, str) else cutoffCons
    if np.isfinite (cutoffCons) or cutoffPct < 1:
        labelSets.append ("MAX-NOISE")
    allSets = config.get ("fuzzy_variables", [f"FS{i}" for i in range (1, numFS + 1)])
    res = {"number_fuzzy_sets": numFS, "label_sets": labelSets, "fuzzy_sets": allSets, "saved_by": savedBy, "reconstruct_params": params}
    return res



def readFV (dir, savedBy, nameDict = dict ()):
    allFV = list ()
    if savedBy == "feature":
        featureList = nameDict["feature"] if "feature" in nameDict else sorted ([file[12:-4] for file in os.listdir (dir) if file.endswith (".tsv")])
        for feature in featureList:
            FV = pd.read_csv (os.path.join (dir, f"fuzzyValues_{feature}.tsv"), index_col = 0, sep = "\t")
            if "sample" in nameDict and "FS" in nameDict:
                allFV.append (FV.loc[nameDict["sample"], nameDict["FS"]].to_numpy ())
            else:
                allFV.append (FV.to_numpy ())
        allFV = np.array (allFV); nameDict = {"feature": featureList, "sample": list (FV.index), "FS": list (FV.columns)}
    elif savedBy == "sample":
        sampleList = nameDict["sample"] if "sample" in nameDict else sorted ([file[12:-4] for file in os.listdir (dir) if file.endswith (".tsv")])
        for sample in sampleList:
            FV = pd.read_csv (os.path.join (dir, f"fuzzyValues_{sample}.tsv"), index_col = 0, sep = "\t")
            if "feauture" in nameDict and "FS" in nameDict:
                allFV.append (FV.loc[nameDict["feature"], nameDict["FS"]].to_numpy ())
            else:
                allFV.append (FV.to_numpy ())
        allFV = np.einsum ("ijk -> jik", allFV); nameDict = {"feature": list (FV.index), "sample": sampleList, "FS": list (FV.columns)}
    else:
        allSets = nameDict["FS"] if "FS" in nameDict else sorted ([file[12:-4] for file in os.listdir (dir) if file.endswith (".tsv")])
        for FS in allSets:
            FV = pd.read_csv (os.path.join (dir, f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
            if "feature" in nameDict and "sample" in nameDict:
                allFV.append (FV.loc[nameDict["feature"], nameDict["sample"]].to_numpy ())
            else:
                allFV.append (FV.to_numpy ())
        allFV = np.einsum ("ijk -> jki", allFV); nameDict = {"feature": list (FV.index), "sample": list (FV.columns), "FS": allSets}
    return allFV, nameDict



def plot_cardinality (allFV, nameDict, clustering, outputPath):
    allClusters = sorted (set (clustering))
    mainFS = pd.DataFrame (allFV.argmax (axis = 2), index = nameDict["feature"], columns = nameDict["sample"])
    mainFS = mainFS.replace (dict (zip (range (len (nameDict["FS"])), nameDict["FS"])))
    pctMainFS = pd.DataFrame (0, index = allClusters, columns = nameDict["FS"], dtype = float)
    for cluster in allClusters:
        tmp = mainFS[clustering[clustering == cluster].index]
        pctMainFS.loc[cluster] = tmp.melt ()["value"].value_counts (normalize = True)
    fig, ax = plt.subplots (figsize = (10, 4))
    sns.lineplot (pctMainFS.reset_index (names = "cluster").melt (id_vars = "cluster", var_name = "fuzzy set"), x = "fuzzy set", y = "value",
                  hue = "cluster", hue_order = allClusters, style = "cluster", markers = True, dashes = False, ax = ax)
    ax.set_xlabel (""); ax.set_ylabel ("percent of raw values", size = 10); ax.legend (title = None)
    fig.tight_layout (); plt.savefig (outputPath); plt.close ()



def getDefaultConcept (numFS, widthFct, slopeFct):
    coords = [widthFct * (i + overlap) for i in np.linspace (-numFS, numFS, numFS + 1) for overlap in [-slopeFct, slopeFct]]
    concept = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFS + 1)], 3).tolist ()
    concept[numFS // 2] = [0, 1]
    return concept


def plot_concept (concept, allSets, colors, xRange, xLabel, outputPath):
    xValues = np.linspace (*xRange, 1000); concept_copy = concept.copy ()
    concept_copy[0][0] = xRange[0] - 1; concept_copy[0][1] = xRange[0] - 1
    concept_copy[-1][2] = xRange[1] + 1; concept_copy[-1][3] = xRange[1] + 1
    fig, ax = plt.subplots (figsize = (6, 3))
    for i in range (len (concept)):
        c = concept_copy[i]
        if len (c) == 2:
            ax.plot (xValues, stats.norm.pdf (xValues, loc = c[0], scale = c[1]) * (c[1] * np.sqrt (2 * np.pi)),
                     color = colors[i], label = allSets[i])
        else:
            ax.plot ((c[0], c[1]), (0, 1), colors[i]); ax.plot ((c[1], c[2]), (1, 1), colors[i])
            ax.plot ((c[2], c[3]), (1, 0), colors[i], label = allSets[i])
    ax.set_xlabel (xLabel, size = 10); ax.set_ylabel ("fuzzy value", size = 10)
    ax.set_xlim (xRange); ax.set_ylim ((0, 1.05)); ax.legend (loc = (1.01, 0.25))
    fig.tight_layout (); plt.savefig (outputPath); plt.close ()



def plotVolcano (log2FC, padj, log2FC_concept, padj_concept, outputPath):
    xRange = [np.floor (log2FC.mask (~np.isfinite (log2FC)).min (axis = None, skipna = True)),
              np.ceil (log2FC.mask (~np.isfinite (log2FC)).max (axis = None, skipna = True))]
    yRange = [np.floor (padj.mask (~np.isfinite (padj)).min (axis = None, skipna = True)) - 0.05,
              np.ceil (padj.mask (~np.isfinite (padj)).max (axis = None, skipna = True)) + 0.1]
    mu, sigma = log2FC_concept[2]
    log2FC_cutoff = [xRange[0], (log2FC_concept[0][2] + log2FC_concept[0][3]) / 2]
    xValues = np.linspace (log2FC_concept[1][2], log2FC_concept[3][1], 1000)
    values = pd.DataFrame ({"x": xValues, "left": ((xValues - log2FC_concept[1][3]) / (log2FC_concept[1][2] - log2FC_concept[1][3])).clip (min = 0),
                            "middle": np.exp (-(xValues - mu) ** 2 / (2 * sigma ** 2)),
                            "right": ((xValues - log2FC_concept[3][0]) / (log2FC_concept[3][1] - log2FC_concept[3][0])).clip (min = 0)})
    values["diff1"] = values["left"] - values["middle"]; values["diff2"] = values["middle"] - values["right"]
    log2FC_cutoff += [values.loc[values["diff1"].abs ().sort_values ().index].iloc[:2, 0].mean (),
                      values.loc[values["diff2"].abs ().sort_values ().index].iloc[:2, 0].mean ()]
    log2FC_cutoff += [(log2FC_concept[4][0] + log2FC_concept[4][1]) / 2, xRange[1]]
    yRange = [np.floor (padj.mask (~np.isfinite (padj)).min (axis = None, skipna = True)),
              np.ceil (padj.mask (~np.isfinite (padj)).max (axis = None, skipna = True)) + 1]
    padj_cutoff = [yRange[0]] + np.array (padj_concept)[:-1, -2:].mean (axis = 1).tolist () + [yRange[1]]
    pltData = log2FC.reset_index ().melt (id_vars = "index", value_name = "log2FC")
    pltData = pltData.merge (padj.reset_index ().melt (id_vars = "index", value_name = "padj"), on = ["index", "variable"], how = "inner")
    pltData["class"] = ""; coords = dict ()
    for (i, j) in itertools.product (range (5), range (5)):
        mask = (pltData["log2FC"] > log2FC_cutoff[i]) & (pltData["log2FC"] < log2FC_cutoff[i + 1]) & (pltData["padj"] > padj_cutoff[j]) & (pltData["padj"] < padj_cutoff[j + 1])
        pltData.loc[mask, "class"] = f"{i}_{j}"; coords[f"{i}_{j}"] = [(log2FC_cutoff[i] + log2FC_cutoff[i + 1]) / 2, (padj_cutoff[j] + padj_cutoff[j + 1]) / 2]
    labels = pltData.value_counts ("class"); labels = {key: f"{(labels.get (key, 0) / pltData.shape[0]):.2%}\n({labels.get (key, 0)})" for key in coords.keys ()}
    pltData = log2FC.reset_index ().melt (id_vars = "index", value_name = "log2FC")
    pltData = pltData.merge (padj.reset_index ().melt (id_vars = "index", value_name = "padj"), on = ["index", "variable"], how = "inner")
    fig, ax = plt.subplots (figsize = (10, 7))
    sns.scatterplot (pltData, x = "log2FC", y = "padj", color = "silver", legend = None, ax = ax)
    ax.set_xlim (xRange); ax.set_ylim (yRange)
    for val in log2FC_cutoff[1:-1]:
        ax.axvline (val, color = "black", linestyle = "dashed")
    for val in padj_cutoff[1:-1]:
        ax.axhline (val, color = "black", linestyle = "dashed")
    for key in coords.keys ():
        ax.text (*coords[key], labels[key], fontdict = {"size": 9, "ha": "center", "weight": "bold"})
    ax.set_xlabel ("DESeq2 average log2 fold change", size = 10); ax.set_ylabel ("-log10 (DESeq2 corrected p-value)", size = 10)
    ax.set_title (f"{log2FC.shape[0]} features in {log2FC.shape[1]} clusters / comparisons", size = 12)
    fig.tight_layout (); plt.savefig (outputPath)



def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--baseDir", type = str, required = True, help = "Base directory for raw value matrices and fuzzy values")
    parser.add_argument ("--config", type = str, required = True, help = "Config file for detailed parameters and colors (JSON)")
    parser.add_argument ("--output", type = str, required = True, help = "Output directory for visualizations")
    args = parser.parse_args ()

    configDir = os.path.dirname (args.config)
    with open (args.config) as f:
        config = json.load (f)
    indexCol = config.get ("metadata_index_column", "index"); clusterCol = config.get ("metadata_cluster_column", "cluster")
    colorDict = config["colors"]

    metadata = pd.read_csv (os.path.join (args.baseDir, "data", "metadata.tsv"), index_col = None, sep = "\t")
    if metadata.columns[0] == "Unnamed: 0":
        metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    clustering = pd.Series (metadata[clusterCol].values, index = metadata[indexCol].values); allClusters = sorted (set (clustering))
    log2FC = pd.read_csv (os.path.join (args.baseDir, "data", "DESeq2_log2FC.tsv"), index_col = 0, sep = "\t")
    padj = pd.read_csv (os.path.join (args.baseDir, "data", "DESeq2_padj.tsv"), index_col = 0, sep = "\t")

    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)

    with open (os.path.join (configDir, "expression.json")) as f:
        exp_config = parseConfig (json.load (f))
    concept = getDefaultConcept (exp_config["number_fuzzy_sets"], *exp_config["reconstruct_params"])
    plot_concept (concept, exp_config["fuzzy_sets"], [colorDict[FS] for FS in exp_config["fuzzy_sets"]], [-6, 6], "z-score",
                  os.path.join (args.output, "concept_expression.png"))

    with open (os.path.join (configDir, "RFC.json")) as f:
        RFC_config = parseConfig (json.load (f))
    rawFC_FV, log2FC_nameDict = readFV (os.path.join (args.baseDir, "FV_paired_log2FC", "fuzzy_values"), RFC_config["saved_by"],
                                        nameDict = {"FS": RFC_config["label_sets"] + RFC_config["fuzzy_sets"]})
    concept = getDefaultConcept (RFC_config["number_fuzzy_sets"], *RFC_config["reconstruct_params"])
    plot_concept (concept, RFC_config["fuzzy_sets"], [colorDict[FS] for FS in RFC_config["fuzzy_sets"]], [-6, 6], "z-score",
                  os.path.join (args.output, "concept_raw_log2FC.png"))
    plot_cardinality (rawFC_FV, log2FC_nameDict, clustering, os.path.join (args.output, "cardinality_rawFC.png"))

    with open (os.path.join (configDir, "FFC.json")) as f:
        FFC_config = parseConfig (json.load (f))
    log2FC_nameDict = {"feature": log2FC_nameDict["feature"], "sample": log2FC_nameDict["sample"],
                       "FS": FFC_config["label_sets"] + FFC_config["fuzzy_sets"]}
    fuzzyFC_FV, _ = readFV (os.path.join (args.baseDir, "FV_fuzzy_log2FC"), FFC_config["saved_by"], nameDict = log2FC_nameDict)
    plot_cardinality (fuzzyFC_FV, log2FC_nameDict, clustering, os.path.join (args.output, "cardinality_fuzzyFC.png"))

    with open (os.path.join (configDir, "DESeq2_log2FC.json")) as f:
        DESeq2_config = parseConfig (json.load (f))
    log2FC_FV, DESeq2_nameDict = readFV (os.path.join (args.baseDir, "FV_DESeq2", "log2FC", "fuzzy_values"), DESeq2_config["saved_by"],
                                         nameDict = {"FS": DESeq2_config["label_sets"] + DESeq2_config["fuzzy_sets"]})
    with open (os.path.join (args.baseDir, "FV_DESeq2", "concepts_log2FC", "concepts_detailed.json")) as f:
        log2FC_concept = json.load (f)["DEFAULT"]
    log2FC_concept = [log2FC_concept[FS][0] for FS in DESeq2_config["fuzzy_sets"]]
    plot_concept (log2FC_concept, DESeq2_config["fuzzy_sets"], [colorDict[FS] for FS in DESeq2_config["fuzzy_sets"]],
                  [-6, 6], "average log2 fold change", os.path.join (args.output, "concept_DESeq2_log2FC.png"))
    plot_cardinality (log2FC_FV, DESeq2_nameDict, pd.Series (allClusters, index = allClusters), os.path.join (args.output, "cardinality_DESeq2_log2FC.png"))

    with open (os.path.join (configDir, "DESeq2_padj.json")) as f:
        DESeq2_config = parseConfig (json.load (f))
    padj_FV, DESeq2_nameDict = readFV (os.path.join (args.baseDir, "FV_DESeq2", "padj", "fuzzy_values"), DESeq2_config["saved_by"],
                                       nameDict = {"feature": DESeq2_nameDict["feature"],"sample": DESeq2_nameDict["sample"],
                                                   "FS": DESeq2_config["label_sets"] + DESeq2_config["fuzzy_sets"]})
    with open (os.path.join (args.baseDir, "FV_DESeq2", "concepts_padj", "concepts_detailed.json")) as f:
        padj_concept = json.load (f)["DEFAULT"]
    padj_concept = [padj_concept[FS][0] for FS in DESeq2_config["fuzzy_sets"]]
    plot_concept (padj_concept, DESeq2_config["fuzzy_sets"], [colorDict[FS] for FS in DESeq2_config["fuzzy_sets"]],
                  [0, 10], "-log10 (corrected p-value)", os.path.join (args.output, "concept_DESeq2_padj.png"))
    plot_cardinality (padj_FV, DESeq2_nameDict, pd.Series (allClusters, index = allClusters), os.path.join (args.output, "cardinality_DESeq2_padj.png"))

    plotVolcano (log2FC, padj, log2FC_concept, padj_concept, os.path.join (args.output, "DESeq2_volcano.png"))



if __name__ == "__main__":
    main ()


