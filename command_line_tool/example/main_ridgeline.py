import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# python main_ridgeline.py --data rawDataDirectory --concepts fuzzyConceptDirectory --result selectionResultDirectory \
#        --indexCol metadataIndexColumn --clusterCol metadataClusterColumn -defaultName defaultConceptName --output outputDirectory


def plotRidgeline (candidates, numerator, denominator, log2FC, clustering, parameters, allMethods, outputPath = "./"):
    if not os.path.exists (outputPath):
        os.makedirs (outputPath, exist_ok = True)
    allClusters = sorted (set (clustering))
    tmpCand = candidates.sort_values ("feature").set_index ("feature")
    tmpCand["method"] = tmpCand["method"].replace (dict (zip (allMethods, [str (i) for i in range (len (allMethods))]))).astype (int)
    for feature in sorted (set (candidates["feature"])):
        featureData = tmpCand.loc[feature]
        clusters = [featureData["cluster"]] if isinstance (featureData, pd.Series) else list (set (featureData["cluster"]))
        pltData = pd.DataFrame ({"tumor": numerator.loc[feature], "normal": denominator.loc[feature], "log2FC": log2FC.loc[feature], "cluster": clustering})
        pltData = pltData.replace ([-np.inf, np.inf], np.nan).dropna ()
        if pltData.empty:
            continue
        xRange = [np.floor (min (pltData["tumor"].min (), pltData["normal"].min ())), np.ceil (max (pltData["tumor"].max (), pltData["normal"].max ()))]
        log2FC_range = [np.floor (pltData["log2FC"].min ()), np.ceil (pltData["log2FC"].max ())]; xValues = np.linspace (*xRange, 1000)
        fig, axs = plt.subplots (len (allClusters), 2, sharex = False, sharey = False, layout = "constrained", figsize = (6, 1.5 * len (allClusters)))
        for idx in range (len (allClusters)):
            tmp = pltData.loc[pltData["cluster"] == allClusters[idx]]
            if isinstance (featureData, pd.Series):
                methods = [featureData["method"]] if featureData["cluster"] == allClusters[idx] else list ()
            else:
                methods = featureData.loc[featureData["cluster"] == allClusters[idx], "method"].tolist ()
            axs[idx, 0].hist (tmp["tumor"], bins = 20, color = "tab:red", alpha = 0.6); axs[idx, 0].hist (tmp["normal"], bins = 20, color = "tab:blue", alpha = 0.6)
            axs[idx, 0].set_xlim (xRange)
            axs[idx, 0].text (0.05, 0.8, allClusters[idx], transform = axs[idx, 0].transAxes, bbox = {"facecolor": "none", "edgecolor": "silver"})
            ax = axs[idx, 0].twinx (); mu, sigma = parameters[allClusters[idx]][feature]
            yValues = stats.norm.pdf (xValues, loc = mu, scale = sigma); ax.set_ylim ((0, 1.05 * yValues.max ()))
            ax.plot (xValues, yValues, color = "black", linewidth = 2, label = f"mu: {mu:.2f}\nsigma: {sigma:.2f}")
            ax.legend (loc = "upper right", fontsize = 7)
            axs[idx, 1].hist (tmp["log2FC"], bins = 20, color = "gray", alpha = 0.6)
            axs[idx, 1].set_xlim (log2FC_range)
            axs[idx, 1].text (0.05, 0.8, allClusters[idx], transform = axs[idx, 1].transAxes, bbox = {"facecolor": "none", "edgecolor": "silver"})
            if allClusters[idx] in clusters:
                axs[idx, 0].set_facecolor ("lavender"); axs[idx, 1].set_facecolor ("lavender")
            for i in range (len (allMethods)):
                if i in methods:
                    axs[idx, 1].scatter (list (), list (), s = 30, c = "black", edgecolors = "black", marker = "o", label = " ")
                else:
                    axs[idx, 1].scatter (list (), list (), s = 30, c = "white", edgecolors = "black", marker = "o", label = " ")
            axs[idx, 1].legend (loc = (0.75, 0.9), ncol = len (allMethods), fontsize = 2, frameon = False)
        axs[idx, 0].set_xlabel ("log2 count\nprimary tumor: red\nsolid normal tissue: blue", size = 10); axs[idx, 1].set_xlabel ("log2 fold change", size = 10)
        fig.supylabel ("number of sample pairs", size = 10); fig.suptitle (feature, size = 12.5)
        plt.savefig (os.path.join (outputPath, f"ridgeline_{feature}.png")); plt.close ()



def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--data", type = str, required = True, help = "Directory for raw value matrices as well as log2 fold change matrix")
    parser.add_argument ("--concepts", type = str, required = True, help = "Directory for fuzzy concepts for raw expression fuzzification")
    parser.add_argument ("--result", type = str, required = True, help = "Directory for FlowSets selection results")
    parser.add_argument ("--indexCol", type = str, required = False, default = "index", help = "Column name for sample names in metadata")
    parser.add_argument ("--clusterCol", type = str, required = False, default = "cluster", help = "Column name for clusters in metadata")
    parser.add_argument ("--defaultName", type = str, required = False, default = "DEFAULT", help = "Name of the default fuzzy concept")
    parser.add_argument ("--output", type = str, required = True, help = "Output directory for ridgeline plots")
    args = parser.parse_args ()

    numerator = pd.read_csv (os.path.join (args.data, "numerator_log.tsv"), index_col = 0, sep = "\t")
    denominator = pd.read_csv (os.path.join (args.data, "denominator_log.tsv"), index_col = 0, sep = "\t")
    log2FC = pd.read_csv (os.path.join (args.data, "paired_log2FC.tsv"), index_col = 0, sep = "\t")
    metadata = pd.read_csv (os.path.join (args.data, "metadata.tsv"), index_col = None, sep = "\t")
    if metadata.columns[0] == "Unnamed: 0":
        metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    clustering = pd.Series (metadata[args.clusterCol].values, index = metadata[args.indexCol].values)

    params = dict (); defaultName = args.defaultName
    for cluster in sorted (set (clustering)):
        with open (os.path.join (args.concepts, f"submatrix_{cluster}", "concepts_detailed.json")) as f:
            concepts = json.load (f)
        default = concepts.get (defaultName, dict ()).get ("MEDIUM", [list (), "", "", 0])
        params[cluster] = {feature: concepts.get (feature, dict ()).get ("MEDIUM", default)[0] for feature in log2FC.index}

    tmp = pd.read_csv (os.path.join (args.result, "raw_log2FC", "cluster-specific_features.tsv"), index_col = None, sep = "\t")
    tmp["method"] = "raw log2FC"; candidates = [tmp]
    tmp = pd.read_csv (os.path.join (args.result, "fuzzy_log2FC", "cluster-specific_features.tsv"), index_col = None, sep = "\t")
    tmp["method"] = "fuzzy log2FC"; candidates.append (tmp); del tmp
    candidates = pd.concat (candidates, axis = 0, ignore_index = True)
    allMethods = ["raw log2FC", "fuzzy log2FC"]

    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)
    plotRidgeline (candidates, numerator, denominator, log2FC, clustering, params, allMethods, outputPath = args.output)



if __name__ == "__main__":
    main ()


