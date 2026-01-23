import os
import json
import argparse
import numpy as np
import pandas as pd
from visualization import plot_concept, heatmap_1dim, heatmap_2aspect

# python main_visualization.py --data rawDataDirectory --result resultDirectory --metadata metadata --config config --output outputDirectory


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--data", type = str, required = True, help = "Directory for splitted matrices")
    parser.add_argument ("--result", type = str, required = True, help = "Directory of comparison tables as results of the analysis pipeline")
    parser.add_argument ("--metadata", type = str, required = True, help = "Metadata containing clustering column (TSV)")
    parser.add_argument ("--config", type = str, required = True, help = "Config file for fuzzy value directories and parameters (JSON)")
    parser.add_argument ("--output", type = str, required = True, help = "Output directory for visualizations")
    args = parser.parse_args ()

    log2FC = pd.read_csv (os.path.join (args.data, "paired_log2FC.tsv"), index_col = 0, sep = "\t")
    candidates = pd.read_csv (os.path.join (args.result, "comparison_results.tsv"), index_col = None, sep = "\t")
    validated_markers = pd.read_csv (os.path.join (args.result, "comparison_validated_markers.tsv"), index_col = None, sep = "\t").replace (np.nan, "")
    numMethods = validated_markers.value_counts ("feature")
    common_markers = validated_markers.loc[validated_markers["feature"].isin (numMethods[numMethods == 4].index), ["feature", "cluster"]].drop_duplicates ()
    metadata = pd.read_csv (args.metadata, index_col = None, sep = "\t")
    if metadata.columns[0] == "Unnamed: 0":
        metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    with open (args.config) as f:
        config = json.load (f)

    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)
    
    indexCol = config.get ("metadata_index_column", "index"); clusterCol = config.get ("metadata_cluster_column", "cluster"); colorDict = config["color"]
    clustering = pd.Series (metadata[clusterCol].values, index = metadata[indexCol].values)[log2FC.columns]
    allClusters = sorted (set (clustering))

    allSets = ["--", "-", "o", "+", "++"]
    coords = [(i + overlap) for i in np.linspace (-5, 5, 6) for overlap in [-0.5, 0.5]]
    concept = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, 6)], 3).tolist (); concept[2] = [0, 1]
    plot_concept (concept, allSets, [colorDict[FS] for FS in allSets], [-6, 6], "z-score", os.path.join (args.output, "concept_fitted_log2FC.png"))
    plot_concept (concept, ["LOW", "low", "MEDIUM", "high", "HIGH"], [colorDict[FS] for FS in allSets], [-6, 6], "z-score",
                  os.path.join (args.output, "concept_fitted_rawExpression.png"))

    with open (os.path.join (config["DESeq2 2-aspect"], "concepts_DESeq2_log2FC.json")) as f:
        concept = json.load (f)
    plot_concept (concept["ALL"][allClusters[0]], allSets, [colorDict[FS] for FS in allSets], [-5, 5], "DESeq2 log2 fold change",
                  os.path.join (args.output, "concept_DESeq2_log2FC.png"))
    allSets = ["o", "*", "**", "***", "****"]
    with open (os.path.join (config["DESeq2 2-aspect"], "concepts_DESeq2_padj.json")) as f:
        concept = json.load (f)
    plot_concept (np.array (list (concept["ALL"].values ())[0]), allSets, [colorDict[FS] for FS in allSets], [0, 10], "-log10 (DESeq2 corrected p-value)",
                  os.path.join (args.output, "concept_DESeq2_padj.png"))

    for method in sorted (set (candidates["method"])):
        dir = config.get (method, "")
        if dir == "":
            continue
        if method.startswith ("DESeq2"):
            avgLog2FC_FS = config["DESeq2_fold_change_fuzzy_variables"]; padj_FS = config["DESeq2_p-value_fuzzy_variables"]
            avgLog2FC_FV = list (); padj_FV = list ()
            for FS in avgLog2FC_FS:
                memberships = pd.read_csv (os.path.join (dir, "log2FC", f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
                avgLog2FC_FV.append (memberships.loc[common_markers["feature"]].to_numpy ())
            nameDict = {"feature": (common_markers["feature"] + "__" + common_markers["cluster"]).tolist (), "sample": list (memberships.columns)}
            for FS in padj_FS:
                memberships = pd.read_csv (os.path.join (dir, "padj", f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
                padj_FV.append (memberships.loc[common_markers["feature"], nameDict["sample"]].to_numpy ())
            heatmap_2aspect (np.einsum ("ijk -> jki", avgLog2FC_FV), np.einsum ("ijk -> jki", padj_FV), avgLog2FC_FS, padj_FS,
                             nameDict, colorDict, os.path.join (args.output, "common_markers_DESeq2_2-aspect.png"))
        else:
            allFV = list (); allSets = config["fuzzy_variables"]
            for FS in allSets:
                memberships = pd.read_csv (os.path.join (dir, f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
                allFV.append (memberships.loc[common_markers["feature"]].to_numpy ())
            nameDict = {"feature": (common_markers["feature"] + "__" + common_markers["cluster"]).tolist (), "sample": list (memberships.columns)}
            heatmap_1dim (np.einsum ("ijk -> jki", allFV), allSets, nameDict, clustering[memberships.columns], colorDict, method,
                          os.path.join (args.output, f"common_markers_{method.replace (" ", "_")}.png"))



if __name__ == "__main__":
    main ()


