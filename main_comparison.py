import os
import json
import argparse
import numpy as np
import pandas as pd

# python main_comparison.py --standard standardDirectory --raw rawDistanceDirectory --fuzzy fuzzyDistanceDirectory --DESeq2 DESeq2Directory --metadata metadata --config config --output outputDirectory


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--standard", type = str, required = True, help = "Directory of DESeq2 log2FC and padj files")
    parser.add_argument ("--raw", type = str, required = True, help = "Directory of raw log2FC fuzzy values")
    parser.add_argument ("--fuzzy", type = str, required = True, help = "Directory of fuzzy log2FC fuzzy values")
    parser.add_argument ("--DESeq2", type = str, required = True, help = "Directory of DESeq2 2-aspect fuzzification")
    parser.add_argument ("--metadata", type = str, required = True, help = "Metadata containing clustering column (TSV)")
    parser.add_argument ("--config", type = str, required = True, help = "Config file for comparison argmuents (JSON)")
    parser.add_argument ("--output", type = str, required = True, help = "Output directory for comparison files")
    args = parser.parse_args ()

    log2FC = pd.read_csv (os.path.join (args.standard, "DESeq2_log2FC.tsv"), index_col = 0, sep = "\t")
    padj = pd.read_csv (os.path.join (args.standard, "DESeq2_padj.tsv"), index_col = 0, sep = "\t")
    metadata = pd.read_csv (args.metadata, index_col = None, sep = "\t")
    if metadata.columns[0] == "Unnamed: 0":
        metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    with open (args.config) as f:
        config = json.load (f)
    
    indexCol = config.get ("metadata_index_column", "index"); clusterCol = config.get ("metadata_cluster_column", "cluster")
    allSets = config["fuzzy_variables"]
    log2FC_cutoff = config.get ("minimal_absolute_standard_log2FC", 1); padj_cutoff = config.get ("maximal_-log10_standard_padj", 1.3)
    avgFV_cutoff = config.get ("minimal_average_fuzzy_value", 0.5); pctMainFS_cutoff = config.get ("minimal_percent_main_fuzzy_set", 0.5)
    rawFV = list (); fuzzyFV = list ()
    for FS in allSets:
        memberships = pd.read_csv (os.path.join (args.raw, f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
        rawFV.append (memberships.to_numpy ())
    nameList = {"feature": list (memberships.index), "sample": list (memberships.columns)}
    for FS in allSets:
        memberships = pd.read_csv (os.path.join (args.fuzzy, "fuzzy_rule", f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
        fuzzyFV.append (memberships.loc[nameList["feature"], nameList["sample"]].to_numpy ())
    rawFV = np.einsum ("ijk -> jki", rawFV); fuzzyFV = np.einsum ("ijk -> jki", fuzzyFV)
    metadata = metadata.set_index (indexCol).loc[nameList["sample"]].reset_index (); allClusters = sorted (set (metadata[clusterCol]))
    clustering = metadata.rename (columns = {"index": "index_1"}).reset_index ().groupby (clusterCol)["index"].agg (list).to_dict ()

    results = list ()

    candidates = pd.DataFrame ("", index = nameList["feature"], columns = allClusters)
    candidates = candidates.mask ((log2FC < -log2FC_cutoff) & (padj > padj_cutoff), "--")
    candidates = candidates.mask ((log2FC > log2FC_cutoff) & (padj > padj_cutoff), "++")
    candidates = candidates.reset_index (names = "feature").melt (id_vars = "feature", var_name = "cluster", value_name = "regulation")
    candidates = candidates.loc[candidates["regulation"] != ""]
    if not candidates.empty:
        candidates["method"] = "DESeq2 standard"
        results.append (candidates)
    
    allPctMainFS = pd.DataFrame (rawFV.argmax (axis = 2), index = nameList["feature"], columns = nameList["sample"])
    allPctMainFS = allPctMainFS.reset_index ().melt (id_vars = "index")
    allPctMainFS["cluster"] = metadata.set_index (indexCol).loc[allPctMainFS["variable"], clusterCol].values
    allPctMainFS = allPctMainFS.groupby (["index", "cluster"])["value"].value_counts (normalize = True).reset_index ()
    allPctMainFS["value"] = allPctMainFS["value"].replace (dict (zip (range (len (allSets)), allSets)))
    candidates = pd.DataFrame ("", index = nameList["feature"], columns = allClusters)
    for cluster in allClusters:
        avgFV = pd.DataFrame (rawFV[:, clustering[cluster], :].mean (axis = 1), index = nameList["feature"], columns = allSets)
        pctMainFS = allPctMainFS.loc[allPctMainFS["cluster"] == cluster].pivot (index = "index", columns = "value", values = "proportion")
        pctMainFS = pctMainFS.rename_axis (None, axis = 0).rename_axis (None, axis = 1)
        pctMainFS = pd.concat ([pctMainFS, pd.DataFrame (0, index = nameList["feature"], columns = list (set (allSets) - set (pctMainFS.columns)))], axis = 1)
        pctMainFS = pctMainFS[allSets].replace (np.nan, 0)
        candidates.loc[(avgFV["-INF"] + avgFV["--"] + avgFV["-"] > avgFV_cutoff) & (pctMainFS["-INF"] + pctMainFS["--"] + pctMainFS["-"] > 0.8 * pctMainFS_cutoff), cluster] = "--"
        candidates.loc[(avgFV["INF"] + avgFV["++"] + avgFV["+"] > avgFV_cutoff) & (pctMainFS["INF"] + pctMainFS["++"] + pctMainFS["+"] > 0.8 * pctMainFS_cutoff), cluster] = "++"
    candidates = candidates.reset_index (names = "feature").melt (id_vars = "feature", var_name = "cluster", value_name = "regulation")
    candidates = candidates.loc[candidates["regulation"] != ""]
    if not candidates.empty:
        candidates["method"] = "raw log2FC"
        results.append (candidates)

    allPctMainFS = pd.DataFrame (fuzzyFV.argmax (axis = 2), index = nameList["feature"], columns = nameList["sample"])
    allPctMainFS = allPctMainFS.reset_index ().melt (id_vars = "index")
    allPctMainFS["cluster"] = metadata.set_index (indexCol).loc[allPctMainFS["variable"], clusterCol].values
    allPctMainFS = allPctMainFS.groupby (["index", "cluster"])["value"].value_counts (normalize = True).reset_index ()
    allPctMainFS["value"] = allPctMainFS["value"].replace (dict (zip (range (len (allSets)), allSets)))
    candidates = pd.DataFrame ("", index = nameList["feature"], columns = allClusters)
    for cluster in allClusters:
        avgFV = pd.DataFrame (fuzzyFV[:, clustering[cluster], :].mean (axis = 1), index = nameList["feature"], columns = allSets)
        pctMainFS = allPctMainFS.loc[allPctMainFS["cluster"] == cluster].pivot (index = "index", columns = "value", values = "proportion")
        pctMainFS = pctMainFS.rename_axis (None, axis = 0).rename_axis (None, axis = 1)
        pctMainFS = pd.concat ([pctMainFS, pd.DataFrame (0, index = nameList["feature"], columns = list (set (allSets) - set (pctMainFS.columns)))], axis = 1)
        pctMainFS = pctMainFS[allSets].replace (np.nan, 0)
        candidates.loc[(avgFV["-INF"] + avgFV["--"] + avgFV["-"] > 1.5 * avgFV_cutoff) & (pctMainFS["-INF"] + pctMainFS["--"] + pctMainFS["-"] > pctMainFS_cutoff), cluster] = "--"
        candidates.loc[(avgFV["INF"] + avgFV["++"] + avgFV["+"] > 1.5 * avgFV_cutoff) & (pctMainFS["INF"] + pctMainFS["++"] + pctMainFS["+"] > pctMainFS_cutoff), cluster] = "++"
    candidates = candidates.reset_index (names = "feature").melt (id_vars = "feature", var_name = "cluster", value_name = "regulation")
    candidates = candidates.loc[candidates["regulation"] != ""]
    if not candidates.empty:
        candidates["method"] = "fuzzy rule"
        results.append (candidates)
    
    log2FC_FV = list (); padj_FV = list ()
    for FS in ["--", "-", "o", "+", "++"]:
        memberships = pd.read_csv (os.path.join (args.DESeq2, "log2FC", f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
        log2FC_FV.append (memberships.to_numpy ())
    nameList = {"feature": list (memberships.index), "sample": list (memberships.columns)}
    for FS in ["o", "*", "**", "***", "****"]:
        memberships = pd.read_csv (os.path.join (args.DESeq2, "padj", f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
        padj_FV.append (memberships.loc[nameList["feature"], nameList["sample"]].to_numpy ())
    log2FC_FV = np.einsum ("ijk -> jki", log2FC_FV); padj_FV = np.einsum ("ijk -> jki", padj_FV)
    candidates = pd.DataFrame ("", index = nameList["feature"], columns = allClusters)
    candidates = candidates.mask (((log2FC_FV[:, :, 0] > 0) | (log2FC_FV[:, :, 1] == 1)) & (padj_FV[:, :, 4] > 0), "--")
    candidates = candidates.mask (((log2FC_FV[:, :, 3] == 1) | (log2FC_FV[:, :, 4] > 0)) & (padj_FV[:, :, 4] > 0), "++")
    candidates = candidates.reset_index (names = "feature").melt (id_vars = "feature", var_name = "cluster", value_name = "regulation")
    candidates = candidates.loc[candidates["regulation"] != ""]
    if not candidates.empty:
        candidates["method"] = "DESeq2 2-aspect"
        results.append (candidates)

    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)
    if len (results) == 0:
        print ("No results!")
    else:
        results = pd.concat (results, axis = 0, ignore_index = True).sort_values (["cluster", "feature", "method"]).reset_index (drop = True)
        results.to_csv (os.path.join (args.output, "comparison_results.tsv"), index = None, sep = "\t")



if __name__ == "__main__":
    main ()


