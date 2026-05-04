import os
import json
import argparse
import pandas as pd
import polars as pl
from flowsets import *
from functools import reduce

# python main_FlowSets.py --expression expression_directory --log2FC log2FC_directory --metadata metadata --config config --output outputDirectory


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--expression", type = str, required = True, help = "Directory for numerator and denominator fuzzy values")
    parser.add_argument ("--log2FC", type = str, required = True, help = "Directory for log2FC fuzzy values")
    parser.add_argument ("--metadata", type = str, required = True, help = "Metadata containing clustering information (TSV)")
    parser.add_argument ("--config", type = str, required = True, help = "Config file containing detailed parameters (JSON)")
    parser.add_argument ("--output", type = str, required = True, help = "Output directory for FlowSets plots and matrix of marker features")
    args = parser.parse_args ()

    class fuzzy:
        def __init__ (self, fuzzy_variables):
            self.terms = {fuzzy_var: list () for fuzzy_var in fuzzy_variables}

    with open (args.config) as f:
        config = json.load (f)
    exp_FS = config["expression_fuzzy_sets"]; log2FC_FS = config["log2FC_fuzzy_sets"]
    indexCol = config.get ("metadata_index_column", "index"); clusterCol = config.get ("metadata_cluster_column", "cluster")
    lfcCutoff = config["log2FC_flow_cutoff"]
    
    metadata = pd.read_csv (args.metadata, index_col = None, sep = "\t")
    if metadata.columns[0] == "Unnamed: 0":
        metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    clustering = pd.Series (metadata[clusterCol].values, index = metadata[indexCol].values); allClusters = sorted (set (clustering))

    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)
    if not os.path.exists (os.path.join (args.output, "FlowSets")):
        os.makedirs (os.path.join (args.output, "FlowSets"), exist_ok = True)

    numDir = os.path.join (args.expression, "numerator"); denDir = os.path.join (args.expression, "denominator")
    numerator = dict (); denominator = dict (); foldChangeDF = list ()
    for cluster in allClusters:
        featureList = sorted (set ([file[12:-4] for file in os.listdir (os.path.join (numDir, f"submatrix_{cluster}", "fuzzy_values"))
                                    if file.endswith (".tsv")]) &
                              set ([file[12:-4] for file in os.listdir (os.path.join (denDir, f"submatrix_{cluster}", "fuzzy_values"))
                                    if file.endswith (".tsv")]))
        sampleList = clustering[clustering == cluster].index
        for feature in featureList:
            numFV = pd.read_csv (os.path.join (numDir, f"submatrix_{cluster}", "fuzzy_values", f"fuzzyValues_{feature}.tsv"),
                                 index_col = 0, sep = "\t").mean (axis = 0)
            numFV = numFV.rename (index = {idx: f"{idx}*numerator" for idx in numFV.index})
            denFV = pd.read_csv (os.path.join (denDir, f"submatrix_{cluster}", "fuzzy_values", f"fuzzyValues_{feature}.tsv"),
                                 index_col = 0, sep = "\t").mean (axis = 0)
            denFV = denFV.rename (index = {idx: f"{idx}*denominator" for idx in denFV.index})
            numerator[f"{feature}__{cluster}"] = numFV; denominator[f"{feature}__{cluster}"] = denFV
        log2FC = [pd.read_csv (os.path.join (args.log2FC, f"fuzzyValues_{sample}.tsv"), index_col = 0, sep = "\t") for sample in sampleList]
        log2FC = (reduce (lambda x, y: x.add (y, fill_value = 0), log2FC) / len (log2FC))[log2FC_FS]
        foldChangeDF.append (log2FC.rename (columns = {col: f"{col}*{cluster}" for col in log2FC.columns}))
    expressionDF = pd.concat ([pd.DataFrame.from_dict (numerator, orient = "index"), pd.DataFrame.from_dict (denominator, orient = "index")], axis = 1)
    expressionDF = expressionDF.reset_index (names = "feature")
    foldChangeDF = pd.concat (foldChangeDF, axis = 1).reset_index (names = "feature")
    
    series2name = tuple ([("numerator", "numerator"), ("denominator", "denominator")])
    mfFuzzy = {"numerator": fuzzy (exp_FS), "denominator": fuzzy (exp_FS)}
    fa_exp = FlowAnalysis (pl.from_pandas (expressionDF), "feature", series2name, mfFuzzy, "*")
    downFlows = fa_exp.flow_finder ([">"], minLevels = ["NOISE", "NOISE"], maxLevels = ["HIGH", "HIGH"], verbose = False)
    expFV = fa_exp.calc_coarse_flow_memberships (use_edges = downFlows).to_pandas ().rename (columns = {"feature": "index"})
    expFV[["feature", "cluster"]] = expFV["index"].str.split ("__", expand = True)
    expFV = expFV.pivot (index = "feature", columns = "cluster", values = "membership")
    expFV = expFV.rename (columns = {col: f"{col}_DOWN" for col in expFV.columns})
    expression_FV = [expFV.rename_axis (None).rename_axis (None, axis = 1)]
    upFlows = fa_exp.flow_finder (["<"], minLevels = ["NOISE", "NOISE"], maxLevels = ["HIGH", "HIGH"], verbose = False)
    expFV = fa_exp.calc_coarse_flow_memberships (use_edges = upFlows).to_pandas ().rename (columns = {"feature": "index"})
    expFV[["feature", "cluster"]] = expFV["index"].str.split ("__", expand = True)
    expFV = expFV.pivot (index = "feature", columns = "cluster", values = "membership")
    expFV = expFV.rename (columns = {col: f"{col}_UP" for col in expFV.columns})
    expression_FV.append (expFV.rename_axis (None).rename_axis (None, axis = 1))
    expression_FV = pd.concat (expression_FV, axis = 1)
    highlight = {f: "steelblue" for f in downFlows}; highlight.update ({f: "firebrick" for f in upFlows})
    try:
        fa_exp.plot_flows (use_edges = downFlows | upFlows, specialColors = highlight, seriesFontsize = 8, classFontsize = 10,
                           title = "regulated features", outfile = os.path.join (args.output, "FlowSets", "expression_regulated"))
    except:
        pass

    series2name = tuple ([(cluster, cluster) for cluster in allClusters]); mfFuzzy = {cluster: fuzzy (log2FC_FS) for cluster in allClusters}
    fa_log2FC = FlowAnalysis (pl.from_pandas (foldChangeDF), "feature", series2name, mfFuzzy, "*")
    log2FC_FV = pd.DataFrame (index = foldChangeDF["feature"].values, dtype = float)
    for idx in range (len (allClusters)):
        cluster = allClusters[idx]; front = idx; back = len (allClusters) - idx - 1
        downFlows = fa_log2FC.flow_finder (["?"] * (len (allClusters) - 1), minLevels = ["NA"] * front + ["-INF"] + ["NA"] * back,
                                           maxLevels = ["+INF"] * front + ["-"] + ["+INF"] * back, verbose = False)
        upFlows = fa_log2FC.flow_finder (["?"] * (len (allClusters) - 1), minLevels = ["NA"] * front + ["+"] + ["NA"] * back,
                                         maxLevels = ["+INF"] * len (allClusters), verbose = False)
        highlight = {f: "steelblue" for f in downFlows - upFlows}; highlight.update ({f: "firebrick" for f in upFlows - downFlows})
        highlight.update ({f: "gray" for f in downFlows & upFlows})
        FV = fa_log2FC.calc_coarse_flow_memberships (use_edges = downFlows).to_pandas ().set_index ("feature")
        log2FC_FV[f"{cluster}_DOWN"] = FV.loc[expression_FV.index, "membership"]
        FV = fa_log2FC.calc_coarse_flow_memberships (use_edges = upFlows).to_pandas ().set_index ("feature")
        log2FC_FV[f"{cluster}_UP"] = FV.loc[expression_FV.index, "membership"]
        try:
            fa_log2FC.plot_flows (use_edges = downFlows | upFlows, specialColors = highlight, title = cluster, figsize = (12, 6), classFontsize = 8,
                                  outfile = os.path.join (args.output, "FlowSets", f"regulated_{cluster}"))
        except:
            continue

    specific = ((expression_FV > 0) & (log2FC_FV > lfcCutoff)).reset_index (names = "feature").melt (id_vars = "feature")
    specific = specific.loc[specific["value"]].drop ("value", axis = 1)
    result = log2FC_FV.reset_index (names = "feature").melt (id_vars = "feature", value_name = "membership").merge (specific, how = "right")
    result[["cluster", "regulation"]] = result["variable"].str.split ("_", expand = True)
    result = result[["feature", "cluster", "regulation", "membership"]].round (3)
    result = result.sort_values (["cluster", "feature", "regulation"]).reset_index (drop = True)
    result.to_csv (os.path.join (args.output, "cluster-specific_features.tsv"), index = None, sep = "\t")



if __name__ == "__main__":
    main ()


