import os
import json
import argparse
import numpy as np
import pandas as pd

### python main_rawFoldChange.tsv --mtx rawValueMatrix --metadata metadata --config config --centralize --output outputDirectory


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--mtx", type = str, required = True, help = "Raw value matrix (TSV)")
    parser.add_argument ("--metadata", type = str, required = True, help = "Metadata containing clustering and comparison columns (TSV)")
    parser.add_argument ("--config", type = str, required = True, help = "Config file for fold change calculation (JSON)")
    parser.add_argument ("--centralize", required = False, action = "store_true", help = "Whether to centralize log2 fold change")
    parser.add_argument ("--output", type = str, required = True, help = "Output directory for splitted matrices")
    args = parser.parse_args ()

    mtx = pd.read_csv (args.mtx, index_col = 0, sep = "\t")
    metadata = pd.read_csv (args.metadata, index_col = None, sep = "\t")
    if metadata.columns[0] == "Unnamed: 0":
        metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    with open (args.config) as f:
        config = json.load (f)

    indexCol = config.get ("metadata_index_column", "index"); clusterCol = config.get ("metadata_cluster_column", "cluster")
    numeratorCol = config["numeratorCol"]; denominatorCol = config["denominatorCol"]
    pseudoCount = config.get ("pseudo_count", 0); pseudoCount = 0 if not isinstance (pseudoCount, (float, int)) else pseudoCount
    mtx += pseudoCount
    if (mtx < 0).any (axis = None):
        raise ValueError ("Negative values in matrix!")

    with np.errstate (divide = "ignore", invalid = "ignore"):
        numerator = np.log2 (pd.DataFrame (mtx[metadata[numeratorCol]].values, index = mtx.index, columns = metadata[indexCol]))
        denominator = np.log2 (pd.DataFrame (mtx[metadata[denominatorCol]].values, index = mtx.index, columns = metadata[indexCol]))
    rawLog2FC = numerator - denominator
    if args.centralize:
        allClusters = sorted (set (metadata[clusterCol]))
        normFct = pd.DataFrame ({"value": rawLog2FC.mask (~np.isfinite (rawLog2FC)).mean (axis = 0, skipna = True), "cluster": metadata[clusterCol].values})
        normFct = normFct.groupby ("cluster").mean ().sort_index ()["value"].rename_axis (None)
        normFct = pd.DataFrame ([[normFct[C1] - normFct[C2] for C2 in allClusters] for C1 in allClusters], index = allClusters, columns = allClusters)
        logRatio = pd.Series (index = allClusters, dtype = float)
        for cluster in allClusters:
            LR = pd.concat ([rawLog2FC[metadata.loc[metadata[clusterCol] == C, indexCol]] + normFct.loc[cluster, C] for C in allClusters], axis = 1)
            LR = LR.mask (~np.isfinite (LR)).mean (axis = 0, skipna = True)
            logRatio[cluster] = LR.mean (skipna = True)
        base = logRatio.abs ().idxmin ()
        rawLog2FC = pd.concat ([rawLog2FC[metadata.loc[metadata[clusterCol] == C, indexCol]] + normFct.loc[base, C] for C in allClusters], axis = 1)[metadata[indexCol]]

    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)
    numerator.to_csv (os.path.join (args.output, "numerator_log.tsv"), sep = "\t")
    denominator.to_csv (os.path.join (args.output, "denominator_log.tsv"), sep = "\t")
    rawLog2FC.to_csv (os.path.join (args.output, "paired_log2FC.tsv"), sep = "\t")



if __name__ == "__main__":
    main ()


