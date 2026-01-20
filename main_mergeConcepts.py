import os
import json
import argparse
import numpy as np
import pandas as pd

# python main_mergeConcepts.py --data rawMatrixDirectory --concepts conceptDirectory --metadata metadata --config config


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--data", type = str, required = True, help = "Directory for splitted matrices")
    parser.add_argument ("--concepts", type = str, required = True, help = "Directory for fitted fuzzy concepts of numerator and denominator")
    parser.add_argument ("--config", type = str, required = True, help = "Config file for fuzzy concept argmuents (JSON)")
    parser.add_argument ("--metadata", type = str, required = False, help = "Metadata containing clustering column (TSV)")
    args = parser.parse_args ()

    numerator = pd.read_csv (os.path.join (args.data, "numerator_log.tsv"), index_col = 0, sep = "\t")
    denominator = pd.read_csv (os.path.join (args.data, "denominator_log.tsv"), index_col = 0, sep = "\t")
    metadata = pd.read_csv ("./data/metadata.tsv", index_col = None, sep = "\t")
    if metadata.columns[0] == "Unnamed: 0":
        metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    with open (os.path.join (args.concepts, "concepts_log_numerator.json")) as f:
        backup = json.load (f)
    with open (os.path.join (args.concepts, "concepts_log_denominator.json")) as f:
        concepts = json.load (f)
    with open (args.config) as f:
        config = json.load (f)
    
    const = {"-Infinity": -np.inf, "-infinity": -np.inf, "-Inf": -np.inf, "-inf": -np.inf,
             "+Infinity": np.inf, "+infinity": np.inf, "+Inf": np.inf, "+inf": np.inf,
             "Infinity": np.inf, "infinity": np.inf, "Inf": np.inf, "inf": np.inf,
             "NaN": np.nan, "NAN": np.nan, "nan": np.nan, "NA": np.nan, "na": np.nan}
    indexCol = config.get ("metadata_index_column", "index")
    clusterCol = config.get ("metadata_cluster_column", "cluster")
    numFuzzySets = config["number_fuzzy_sets"]
    labels = [const.get (x, x) for x in config.get ("label_values", list ())]
    widthFct = config.get ("default__width_factor", 1)
    slopeFct = config.get ("default__slope_factor", 0.5)

    allClusters = sorted (set (metadata[clusterCol]))
    concepts_merged = dict ()
    for cluster in allClusters:
        concepts_merged[cluster] = dict ()
        sampleList = metadata.loc[metadata[clusterCol] == cluster, indexCol]
        for feature in denominator.index:
            values = pd.concat ([numerator.loc[feature, sampleList], denominator.loc[feature, sampleList]]).replace (labels, np.nan).dropna ()
            if values.empty:
                xRange = [-6, 6]
            else:
                xRange = [np.floor (values.min ()) - 1, np.ceil (values.max ()) + 1]
            if feature in concepts[cluster]:
                concept = concepts[cluster][feature].copy ()
            else:
                if feature in backup[cluster]:
                    concept = backup[cluster][feature].copy ()
                else:
                    mu = values[np.isfinite (values)].mean (skipna = True); sigma = values[np.isfinite (values)].std (skipna = True)
                    mu = 0 if np.isnan (mu) else mu; sigma = 1 if np.isnan (sigma) or sigma == 0 else sigma
                    coords = [mu + widthFct * (i + overlap) * sigma for i in np.linspace (-numFuzzySets, numFuzzySets, numFuzzySets + 1)
                              for overlap in [-slopeFct, slopeFct]]
                    concept = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFuzzySets + 1)], 3).tolist ()
                    concept[2] = [round (mu, 3), round (sigma, 3)]
            left = min (xRange[0], concept[0][2]); right = max (xRange[1], concept[-1][1])
            concept[0][0] = left; concept[0][1] = left; concept[-1][2] = right; concept[-1][3] = right
            concepts_merged[cluster][feature] = concept.copy ()
    
    with open (os.path.join (args.concepts, "concepts_log_feature-wise.json"), "w", encoding = "utf-8") as f:
        json.dump (concepts_merged, f, ensure_ascii = False, indent = 4, allow_nan = True)



if __name__ == "__main__":
    main ()


