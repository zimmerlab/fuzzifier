import os
import json
import argparse
import numpy as np
import pandas as pd
from estimator import estimatorByCutoff, estimatorByParameter, estimatorByDefault

# python main_concepts.py --mtx rawValueMatrix --metadata metadata --config config --perCluster --output outputDirectory


def getConcepts (mtx, numFuzzySets, fuzzyBy, mode, config):
    if mode == "cutoff":
        functionType = config["function_type"]
        cutoffBy = config.get ("cutoff__method", "proportion")
        percents = config.get ("cutoff__percent_per_fuzzy_set", list ())
        slope = config.get ("cutoff__slope_per_cutoff", list ())
        concepts = estimatorByCutoff (mtx, numFuzzySets, functionType, fuzzyBy = fuzzyBy, cutoffBy = cutoffBy,
                                      percents = percents, slope = slope, labelValues = [np.nan])
    elif mode == "parameter":
        functionType = config["function_type"]
        paramBy = config.get ("parameter__method", "percentile")
        functionParams = config["parameter__values"]
        concepts = estimatorByParameter (mtx, functionType, functionParams, fuzzyBy = fuzzyBy, paramBy = paramBy, labelValues = [np.nan])
    elif mode == "default":
        bwFct = config.get ("default__band_width_factor", 1)
        widthFct = config.get ("default__width_factor", 1)
        slopeFct = config.get ("default__slope_factor", 0.5)
        if fuzzyBy == "feature":
            concepts = estimatorByDefault (mtx, numFuzzySets, widthFactor = widthFct, slopeFactor = slopeFct, bwFactor = bwFct)
        elif fuzzyBy == "sample":
            concepts = estimatorByDefault (mtx.T, numFuzzySets, widthFactor = widthFct, slopeFactor = slopeFct, bwFactor = bwFct)
        elif fuzzyBy == "matrix":
            values = mtx.melt ()["value"].dropna (); mu = values.mean ()
            sigma = np.sqrt (values[values < mu].std () ** 2 + values[values > mu].std () ** 2)
            xRange = [np.floor (values.min ()) - 1, np.ceil (values.max ()) + 1]; centerIdx = int (numFuzzySets / 2)
            coords = [mu + widthFct * (i + overlap) * sigma for i in np.linspace (-numFuzzySets, numFuzzySets, numFuzzySets + 1)
                      for overlap in [-slopeFct, slopeFct]]
            concept = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFuzzySets + 1)], 3).tolist ()
            left = min (xRange[0], concept[0][3]); right = max (xRange[1], concept[-1][1])
            concept[0][0] = left; concept[0][1] = left; concept[-1][2] = right; concept[-1][3] = right
            concept[centerIdx] = [round (mu, 3), round (sigma, 3)]
            concepts = {sample: concept for sample in mtx.columns}
    else:
        raise ValueError
    return concepts



def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--mtx", type = str, required = True, help = "Raw value matrix (TSV)")
    parser.add_argument ("--config", type = str, required = True, help = "Config file for fuzzy concept argmuents (JSON)")
    parser.add_argument ("--metadata", type = str, required = False, help = "Metadata containing clustering column (TSV)")
    parser.add_argument ("--perCluster", required = False, action = "store_true", help = "Whether to define fuzzy concept(s) per cluster")
    parser.add_argument ("--output", type = str, required = True, help = "Output file name for fuzzy concepts")
    args = parser.parse_args ()
    
    mtx = pd.read_csv (args.mtx, index_col = 0, sep = "\t")
    mtx = mtx.rename (columns = {col: col[:-2] if col.endswith (".1") else col for col in mtx.columns})
    if args.perCluster:
        metadata = pd.read_csv (args.metadata, index_col = None, sep = "\t")
        if metadata.columns[0] == "Unnamed: 0":
            metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    with open (args.config) as f:
        config = json.load (f)

    const = {"-Infinity": -np.inf, "-infinity": -np.inf, "-Inf": -np.inf, "-inf": -np.inf,
             "+Infinity": np.inf, "+infinity": np.inf, "+Inf": np.inf, "+inf": np.inf,
             "Infinity": np.inf, "infinity": np.inf, "Inf": np.inf, "inf": np.inf,
             "NaN": np.nan, "NAN": np.nan, "nan": np.nan, "NA": np.nan, "na": np.nan}
    numFuzzySets = config["number_fuzzy_sets"]
    fuzzyBy = config.get ("define_concept_per", "feature")
    labels = [const.get (x, x) for x in config.get ("label_values", list ())]
    mtx = mtx.replace (labels, np.nan)
    if isinstance (const.get ("left_noise_cutoff", "-Infinity"), (int, float)):
        mtx = mtx.mask ((~np.isnan (mtx)) & (mtx <= const["left_noise_cutoff"]))
    if isinstance (const.get ("right_noise_cutoff", "+Infinity"), (int, float)):
        mtx = mtx.mask ((~np.isnan (mtx)) & (mtx >= const["right_noise_cutoff"]))

    mode = config.get ("define_concept_by", "default"); constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
    if args.perCluster and fuzzyBy != "sample":
        indexCol = config.get ("metadata_index_column", "index"); clusterCol = config.get ("metadata_cluster_column", "cluster")
        fuzzyConcepts = dict ()
        for cluster in sorted (set (metadata[clusterCol])):
            tmp = getConcepts (mtx[metadata.loc[metadata[clusterCol] == cluster, indexCol]], numFuzzySets, fuzzyBy, mode, config)
            fuzzyConcepts[cluster] = {key: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t] for t in tmp[key]]
                                      for key in tmp.keys ()}
    else:
        tmp = getConcepts (mtx, numFuzzySets, fuzzyBy, mode, config)
        fuzzyConcepts = {"ALL": {key: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t] for t in tmp[key]]
                                 for key in tmp.keys ()}}

    if not os.path.exists (os.path.dirname (args.output)):
        os.makedirs (os.path.dirname (args.output))
    with open (args.output, "w", encoding = "utf-8") as f:
        json.dump (fuzzyConcepts, f, ensure_ascii = False, indent = 4, allow_nan = True)



if __name__ == "__main__":
    main ()


