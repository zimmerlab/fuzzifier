import os
import json
import argparse
import numpy as np
import pandas as pd
from fuzzifier import fuzzify

### python main_fuzzifier.py --mtx rawValueMatrix --concept fuzzyConcepts --metadata metadata --config config --perCluster --output outputDirectory


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--mtx", type = str, required = True, help = "Raw value matrix (TSV)")
    parser.add_argument ("--concept", type = str, required = True, help = "Fuzzy concepts (JSON)")
    parser.add_argument ("--metadata", type = str, required = False, help = "Metadata containing clustering column (TSV)")
    parser.add_argument ("--config", type = str, required = True, help = "Config file for fuzzification arguments (JSON)")
    parser.add_argument ("--perCluster", required = False, action = "store_true", help = "Whether to define fuzzy concept(s) per cluster")
    parser.add_argument ("--output", type = str, required = True, help = "Output directory for fuzzy values")
    args = parser.parse_args ()

    mtx = pd.read_csv (args.mtx, index_col = 0, sep = "\t")
    if args.perCluster:
        metadata = pd.read_csv (args.metadata, index_col = None, sep = "\t")
        if metadata.columns[0] == "Unnamed: 0":
            metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    with open (args.config) as f:
        config = json.load (f); f.close ()
    const = {"-Infinity": -np.inf, "-infinity": -np.inf, "-Inf": -np.inf, "-inf": -np.inf,
             "+Infinity": np.inf, "+infinity": np.inf, "+Inf": np.inf, "+inf": np.inf,
             "Infinity": np.inf, "infinity": np.inf, "Inf": np.inf, "inf": np.inf,
             "NaN": np.nan, "NAN": np.nan, "nan": np.nan, "NA": np.nan, "na": np.nan}
    labels = [const.get (x, x) for x in config.get ("label_values", list ())]
    cutoffLeft = config.get ("left_noise_cutoff", "-Infinity")
    cutoffRight = config.get ("right_noise_cutoff", "+Infinity")
    fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
    fuzzyBy = config.get ("define_concept_per", "feature")
    renameDict = config.get ("rename_fuzzy_sets", dict ())
    noiseRep = [np.floor (mtx.mask (~np.isfinite (mtx)).min (axis = None, skipna = True)) - 1,
                np.ceil (mtx.mask (~np.isfinite (mtx)).max (axis = None, skipna = True)) + 1]
    if isinstance (cutoffLeft, (int, float)):
        mtx = mtx.mask ((~np.isnan (mtx.replace (labels, np.nan))) & (mtx <= cutoffLeft), noiseRep[0])
        labels.append (noiseRep[0])
    if isinstance (cutoffRight, (int, float)):
        mtx = mtx.mask ((~np.isnan (mtx.replace (labels, np.nan))) & (mtx >= cutoffRight), noiseRep[1])
        labels.append (noiseRep[1])
    
    with open (args.concept) as f:
        fuzzyConcepts = json.load (f)
    if fuzzyBy == "matrix":
        for key in fuzzyConcepts.keys ():
            if len (fuzzyConcepts[key].keys ()) == 1:
                fuzzyBy = "sample"; x = list (fuzzyConcepts[key].keys ())[0]
                fuzzyConcepts[key] = {sample: fuzzyConcepts[key[x]] for sample in mtx.columns}
            elif len (fuzzyConcepts[key].keys ()) == mtx.shape[0]:
                if set (fuzzyConcepts[key].keys ()) == set (mtx.index):
                    fuzzyBy = "feature"
                else:
                    raise ValueError
            elif len (fuzzyConcepts[key].keys ()) == mtx.shape[1]:
                if set (fuzzyConcepts[key].keys ()) == set (mtx.columns):
                    fuzzyBy = "sample"
                else:
                    raise ValueError
            else:
                raise ValueError
    
    allFuzzyValues = list ()
    if args.perCluster:
        indexCol = config.get ("metadata_index_column", "index"); clusterCol = config.get ("metadata_cluster_column", "cluster")
        clustering = metadata.groupby (clusterCol)[indexCol].agg (list).to_dict ()
    else:
        clustering = {"ALL": mtx.columns}
    if fuzzyBy == "feature":
        for feature in mtx.index:
            memberships = pd.concat ([fuzzify (mtx.loc[feature, clustering[cluster]], fuzzyConcepts[cluster][feature], furtherParams = fuzzyParams)
                                      for cluster in fuzzyConcepts.keys ()], axis = 0).loc[mtx.columns]
            allFuzzyValues.append (memberships.round (3).to_numpy ())
        allFuzzyValues = np.array (allFuzzyValues)
    else:
        for sample in mtx.columns:
            memberships = pd.concat ([fuzzify (mtx[sample], fuzzyConcepts[cluster][sample], furtherParams = fuzzyParams)
                                      for cluster in fuzzyConcepts.keys ()], axis = 0).loc[mtx.index]
            allFuzzyValues.append (memberships.round (3).to_numpy ())
        allFuzzyValues = np.einsum ("ijk -> jik", np.array (allFuzzyValues))
    print (allFuzzyValues.shape)
    print ("sum minimum", allFuzzyValues.sum (axis = 2).min (axis = None), "\t",
           "sum maximum", allFuzzyValues.sum (axis = 2).max (axis = None))

    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)
    renameDict[f"FS0_{noiseRep[0]}"] = "MIN-NOISE"; renameDict[f"FS0_{noiseRep[1]}"] = "MAX-NOISE"
    for idx in range (allFuzzyValues.shape[2]):
        nameFS = memberships.columns[idx]; nameFS = renameDict.get (nameFS, nameFS)
        outputDF = pd.DataFrame (allFuzzyValues[:, :, idx], index = mtx.index, columns = mtx.columns)
        outputDF.to_csv (os.path.join (args.output, f"fuzzyValues_{nameFS}.tsv"), sep = "\t")



if __name__ == "__main__":
    main ()


