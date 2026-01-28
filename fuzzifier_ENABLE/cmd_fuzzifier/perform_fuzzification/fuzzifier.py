import os
import json
import argparse
import numpy as np
import pandas as pd
from fuzzification import fuzzify

### python main_fuzzifier.py --mtx crispMatrix --concept fuzzyConceptFile --config configFile --output outputDirectory

# Read command line arguments.
parser = argparse.ArgumentParser ()
parser.add_argument ("--mtx", type = str, required = True, help = "TSV file of raw value matrix to be fuzzified")
parser.add_argument ("--concept", type = str, required = True, help = "JSON file of fuzzy concepts")
parser.add_argument ("--config", type = str, required = False, help = "JSON config file for fuzzification parameters")
parser.add_argument ("--output", type = str, required = True, help = "Output directory for fuzzy values.")
args = parser.parse_args ()

# Load crisp matrix.
mtx = pd.read_csv (args.mtx, index_col = 0, sep = "\t")

# Load config file.
if args.config is None:
    labels = list ()
    fuzzyParams = {"addIndicator": False, "indicateValue": list ()}
else:
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
    conceptBy = config.get ("define_concept_per", "feature")
    fuzzyBy = config.get ("fuzzy_by", "feature")
    downloadBy = config.get ("save_fuzzy_values_per", "feature")
    renameDict = config.get ("rename", dict ())
noiseRep = [np.floor (mtx.mask (~np.isfinite (mtx)).min (axis = None, skipna = True)) - 1,
            np.ceil (mtx.mask (~np.isfinite (mtx)).max (axis = None, skipna = True)) + 1]
if isinstance (cutoffLeft, (int, float)):
    mtx = mtx.mask ((~np.isnan (mtx.replace (labels, np.nan))) & (mtx <= cutoffLeft), noiseRep[0])
    labels.append (noiseRep[0]); renameDict[f"FS0_{noiseRep[0]}"] = "MIN-NOISE"
if isinstance (cutoffRight, (int, float)):
    mtx = mtx.mask ((~np.isnan (mtx.replace (labels, np.nan))) & (mtx >= cutoffRight), noiseRep[1])
    labels.append (noiseRep[1]); renameDict[f"FS0_{noiseRep[1]}"] = "MAX-NOISE"

# Load fuzzy concepts.
with open (args.concept) as f:
    fuzzyConcepts = json.load (f); f.close ()
if conceptBy != "feature" and conceptBy != "sample":
    concept = fuzzyConcepts[list (fuzzyConcepts.keys ())[0]]
    if fuzzyBy == "feature":
        fuzzyConcepts = {feature: concept for feature in mtx.index}
    else:
        fuzzyConcepts = {sample: concept for sample in mtx.columns}

# Fuzzify crisp values per feature in the input matrix.
allFuzzyValues = list ()
if fuzzyBy == "feature":
    for feature in mtx.index:
        memberships = fuzzify (mtx.loc[feature], fuzzyConcepts[feature], furtherParams = fuzzyParams).round (3)
        allFuzzyValues.append (memberships.to_numpy ())
    allFuzzyValues = np.array (allFuzzyValues); allSets = list (memberships.columns)
else:
    for sample in mtx.columns:
        memberships = fuzzify (mtx[sample], fuzzyConcepts[sample], furtherParams = fuzzyParams).round (3)
        allFuzzyValues.append (memberships.to_numpy ())
    allFuzzyValues = np.einsum ("ijk -> jik", np.array (allFuzzyValues)); allSets = list (memberships.columns)
print (allFuzzyValues.shape)
print ("sum minimum", allFuzzyValues.sum (axis = 2).min (axis = None), "\t",
       "sum maximum", allFuzzyValues.sum (axis = 2).max (axis = None))

# Generate dataframes for fuzzy value output.
if not os.path.exists (args.output):
    os.makedirs (args.output)
if downloadBy == "feature":
    for idx in range (mtx.shape[0]):
        outputDF = pd.DataFrame (allFuzzyValues[idx, :, :], index = mtx.columns, columns = allSets)
        outputDF = outputDF.rename (columns = renameDict)
        outputDF.to_csv (os.path.join (args.output, f"fuzzyValues_{mtx.index[idx]}.tsv"), sep = "\t")
elif downloadBy == "sample":
    for idx in range (mtx.shape[1]):
        outputDF = pd.DataFrame (allFuzzyValues[:, idx, :], index = mtx.index, columns = allSets)
        outputDF = outputDF.rename (columns = renameDict)
        outputDF.to_csv (os.path.join (args.output, f"fuzzyValues_{mtx.columns[idx]}.tsv"), sep = "\t")
elif downloadBy == "fuzzy set":
    for idx in range (allFuzzyValues.shape[2]):
        nameFS = memberships.columns[idx]; nameFS = renameDict.get (nameFS, nameFS)
        outputDF = pd.DataFrame (allFuzzyValues[:, :, idx], index = mtx.index, columns = mtx.columns)
        outputDF.to_csv (os.path.join (args.output, f"fuzzyValues_{nameFS}.tsv"), sep = "\t")
else:
    raise ValueError


