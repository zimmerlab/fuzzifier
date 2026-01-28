import os
import json
import argparse
import numpy as np
import pandas as pd
from estimator_cutoff import estimatorByCutoff
from estimator_default import estimatorByDefault
from estimator_parameter import estimatorByParameter

### python main_estimator.py --mtx crispMatrix --config configFile --output outputPath

# Read command line arguments.
parser = argparse.ArgumentParser ()
parser.add_argument ("--mtx", type = str, required = True, help = "TSV file of raw value matrix to be processed")
parser.add_argument ("--config", type = str, required = True, help = "JSON config file for fuzzy concept argmuents")
parser.add_argument ("--output", type = str, required = True, help = "Output path for fuzzy concepts")
args = parser.parse_args ()

# Load crisp matrix.
mtx = pd.read_csv (args.mtx, index_col = 0, sep = "\t")

# Load config file.
with open (args.config) as f:
    config = json.load (f); f.close ()
const = {"-Infinity": -np.inf, "-infinity": -np.inf, "-Inf": -np.inf, "-inf": -np.inf,
         "+Infinity": np.inf, "+infinity": np.inf, "+Inf": np.inf, "+inf": np.inf,
         "Infinity": np.inf, "infinity": np.inf, "Inf": np.inf, "inf": np.inf,
         "NaN": np.nan, "NAN": np.nan, "nan": np.nan, "NA": np.nan, "na": np.nan}

# Get parameters from config file.
numFuzzySets = config["number_fuzzy_sets"]
fuzzyBy = config.get ("define_concept_per", "feature")
mtx = mtx.replace ([const.get (x, x) for x in config.get ("label_values", list ())], np.nan)
if isinstance (config.get ("left_noise_cutoff", "-Infinity"), (int, float)):
    mtx = mtx.mask ((~np.isnan (mtx)) & (mtx <= config["left_noise_cutoff"]))
if isinstance (config.get ("right_noise_cutoff", "+Infinity"), (int, float)):
    mtx = mtx.mask ((~np.isnan (mtx)) & (mtx >= config["right_noise_cutoff"]))

# Derive fuzzy concept(s).
match config.get ("define_concept_by", "default"):
    case "cutoff":
        functionType = config["function_type"]
        cutoffBy = config.get ("cutoff__method", "proportion")
        percents = config.get ("cutoff__percent_per_fuzzy_set", list ())
        slope = config.get ("cutoff__slope_per_cutoff", list ())
        tmp = estimatorByCutoff (mtx, numFuzzySets, functionType, fuzzyBy = fuzzyBy, cutoffBy = cutoffBy,
                                 percents = percents, slope = slope, labelValues = [np.nan])
    case "default":
        widthFct = config.get ("default__width_factor", 1)
        slopeFct = config.get ("default__slope_factor", 0.5)
        bwFct = config.get ("default__band_width_factor", 1)
        useOptimize = config.get ("default__use_scipy_optimization", False)
        tmp = estimatorByDefault (mtx, numFuzzySets, fuzzyBy = fuzzyBy, widthFactor = widthFct, slopeFactor = slopeFct, bwFactor = bwFct,
                                  useOptimize = useOptimize)
    case "parameter":
        functionType = config["function_type"]
        paramBy = config.get ("parameter__method", "percentile")
        functionParams = config["parameter__values"]
        tmp = estimatorByParameter (mtx, functionType, functionParams, fuzzyBy = fuzzyBy, paramBy = paramBy,
                                    labelValues = [np.nan])
    case _:
        raise ValueError

# Parse fuzzy concept(s) for output.
constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
fuzzyConcepts = {feature: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t] for t in tmp[feature]]
                 for feature in tmp.keys ()}

# Save generated fuzzy concepts.
if not os.path.isdir (os.path.dirname (args.output)):
    os.makedirs (os.path.dirname (args.output), exist_ok = True)
with open (args.output, "w", encoding = "utf-8") as f:
    json.dump (fuzzyConcepts, f, ensure_ascii = False, indent = 4, allow_nan = True); f.close ()


