import warnings
warnings.simplefilter (action = "ignore", category = FutureWarning)

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from helper_functions import getConcept, getPercentage, getSubarea

# python main_concepts.py --mtx rawValueMatrix --config config --output outputDirectory


parser = argparse.ArgumentParser ()
parser.add_argument ("--mtx", type = str, required = True, help = "Raw value matrix (TSV or H5AD)")
parser.add_argument ("--config", type = str, required = True, help = "Config file for detailed parameters (JSON)")
parser.add_argument ("--output", type = str, required = True, help = "Output directory for constraints and detailed fuzzy concepts")
args = parser.parse_args ()

with open (args.config) as f:
    config = json.load (f); f.close ()
const = {"-infinity": -np.inf, "-inf": -np.inf,
         "+infinity": np.inf, "+inf": np.inf, "infinity": np.inf, "inf": np.inf,
         "nan": np.nan, "na": np.nan}
constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
labels = [const.get (x.lower ()) if isinstance (x, str) else x for x in config.get ("label_values", list ())]
outputLabels = [constRev.get (x, x) if not np.isnan (x) else "NA" for x in labels]
minLevelCons = config.get ("left_noise_cutoff_constant", -np.inf); maxLevelCons = config.get ("right_noise_cutoff_constant", np.inf)
minLevelCons = const.get (minLevelCons.lower (), -np.inf) if isinstance (minLevelCons, str) else minLevelCons
maxLevelCons = const.get (maxLevelCons.lower (), np.inf) if isinstance (maxLevelCons, str) else maxLevelCons
minLevelPct = config.get ("left_noise_cutoff_percent", 0); maxLevelPct = config.get ("right_noise_cutoff_percent", 1)
minLevelPct = const.get (minLevelPct.lower (), 0) if isinstance (minLevelPct, str) else minLevelPct
maxLevelPct = const.get (maxLevelPct.lower (), 1) if isinstance (maxLevelPct, str) else maxLevelPct
defaultName = config.get ("key_default_concept", "DEFAULT"); direction = config.get ("define_concept_per", "feature")
method = config["define_concept_by"]; params = config.get (f"parameters_{method}", dict ())
renameFS = config.get ("fuzzy_variables", list ())

typeFS_dict = {2: "Gaussian", 4: "trapezoidal"}
defaultColors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                 "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
                 "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
                 "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"]
if method == "constraint":
    consType = params.get ("constraint_type", "fixed"); concept_cons = params["constraints"]; numFS = len (concept_cons)
    if consType == "fixed":
        consValue = list (); useFit = False; useOptimize = False; percentage = [0] * numFS
    elif consType == "proportion" or consType == "z-score":
        consValue = set ()
        for idx in range (numFS):
            if len (concept_cons[idx]) == 4:
                consValue |= concept_cons[idx]
            elif len (concept_cons[idx]) == 2:
                consValue |= {concept_cons[idx][0]}
            else:
                raise ValueError
        consValue = sorted (consValue); useFit = (consType == "z-score"); useOptimize = params.get ("use_scipy_optimization", False)
        percentage = getPercentage (np.linspace (0, 1, 1001), concept_cons, minLevel = -np.inf, maxLevel = np.inf)
    else:
        raise ValueError
    bwFct = 1; widthFct = 1; slopeFct = 0.5; centerIdx = 0
    if consType == "z-score":
        percentage = getSubarea (0, 1, concept_cons, minLevel = -np.inf, maxLevel = np.inf)
elif method == "default":
    consType = "z-score"; centerIdx = params["number_fuzzy_sets_per_side"]; numFS = 2 * centerIdx + 1; useFit = True
    bwFct = params.get ("band_width_factor", 1); widthFct = params.get ("width_scale_factor", 1); slopeFct = params.get ("slope_percentage", 0.5)
    useOptimize = params.get ("use_scipy_optimization", False)
    coords = [widthFct * (i + overlap) for i in np.linspace (-numFS, numFS, numFS + 1) for overlap in [-slopeFct, slopeFct]]
    concept_cons = np.round ([coords[(2 * k - 2):(2 * k + 2)] for k in range (1, numFS + 1)], 3).tolist ()
    concept_cons[centerIdx] = [0, widthFct]; concept_cons[0][0] = concept_cons[0][1]; concept_cons[-1][3] = concept_cons[-1][2]; consValue = list ()
    percentage = getSubarea (0, 1, concept_cons, minLevel = -np.inf, maxLevel = np.inf)
else:
    raise ValueError
if len (renameFS) == 0:
    renameFS = [f"FS{i}" for i in range (1, numFS + 1)]

basicInfo = {"number_fuzzy_sets": numFS, "label_values": outputLabels}
constraint = {"value_type": consType, "number_fuzzy_sets": numFS, "label_values": outputLabels,
              "fit_Gaussian_curve": useFit, "use_scipy_optimization": useOptimize, "band_width_factor": bwFct}
for idx in range (numFS):
    constraint[renameFS[idx]] = [concept_cons[idx], typeFS_dict[len (concept_cons[idx])], defaultColors[idx], round (percentage[idx], 5)]

if args.mtx.lower ().endswith ("tsv"):
    with open (args.mtx) as f:
        samples = f.readline ().strip ("\n").split ("\t")[1:]
        features = [line.strip ("\n").split ("\t")[0] for line in f.readlines ()]
    with open (args.mtx) as f:
        values = [[np.nan if x == "" else float (x) for x in line.strip ("\n").split ("\t")[1:]] for line in f.readlines ()[1:]]
    values = pd.Series (sum (values, list ())).round (5)
elif args.mtx.lower ().endswith ("h5ad"):
    if direction == "feature":
        adata = sc.read_h5ad (args.mtx).T; features = list (adata.obs_names); samples = list (adata.var_names)
    else:
        adata = sc.read_h5ad (args.mtx); features = list (adata.var_names); samples = list (adata.obs_names)
    values = pd.Series (np.array (adata[adata.obs_names].X.data).reshape ((1, -1))[0]).round (5)
else:
    raise TypeError
default = getConcept (values, method, consType, basicInfo, numFS, renameFS, labels,
                      minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                      useFit = False, useOptimize = False, bwFct = bwFct,
                      refConcept = concept_cons, consValue = consValue,
                      widthFct = widthFct, slopeFct = slopeFct, centerIdx = centerIdx)
del values

detailedConcept = {defaultName: default}
if direction == "feature":
    if args.mtx.lower ().endswith ("tsv"):
        with open (args.mtx) as f:
            _ = f.readline ()
            for feature in features:
                values = pd.Series ([np.nan if x == "" else float (x) for x in f.readline ().strip ("\n").split ("\t")[1:]],
                                    index = samples).round (5)
                detailedConcept[feature] = getConcept (values, method, consType, basicInfo, numFS, renameFS, labels,
                                                       minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                                       useFit = useFit, useOptimize = useOptimize, bwFct = bwFct,
                                                       refConcept = concept_cons, consValue = consValue,
                                                       widthFct = widthFct, slopeFct = slopeFct, centerIdx = centerIdx)
    if args.mtx.lower ().endswith ("h5ad"):
        for feature in features:
            values = pd.Series (np.array (adata[feature].X.data)[0]).round (5)
            detailedConcept[feature] = getConcept (values, method, consType, basicInfo, numFS, renameFS, labels,
                                                   minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                                   useFit = useFit, useOptimize = useOptimize, bwFct = bwFct,
                                                   refConcept = concept_cons, consValue = consValue,
                                                   widthFct = widthFct, slopeFct = slopeFct, centerIdx = centerIdx)
elif direction == "sample":
    if args.mtx.lower ().endswith ("tsv"):
        maxSplit = 2
        for sample in samples:
            if sample == samples[-1]:
                with open (args.mtx) as f:
                    values = pd.Series ([line.strip ("\n").split ("\t")[-1] for line in f.readlines ()[1:]], index = features)
            else:
                with open (args.mtx) as f:
                    values = pd.Series ([line.strip ("\n").split ("\t", maxsplit = maxSplit)[-2] for line in f.readlines ()[1:]],
                                        index = features)
            values[values == ""] = np.nan; values = values.astype (float).round (5); maxSplit += 1
            detailedConcept[sample] = getConcept (values, method, consType, basicInfo, numFS, renameFS, labels,
                                                  minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                                  useFit = useFit, useOptimize = useOptimize, bwFct = bwFct,
                                                  refConcept = concept_cons, consValue = consValue,
                                                  widthFct = widthFct, slopeFct = slopeFct, centerIdx = centerIdx)
    if args.mtx.lower ().endswith ("h5ad"):
        maxSplit = 2
        for sample in samples:
            values = pd.Series (np.array (adata[sample].X.data)[0]).round (5)
            detailedConcept[sample] = getConcept (values, method, consType, basicInfo, numFS, renameFS, labels,
                                                  minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                                  useFit = useFit, useOptimize = useOptimize, bwFct = bwFct,
                                                  refConcept = concept_cons, consValue = consValue,
                                                  widthFct = widthFct, slopeFct = slopeFct, centerIdx = centerIdx)
    
    
if not os.path.exists (args.output):
    os.makedirs (args.output, exist_ok = True)
with open (os.path.join (args.output, "concepts_constraints.json"), "w", encoding = "utf-8") as f:
    json.dump (constraint, f, ensure_ascii = False, indent = 4, allow_nan = True)
with open (os.path.join (args.output, "concepts_detailed.json"), "w", encoding = "utf-8") as f:
    json.dump (detailedConcept, f, ensure_ascii = False, indent = 4, allow_nan = True)


