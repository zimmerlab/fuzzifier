import warnings
warnings.simplefilter (action = "ignore", category = FutureWarning)

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from helper_functions import getConcept, parseConcept, fuzzify, getReport, getClusterMap

# python main_fuzzifier.py --mtx rawValueMatrix --concept fuzzyConcepts --config config --output outputDirectory


parser = argparse.ArgumentParser ()
parser.add_argument ("--mtx", type = str, required = True, help = "Raw value matrix (TSV or H5AD)")
parser.add_argument ("--concept", type = str, required = True, help = "Constraints or detailed fuzzy concepts (JSON)")
parser.add_argument ("--config", type = str, required = True, help = "Config file for detailed parameters (JSON)")
parser.add_argument ("--output", type = str, required = True, help = "Ouptut directory for fuzzy values")
args = parser.parse_args ()

with open (args.concept) as f:
    concepts = json.load (f)

with open (args.config) as f:
    config = json.load (f)
const = {"-infinity": -np.inf, "-inf": -np.inf,
         "+infinity": np.inf, "+inf": np.inf, "infinity": np.inf, "inf": np.inf,
         "nan": np.nan, "na": np.nan, "zero": 0}
defaultName = config.get ("key_default_concept", "DEFAULT"); direction = config.get ("fuzzify_per", "feature")
noMinNoise = config.get ("ignore_MIN-NOISE", False); noMaxNoise = config.get ("ignore_MAX-NOISE", False)
scaleSum = config.get ("force_sum_one", True); renameLabels = config.get ("rename_labels", dict ())
renameLabels = {const.get (val.lower ()): renameLabels[val] for val in renameLabels.keys () if isinstance (val, str)}
if (noMinNoise and (not noMaxNoise)) or ((not noMinNoise) and noMaxNoise):
    renameLabels["MIN-NOISE"] = "NOISE"; renameLabels["MAX-NOISE"] = "NOISE"
generatePlots = config.get ("generate_report_plots", False)

with open (args.concept) as f:
    concepts = json.load (f)

if not os.path.exists (args.output):
    os.makedirs (args.output, exist_ok = True)
if not os.path.exists (os.path.join (args.output, "fuzzy_values")):
    os.makedirs (os.path.join (args.output, "fuzzy_values"))
if generatePlots and not os.path.exists (os.path.join (args.output, "reports")):
    os.makedirs (os.path.join (args.output, "reports"))
if not os.path.exists (os.path.join (args.output, "evaluations")):
    os.makedirs (os.path.join (args.output, "evaluations"))

deriveConcepts = (not isinstance (list (concepts.values ())[0], dict))
if args.mtx.lower ().endswith ("tsv"):
    with open (args.mtx) as f:
        samples = f.readline ().lstrip ().rstrip ("\n").split ("\t")
        features = [line.split ("\t")[0] for line in f.readlines ()]
elif args.mtx.lower ().endswith ("h5ad"):
    if direction == "sample":
        adata = sc.read_h5ad (args.mtx); features = list (adata.var_names); samples = list (adata.obs_names)
    else:
        adata = sc.read_h5ad (args.mtx).T; features = list (adata.obs_name); samples = list (adata.var_names)
else:
    raise TypeError

if deriveConcepts:
    minLevelCons = config.get ("left_noise_cutoff_constant", -np.inf); maxLevelCons = config.get ("right_noise_cutoff_constant", np.inf)
    minLevelCons = const.get (minLevelCons.lower (), -np.inf) if isinstance (minLevelCons, str) else minLevelCons
    maxLevelCons = const.get (maxLevelCons.lower (), np.inf) if isinstance (maxLevelCons, str) else maxLevelCons
    minLevelPct = config.get ("left_noise_cutoff_percent", 0); maxLevelPct = config.get ("right_noise_cutoff_percent", 1)
    minLevelPct = const.get (minLevelPct.lower (), 0) if isinstance (minLevelPct, str) else minLevelPct
    maxLevelPct = const.get (maxLevelPct.lower (), 1) if isinstance (maxLevelPct, str) else maxLevelPct
    param_keys = ["value_type", "number_fuzzy_sets", "label_values",
                  "fit_Gaussian_curve", "use_scipy_optimization", "band_width_factor"]
    consType = concepts.get ("value_type", "fixed"); consValue = list (); bwFct = concepts.get ("band_width_factor", 1)
    outputLabels = concepts.get ("label_values", list ())
    labels = [const.get (x.lower ()) if isinstance (x, str) else x for x in outputLabels]
    renameFS = [key for key in concepts.keys () if key not in param_keys]
    numFS = concepts.get ("number_fuzzy_sets", len (renameFS)); concept_cons = [concepts[FS][0] for FS in renameFS]
    useFit = concepts.get ("fit_Gaussian_curve", False); useOptimize = concepts.get ("use_scipy_optimization", False)
    if consType == "proportion":
        consValue = set ()
        for FS in renameFS:
            params, typeFS, _ = concepts[FS]
            if typeFS == "trapezoidal":
                consValue |= set (params)
            else:
                consValue |= {params[0]}
    basicInfo = {"number_fuzzy_sets": numFS, "label_values": labels}
    if args.mtx.lower ().endswith ("tsv"):
        with open (args.mtx) as f:
            values = [[np.nan if x == "" else float (x) for x in line.strip ("\n").split ("\t")[1:]] for line in f.readlines ()[1:]]
        values = pd.Series (sum (values, list ())).round (5)
    if args.mtx.lower ().endswith ("h5ad"):
        values = pd.Series (np.array (adata[adata.obs_names].X.data).reshape ((1, -1))[0]).round (5)
    default = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                          minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                          refConcept = concept_cons, consValue = consValue,
                          useFit = False, useOptimize = False, bwFct = bwFct)
    defaultOutput = default.copy (); defaultOutput["label_values"] = outputLabels; allConcepts = {defaultName: defaultOutput}
    del values
else:
    default = parseConcept (concepts.get (defaultName, dict ()))

summary = dict (); expectation = dict (); observation = dict ()
if direction == "sample":
    maxSplit = 2
    for sample in samples:
        if args.mtx.lower ().endswith ("tsv"):
            if sample == samples[-1]:
                with open (args.mtx) as f:
                    values = pd.Series ([line.strip ("\n").split ("\t")[-1] for line in f.readlines ()[1:]], index = features)
            else:
                with open (args.mtx) as f:
                    values = pd.Series ([line.strip ("\n").split ("\t", maxsplit = maxSplit)[-2] for line in f.readlines ()[1:]],
                                        index = features)
            values[values == ""] = np.nan; values = values.astype (float).round (5); maxSplit += 1
        else:
            values = adata[sample].to_df ().loc[sample].round (5)
        if deriveConcepts:
            concept = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                                  minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                  refConcept = concept_cons, consValue = consValue,
                                  useFit = useFit, useOptimize = useOptimize, bwFct = bwFct)
            if concept["number_fuzzy_sets"] == 0:
                concept = default.copy (); isFitted = False
            else:
                outputConcept = concept.copy (); outputConcept["label_values"] = outputLabels
                allConcepts[sample] = outputConcept; isFitted = True
        else:
            concept = concepts.get (sample, {"number_fuzzy_sets": 0}).copy ()
            if concept["number_fuzzy_sets"] == 0:
                concept = default.copy (); isFitted = False
            else:
                concept = parseConcept (concept); isFitted = True
        memberships, exp, obs, deviation = fuzzify (values, concept, renameLabels = renameLabels,
                                                    ignoreMinNoise = noMinNoise, ignoreMaxNoise = noMaxNoise,
                                                    scaleSum = scaleSum)
        expectation[sample] = exp.round (5); observation[sample] = obs.round (5)
        summary[sample] = {"deviation": round (deviation, 5), "individual_concept": isFitted}
        if not memberships.empty:
            memberships.round (3).to_csv (os.path.join (args.output, "fuzzy_values", f"fuzzyValues_{sample}.tsv"), sep = "\t")
            if generatePlots:
                getReport (values, concept, exp, obs, title = sample, ignoreMinNoise = noMinNoise, ignoreMaxNoise = noMaxNoise,
                           outputPath = os.path.join (args.output, "reports", f"report_{sample}.png"))
else:
    if args.mtx.lower ().endswith ("tsv"):
        with open (args.mtx) as f:
            _ = f.readline ()
            for feature in features:
                values = pd.Series ([np.nan if x == "" else float (x) for x in f.readline ().strip ("\n").split ("\t")[1:]],
                                    index = samples).round (5)
                if direction == "feature":
                    if deriveConcepts:
                        concept = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                                              minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                              refConcept = concept_cons, consValue = consValue,
                                              useFit = useFit, useOptimize = useOptimize, bwFct = bwFct)
                        if concept["number_fuzzy_sets"] == 0:
                            concept = default.copy (); isFitted = False
                        else:
                            outputConcept = concept.copy (); outputConcept["label_values"] = outputLabels
                            allConcepts[feature] = outputConcept; isFitted = True
                    else:
                        concept = concepts.get (feature, {"number_fuzzy_sets": 0}).copy ()
                        if concept["number_fuzzy_sets"] == 0:
                            concept = default.copy (); isFitted = False
                        else:
                            concept = parseConcept (concept); isFitted = True
                else:
                    concept = default.copy (); isFitted = False
                memberships, exp, obs, deviation = fuzzify (values, concept, renameLabels = renameLabels,
                                                            ignoreMinNoise = noMinNoise, ignoreMaxNoise = noMaxNoise,
                                                            scaleSum = scaleSum)
                expectation[feature] = exp.round (5); observation[feature] = obs.round (5)
                summary[feature] = {"deviation": round (deviation, 5), "individual_concept": isFitted}
                if not memberships.empty:
                    memberships.round (3).to_csv (os.path.join (args.output, "fuzzy_values", f"fuzzyValues_{feature}.tsv"), sep = "\t")
                    if generatePlots:
                        getReport (values, concept, exp, obs, title = feature, ignoreMinNoise = noMinNoise, ignoreMaxNoise = noMaxNoise,
                                   outputPath = os.path.join (args.output, "reports", f"report_{feature}.png"))
    if args.mtx.lower ().endswith ("h5ad"):
        for feature in features:
            values = adata[feature].to_df ().loc[feature].round (5)
            if direction == "feature":
                if deriveConcepts:
                    concept = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                                          minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                          refConcept = concept_cons, consValue = consValue,
                                          useFit = useFit, useOptimize = useOptimize, bwFct = bwFct)
                    if concept["number_fuzzy_sets"] == 0:
                        concept = default.copy (); isFitted = False
                    else:
                        outputConcept = concept.copy (); outputConcept["label_values"] = outputLabels
                        allConcepts[feature] = outputConcept; isFitted = True
                else:
                    concept = concepts.get (feature, {"number_fuzzy_sets": 0}).copy ()
                    if concept["number_fuzzy_sets"] == 0:
                        concept = default.coyp (); isFitted = False
                    else:
                        concept = parseConcept (concept); isFitted = True
            else:
                concept = default.copy (); isFitted = False
            memberships, exp, obs, deviation = fuzzify (values, concept, renameLabels = renameLabels,
                                                        ignoreMinNoise = noMinNoise, ignoreMaxNoise = noMaxNoise,
                                                        scaleSum = scaleSum)
            expectation[feature] = exp.round (5); observation[feature] = obs.round (5)
            summary[feature] = {"deviation": round (deviation, 5), "individual_concept": isFitted}
            if not memberships.empty:
                memberships.round (3).to_csv (os.path.join (args.output, "fuzzy_values", f"fuzzyValues_{feature}.tsv"), sep = "\t")
                if generatePlots:
                    getReport (values, concept, exp, obs, title = feature, ignoreMinNoise = noMinNoise, ignoreMaxNoise = noMaxNoise,
                               outputPath = os.path.join (args.output, "reports", f"report_{feature}.png"))

expectation = pd.DataFrame (expectation).T; expectation.to_csv (os.path.join (args.output, "evaluations", "expected_percentage.tsv"), sep = "\t")
observation = pd.DataFrame (observation).T; observation.to_csv (os.path.join (args.output, "evaluations", "observed_percentage.tsv"), sep = "\t")
getClusterMap (expectation, "Blues", "sample" if direction == "sample" else "feature", center = None, title = "expected percentage",
               outputPath = os.path.join (args.output, "evaluations", "expected_percentage.png"))
getClusterMap (observation, "Blues", "sample" if direction == "sample" else "feature", center = None, title = "observed percentage",
               outputPath = os.path.join (args.output, "evaluations", "observed_percentage.png"))
getClusterMap (observation - expectation, "vlag", "sample" if direction == "sample" else "feature", center = 0, title = "observation - expectation",
               outputPath = os.path.join (args.output, "evaluations", "deviation.png"))

summary = pd.DataFrame.from_dict (summary, orient = "index"); summary.to_csv (os.path.join (args.output, "evaluations", "summary.tsv"), sep = "\t")
fig, ax = plt.subplots (figsize = (6, 4))
if summary["individual_concept"].any ():
    ax.hist (summary.loc[summary["individual_concept"], "deviation"], bins = 25, color = "firebrick", alpha = 0.6, label = "individual fuzzy concept")
if not summary["individual_concept"].all ():
    ax.hist (summary.loc[~summary["individual_concept"], "deviation"], bins = 25, color = "steelblue", alpha = 0.6, label = "default fuzzy concept")
ax.set_xlabel ("observation - expectation", size = 10); ax.set_ylabel ("number of samples", size = 10); ax.legend (facecolor = "white")
fig.tight_layout (); plt.savefig (os.path.join (args.output, "evaluations", "distribution_deviation.png")); plt.close ()

if deriveConcepts:
    with open (os.path.join (args.output, "concepts_detailed.json"), "w", encoding = "utf-8") as f:
        json.dump (allConcepts, f, ensure_ascii = False, indent = 4, allow_nan = True)


