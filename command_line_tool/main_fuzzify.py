import os
import json
import argparse
import numpy as np
import pandas as pd
from helper_functions import fuzzify, getConcept

# python main_fuzzifier.py --mtx rawValueMatrix --concept fuzzyConcept(s) --config config --output outputDirectory


parser = argparse.ArgumentParser ()
parser.add_argument ("--mtx", type = str, required = True, help = "Raw value matrix (TSV)")
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
minLevelCons = config.get ("left_noise_cutoff_constant", -np.inf); maxLevelCons = config.get ("right_noise_cutoff_constant", np.inf)
minLevelCons = const.get (minLevelCons.lower (), -np.inf) if isinstance (minLevelCons, str) else minLevelCons
maxLevelCons = const.get (maxLevelCons.lower (), np.inf) if isinstance (maxLevelCons, str) else maxLevelCons
minLevelPct = config.get ("left_noise_cutoff_percent", 0); maxLevelPct = config.get ("right_noise_cutoff_percent", 1)
minLevelPct = const.get (minLevelPct.lower (), 0) if isinstance (minLevelPct, str) else minLevelPct
maxLevelPct = const.get (maxLevelPct.lower (), 1) if isinstance (maxLevelPct, str) else maxLevelPct
defaultName = config.get ("key_default_concept", "DEFAULT"); direction = config.get ("fuzzify_per", "feature")
renameLabels = config.get ("rename_labels", dict ())
renameLabels = {const.get (val.lower ()): renameLabels[val] for val in renameLabels.keys () if isinstance (val, str)}

with open (args.mtx) as f:
    samples = f.readline ().strip ("\n").split ("\t")[1:]
    features = [line.split ("\t")[0] for line in f.readlines ()]
    
if not os.path.exists (args.output):
    os.makedirs (args.output, exist_ok = True)

if isinstance (list (concepts.values ())[0], dict):
    default = concepts.get (defaultName, dict ())
    if direction == "sample":
        maxSplit = 2
        for sample in samples:
            with open (args.mtx) as f:
                values = pd.Series ([line.strip ("\n").split ("\t", maxsplit = maxSplit)[-2] for line in f.readlines ()[1:]], index = features)
            values[values == ""] = np.nan; values = values.astype (float); concept = concepts.get (sample, default).copy (); maxSplit += 1
            concept["label_values"] = [const.get (x.lower ()) if isinstance (x, str) else x for x in concept.get ("label_values", list ())]
            memberships = fuzzify (values, concept, renameLabels = renameLabels).round (3)
            if not memberships.empty:
                memberships.to_csv (os.path.join (args.output, f"fuzzyValues_{sample}.tsv"), sep = "\t")
    else:
        with open (args.mtx) as f:
            _ = f.readline ()
            for feature in features:
                values = pd.Series ([np.nan if x == "" else float (x) for x in f.readline ().strip ("\n").split ("\t")[1:]], index = samples)
                if direction == "feature":
                    concept = concepts.get (feature, default).copy ()
                else:
                    concept = default.copy ()
                concept["label_values"] = [const.get (x.lower ()) if isinstance (x, str) else x for x in concept.get ("label_values", list ())]
                memberships = fuzzify (values, concept, renameLabels = renameLabels).round (3)
                if not memberships.empty:
                    memberships.to_csv (os.path.join (args.output, f"fuzzyValues_{feature}.tsv"), sep = "\t")
else:
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
    with open (args.mtx) as f:
        values = [[np.nan if x == "" else float (x) for x in line.strip ("\n").split ("\t")[1:]] for line in f.readlines ()[1:]]
    values = pd.Series (sum (values, list ()))
    default = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                          minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                          refConcept = concept_cons, consValue = consValue,
                          useFit = False, useOptimize = False, bwFct = bwFct)
    defaultOutput = default.copy (); defaultOutput["label_values"] = outputLabels; allConcepts = {defaultName: defaultOutput}
    del values
    if direction == "sample":
        maxSplit = 2
        for sample in samples:
            with open (args.mtx) as f:
                values = pd.Series ([line.strip ("\n").split ("\t", maxsplit = maxSplit)[-2] for line in f.readlines ()[1:]], index = features)
            values[values == ""] = np.nan; values = values.astype (float); maxSplit += 1 
            concept = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                                  minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                  refConcept = concept_cons, consValue = consValue,
                                  useFit = useFit, useOptimize = useOptimize, bwFct = bwFct)
            if concept["number_fuzzy_sets"] == 0:
                concept = default.copy ()
            else:
                outputConcept = concept.copy (); outputConcept["label_values"] = outputLabels; allConcepts[sample] = outputConcept
            memberships = fuzzify (values, concept, renameLabels = renameLabels).round (3)
            if not memberships.empty:
                memberships.to_csv (os.path.join (args.output, f"fuzzyValues_{sample}.tsv"), sep = "\t")
    else:
        with open (args.mtx) as f:
            _ = f.readline ()
            for feature in features:
                values = pd.Series ([np.nan if x == "" else float (x) for x in f.readline ().strip ("\n").split ("\t")[1:]], index = samples)
                if direction == "feature":
                    concept = getConcept (values, "constraint", consType, basicInfo, numFS, renameFS, labels,
                                          minLevelCons, minLevelPct, maxLevelCons, maxLevelPct,
                                          refConcept = concept_cons, consValue = consValue,
                                          useFit = useFit, useOptimize = useOptimize, bwFct = bwFct)
                    if concept["number_fuzzy_sets"] == 0:
                        concept = default.copy ()
                    else:
                        outputConcept = concept.copy (); outputConcept["label_values"] = outputLabels; allConcepts[feature] = outputConcept
                else:
                    concept = default.copy ()
                memberships = fuzzify (values, concept, renameLabels = renameLabels).round (3)
                if not memberships.empty:
                    memberships.to_csv (os.path.join (args.output, f"fuzzyValues_{feature}.tsv"), sep = "\t")
    with open (os.path.join (args.output, "concepts_detailed.json"), "w", encoding = "utf-8") as f:
        json.dump (allConcepts, f, ensure_ascii = False, indent = 4, allow_nan = True)


