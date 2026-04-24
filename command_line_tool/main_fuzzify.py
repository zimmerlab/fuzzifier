import warnings
warnings.simplefilter (action = "ignore", category = FutureWarning)

import os
import json
import argparse
import numpy as np
from helper_functions import getFuzzy

# python main_fuzzifier.py --mtx rawValueMatrix --concept fuzzyConcept(s) --config config --output outputDirectory


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
minLevelCons = config.get ("left_noise_cutoff_constant", -np.inf); maxLevelCons = config.get ("right_noise_cutoff_constant", np.inf)
minLevelCons = const.get (minLevelCons.lower (), -np.inf) if isinstance (minLevelCons, str) else minLevelCons
maxLevelCons = const.get (maxLevelCons.lower (), np.inf) if isinstance (maxLevelCons, str) else maxLevelCons
minLevelPct = config.get ("left_noise_cutoff_percent", 0); maxLevelPct = config.get ("right_noise_cutoff_percent", 1)
minLevelPct = const.get (minLevelPct.lower (), 0) if isinstance (minLevelPct, str) else minLevelPct
maxLevelPct = const.get (maxLevelPct.lower (), 1) if isinstance (maxLevelPct, str) else maxLevelPct
defaultName = config.get ("key_default_concept", "DEFAULT"); direction = config.get ("fuzzify_per", "feature")
renameLabels = config.get ("rename_labels", dict ())
renameLabels = {const.get (val.lower ()): renameLabels[val] for val in renameLabels.keys () if isinstance (val, str)}
    
if not os.path.exists (args.output):
    os.makedirs (args.output, exist_ok = True)

if isinstance (list (concepts.values ())[0], dict):
    getFuzzy (args.mtx, direction, concepts, defaultName, renameLabels, args.output, deriveConcepts = False)
else:
    getFuzzy (args.mtx, direction, concepts, defaultName, renameLabels, args.output, deriveConcepts = True,
              minLevelCons = minLevelCons, minLevelPct = minLevelPct,
              maxLevelCons = maxLevelCons, maxLevelPct = maxLevelPct)


