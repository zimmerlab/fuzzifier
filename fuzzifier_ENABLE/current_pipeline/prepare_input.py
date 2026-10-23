import os
import argparse
import numpy as np
import pandas as pd

# python prepare_input.py --excel excelDirectory --base logBase --pseudo pseudoCount --output outputDirectory


parser = argparse.ArgumentParser ()
parser.add_argument ("--excel", type = str, required = True, help = "Directory for EXCEL table inputs")
parser.add_argument ("--base", type = int, required = False, default = 2, help = "Base for logarithmus")
parser.add_argument ("--pseudo", type = int, required = False, default = 0, help = "Pseudocount")
parser.add_argument ("--output", type = str, required = False, default = "./", help = "Output directory for raw value matrix and metadata")
args = parser.parse_args ()

if (not isinstance (args.base, int)) or args.base < 0:
    raise ValueError

fullMtx = pd.read_excel (os.path.join (args.excel, "RZ_Targeted Hits_Rohdaten_Erwachsene_enable1.xlsx"), header = None)
adult = fullMtx.iloc[[1, 0], 3:-2].T.astype (str); adult.columns = ["patient", "category"]
adultData = fullMtx.iloc[3:67, 3:-2]; adultData.index = fullMtx.iloc[3:67, 0].values; adultData.columns = fullMtx.iloc[1, 3:-2].values

fullMtx = pd.read_excel (os.path.join (args.excel, "RZ_Targeted Hits_Rohdaten_Kinder_90.Perzentile_Neutros_enable2.xlsx"), header = None)
kid_N = fullMtx.iloc[[1, 0], 3:-2].T.astype (str); kid_N.columns = ["patient", "category"]
kidData_N = fullMtx.iloc [3:67, 3:-2]; kidData_N.index = fullMtx.iloc[3:67, 0].values; kidData_N.columns = fullMtx.iloc[1, 3:-2].values

fullMtx = pd.read_excel (os.path.join (args.excel, "Targeted Hits_Rohdaten_Kinder_90.Perzentile_T Zellen_enable2.xlsx"), header = None)
kid_T = fullMtx.iloc[[1, 0], 3:].T.astype (str); kid_T.columns = ["patient", "category"]
kidData_T = fullMtx.iloc[3:, 3:]; kidData_T.index = fullMtx.iloc[3:, 0].values; kidData_T.columns = fullMtx.iloc[1, 3:].values

os.makedirs (args.output, exist_ok = True)
if args.base != 0:
    with np.errstate (divide = "ignore", invalid = "ignore"):
        data = np.log (pd.concat ([adultData, kidData_N, kidData_T], axis = 1).astype (int) + args.pseudo) / np.log (args.base)
data.to_csv (os.path.join (args.output, "raw_expression.tsv"), sep = "\t")
metadata = pd.concat ([adult, kid_N, kid_T], axis = 0)
metadata.to_csv (os.path.join (args.output, "metadata.tsv"), index = None, sep = "\t")


