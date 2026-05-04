import os
import argparse
import numpy as np
import pandas as pd

# python main_fuzzyRule.py --numerator numeratorDirectory --denominator denominatorDirectory --rules fuzzyRules --savedBy fileSavedBy --fuzzySets fuzzySetsForFC --output outputDirectory


def readFV (dir, savedBy, nameDict = dict ()):
    allFV = list ()
    if savedBy == "feature":
        featureList = nameDict["feature"] if "feature" in nameDict else sorted ([file[12:-4] for file in os.listdir (dir)])
        for feature in featureList:
            FV = pd.read_csv (os.path.join (dir, f"fuzzyValues_{feature}.tsv"), index_col = 0, sep = "\t")
            if "sample" in nameDict and "FS" in nameDict:
                allFV.append (FV.loc[nameDict["sample"], nameDict["FS"]].to_numpy ())
            else:
                allFV.append (FV.to_numpy ())
        allFV = np.array (allFV); nameDict = {"feature": featureList, "sample": list (FV.index), "FS": list (FV.columns)}
    elif savedBy == "sample":
        sampleList = nameDict["sample"] if "sample" in nameDict else sorted ([file[12:-4] for file in os.listdir (dir)])
        for sample in sampleList:
            FV = pd.read_csv (os.path.join (dir, f"fuzzyValues_{sample}.tsv"), index_col = 0, sep = "\t")
            if "feauture" in nameDict and "FS" in nameDict:
                allFV.append (FV.loc[nameDict["feature"], nameDict["FS"]].to_numpy ())
            else:
                allFV.append (FV.to_numpy ())
        allFV = np.einsum ("ijk -> jik", allFV); nameDict = {"feature": list (FV.index), "sample": sampleList, "FS": list (FV.columns)}
    else:
        allSets = nameDict["FS"] if "FS" in nameDict else sorted ([file[12:-4] for file in os.listdir (dir)])
        for FS in allSets:
            FV = pd.read_csv (os.path.join (dir, f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
            if "feature" in nameDict and "sample" in nameDict:
                allFV.append (FV.loc[nameDict["feature"], nameDict["sample"]].to_numpy ())
            else:
                allFV.append (FV.to_numpy ())
        allFV = np.einsum ("ijk -> jki", allFV); nameDict = {"feature": list (FV.index), "sample": list (FV.columns), "FS": allSets}
    return allFV, nameDict



def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--numerator", type = str, required = True, help = "Directory of fuzzy values in numerator samples")
    parser.add_argument ("--denominator", type = str, required = True, help = "Directory of fuzzy values in denominator samples")
    parser.add_argument ("--rules", type = str, required = True, help = "Table of fuzzy rules (CSV)")
    parser.add_argument ("--savedBy", type = str, required = True, help = "Fuzzy value files saved per feature / sample / fuzzy set")
    parser.add_argument ("--fuzzySets", type = str, required = True, help = "String of ordered list for log2FC fuzzy variables")
    parser.add_argument ("--output", type = str, required = True, help = "Ouput directory for fuzzy fold changes")
    args = parser.parse_args ()

    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)

    rules = pd.read_csv (args.rules, index_col = 0, sep = ",").replace (np.nan, "NA").replace (np.inf, "+INF").replace (-np.inf, "-INF")
    allSets = args.fuzzySets.split (","); print (allSets); print (rules)
    mask = {FS: rules.to_numpy () == FS for FS in allSets}
    for dir in set (os.listdir (args.numerator)) & set (os.listdir (args.denominator)):
        numDir = os.path.join (args.numerator, dir, "fuzzy_values"); denDir = os.path.join (args.denominator, dir, "fuzzy_values")
        if args.savedBy == "feature":
            nameDict = {"feature": [file[12:-4] for file in sorted (set (os.listdir (numDir)) & set (os.listdir (denDir))) if file.endswith (".tsv")],
                        "FS": list (rules.index)}
        elif args.savedBy == "sample":
            nameDict = {"sample": [file[12:-4] for file in sorted (set (os.listdir (numDir)) & set (os.listdir (denDir))) if file.endswith (".tsv")],
                        "FS": list (rules.index)}
        else:
            nameDict = {"FS": list (rules.index)}
        numeratorFV, nameDict = readFV (numDir, args.savedBy, nameDict = nameDict)
        denominatorFV, _ = readFV (denDir, args.savedBy, nameDict = {"feature": nameDict["feature"], "sample": nameDict["sample"],
                                                                     "FS": list (rules.columns)})
        prod = np.einsum ("ijk, ijl -> ijkl", numeratorFV, denominatorFV)
        fuzzyFC = np.array ([[[prod[i, j, :, :][mask[FS]].sum () for FS in allSets] for j in range (prod.shape[1])] for i in range (prod.shape[0])])
        for idx in range (len (nameDict["sample"])):
            sample = nameDict["sample"][idx]
            outputMtx = pd.DataFrame (fuzzyFC[:, idx, :], index = nameDict["feature"], columns = allSets).round (3)
            outputMtx.to_csv (os.path.join (args.output, f"fuzzyValues_{sample}.tsv"), sep = "\t")



if __name__ == "__main__":
    main ()


