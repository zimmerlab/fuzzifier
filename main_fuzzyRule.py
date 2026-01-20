import os
import argparse
import numpy as np
import pandas as pd

# python main_fuzzyRule.py --numerator numeratorDirectory --denominator denominatorDirectory --output outputDirectory


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--numerator", type = str, required = True, help = "Directory of fuzzy values in numerator samples")
    parser.add_argument ("--denominator", type = str, required = True, help = "Directory of fuzzy values in denominator samples")
    parser.add_argument ("--output", type = str, required = True, help = "Ouput directory for fuzzy fold changes")
    args = parser.parse_args ()

    numeratorFV = list (); denominatorFV = list ()
    for FS in [f"FS{i}" for i in range (1, 6)]:
        memberships = pd.read_csv (os.path.join (args.numerator, f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
        numeratorFV.append (memberships.to_numpy ())
    nameList = {"feature": list (memberships.index), "sample": list (memberships.columns)}
    for FS in [f"FS{i}" for i in range (1, 6)]:
        memberships = pd.read_csv (os.path.join (args.denominator, f"fuzzyValues_{FS}.tsv"), index_col = 0, sep = "\t")
        denominatorFV.append (memberships.loc[nameList["feature"], nameList["sample"]].to_numpy ())
    numeratorFV = np.einsum ("ijk -> jki", numeratorFV); denominatorFV = np.einsum ("ijk -> jki", denominatorFV)
    
    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)
    allSets = ["NA", "-INF", "INF", "--", "-", "o", "+", "++"]
    numeratorFV = np.array ([(numeratorFV == 0).all (axis = 2), numeratorFV[:, :, 0] + numeratorFV[:, :, 1],
                             numeratorFV[:, :, 2], numeratorFV[:, :, 3] + numeratorFV[:, :, 4]])
    denominatorFV = np.array ([(denominatorFV == 0).all (axis = 2), denominatorFV[:, :, 0] + denominatorFV[:, :, 1],
                               denominatorFV[:, :, 2], denominatorFV[:, :, 3] + denominatorFV[:, :, 4]])
    dist = np.einsum ("kij, lij -> ijkl", numeratorFV, denominatorFV)
    fuzzyLog2FC = np.array ([[[dist[i, j][0, 0], dist[i, j][0, 3], dist[i, j][3, 0],
                               dist[i, j][0, 2] + dist[i, j][1, 3], dist[i, j][0, 1] + dist[i, j][1, 2] + dist[i, j][2, 3],
                               dist[i, j][1, 1] + dist[i, j][2, 2] + dist[i, j][3, 3],
                               dist[i, j][1, 0] + dist[i, j][2, 1] + dist[i, j][3, 2], dist[i, j][2, 0] + dist[i, j][3, 1]]
                              for j in range (dist.shape[1])]
                             for i in range (dist.shape[0])])
    for idx in range (len (allSets)):
        outputMtx = pd.DataFrame (fuzzyLog2FC[:, :, idx], index = nameList["feature"], columns = nameList["sample"]).round (3)
        outputMtx.to_csv (os.path.join (args.output, f"fuzzyValues_{allSets[idx]}.tsv"), sep = "\t")



if __name__ == "__main__":
    main ()


