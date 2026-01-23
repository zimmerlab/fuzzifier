import os
import argparse
import numpy as np
import pandas as pd

# python main_CMC-validation.py --data rawDataDirectory --metadata metadata --result comparisonResults --reference directoryCMC --cmcCut scoreCutoff


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--data", type = str, required = True, help = "Directory for splitted matrices")
    parser.add_argument ("--metadata", type = str, required = True, help = "Metadata containing clustering column (TSV)")
    parser.add_argument ("--result", type = str, required = True, help = "Comparison tables as results of the analysis pipeline (TSV)")
    parser.add_argument ("--reference", type = str, required = True, help = "Directory of summarized CMC tables")
    parser.add_argument ("--cmcCut", type = float, required = False, default = 0, help = "Cutoff for minimal CMC score")
    args = parser.parse_args ()

    numerator = pd.read_csv (os.path.join (args.data, "numerator_log.tsv"), index_col = 0, sep = "\t")
    denominator = pd.read_csv (os.path.join (args.data, "denominator_log.tsv"), index_col = 0, sep = "\t")
    metadata = pd.read_csv (args.metadata, index_col = None, sep = "\t")
    if metadata.columns[0] == "Unnamed: 0":
        metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
    candidates = pd.read_csv (args.result, index_col = None, sep = "\t")
    candidates = candidates.loc[candidates["method"] != "fuzzy EMD-like distance"]
    numClusters = candidates.drop ("method", axis = 1).drop_duplicates ().value_counts ("feature")
    markers = candidates.loc[candidates["feature"].isin (numClusters[numClusters == 1].index)]

    known_specific = pd.read_csv (os.path.join (args.reference, "known_cancer-specific.tsv"), index_col = None, sep = "\t")
    known_specific = known_specific.loc[known_specific["CMC_score"] >= args.cmcCut]
    known_regulation = pd.read_csv (os.path.join (args.reference, "known_CMC_regulation.tsv"), index_col = None, sep = "\t")

    validated = candidates.merge (known_specific, on = ["feature", "cluster"], how = "inner")
    validated = validated.merge (known_regulation, on = ["feature", "cluster"], how = "left", suffixes = ("_identified", "_known")).replace (np.nan, "")
    validated = validated.loc[(validated["regulation_identified"] == validated["regulation_known"]) | (validated["regulation_known"] == "")]
    validated = validated[["feature", "type", "cluster", "regulation_identified", "regulation_known", "CMC_score", "method"]]
    novels = candidates.merge (validated, on = ["feature", "cluster", "method"], how = "left")
    novels = novels.loc[np.isnan (novels["CMC_score"]), ["feature", "cluster", "regulation", "method"]]
    idxList = list ()
    for idx in novels.index:
        feature, cluster, regulation = novels.loc[idx, ["feature", "cluster", "regulation"]]
        if regulation == "--":
            pctExpressed = np.isfinite (denominator.loc[feature, metadata.loc[metadata["context"] == cluster, "sample"]]).mean ()
        else:
            pctExpressed = np.isfinite (numerator.loc[feature, metadata.loc[metadata["context"] == cluster, "sample"]]).mean ()
        if pctExpressed > 0.5:
            idxList.append (idx)
    novel_specific = novels.loc[idxList].merge (candidates, how = "left")

    validated_markers = markers.merge (known_specific, on = ["feature", "cluster"], how = "inner")
    validated_markers = validated_markers.merge (known_regulation, on = ["feature", "cluster"], how = "left", suffixes = ("_identified", "_known")).replace (np.nan, "")
    validated_markers = validated_markers.loc[(validated_markers["regulation_identified"] == validated_markers["regulation_known"]) | (validated_markers["regulation_known"] == "")]
    validated_markers = validated_markers[["feature", "type", "cluster", "regulation_identified", "regulation_known", "CMC_score", "method"]]
    novels = sorted (set (markers["feature"]) - set (validated_markers["feature"]))
    novels = markers.loc[markers["feature"].isin (novels)].drop ("method", axis = 1).drop_duplicates ()
    idxList = list ()
    for idx in novels.index:
        feature, cluster, regulation = novels.loc[idx]
        if regulation == "--":
            pctExpressed = np.isfinite (denominator.loc[feature, metadata.loc[metadata["context"] == cluster, "sample"]]).mean ()
        else:
            pctExpressed = np.isfinite (numerator.loc[feature, metadata.loc[metadata["context"] == cluster, "sample"]]).mean ()
        if pctExpressed > 0.5:
            idxList.append (idx)
    novel_markers = markers.merge (markers.loc[idxList, ["feature", "cluster", "regulation"]], how = "inner")
    
    dir = os.path.abspath (os.path.dirname (args.result))
    validated.to_csv (os.path.join (dir, "comparison_validated.tsv"), index = None, sep = "\t")
    validated_markers.to_csv (os.path.join (dir, "comparison_validated_markers.tsv"), index = None, sep = "\t")
    novel_specific.to_csv (os.path.join (dir, "candidates_novel_specific_CMC.tsv"), index = None, sep = "\t")
    novel_markers.to_csv (os.path.join (dir, "candidates_novel_markers_CMC.tsv"), index = None, sep = "\t")



if __name__ == "__main__":
    main ()


