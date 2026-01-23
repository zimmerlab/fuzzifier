import os
import argparse
import numpy as np
import pandas as pd
from helper_functions import getAnalysis, getAnalysis_combinedMtx

# python run_hypothesis_tests.py --data dataDirectory --output outputDirectory


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--data", type = str, required = True, help = "Directory for input tables and metadata")
    parser.add_argument ("--output", type = str, required = True, help = "Output directory for visualizations")
    args = parser.parse_args ()

    adultData = pd.read_csv (os.path.join (args.data, "adult_data_ENABLE.tsv"), index_col = 0, sep = "\t")
    adultMetadata = pd.read_csv (os.path.join (args.data, "adult_metadata_ENABLE.tsv"), index_col = 0, sep = "\t")
    kidData_N = pd.read_csv (os.path.join (args.data, "kid_neutrophil_data_ENABLE.tsv"), index_col = 0, sep = "\t")
    kidMetadata_N = pd.read_csv (os.path.join (args.data, "kid_neutrophil_metadata.tsv"), index_col = 0, sep = "\t")
    kidData_T = pd.read_csv (os.path.join (args.data, "kid_T-cell_data_ENABLE.tsv"), index_col = 0, sep = "\t")
    kidMetadata_T = pd.read_csv (os.path.join (args.data, "kid_T-cell_metadata.tsv"), index_col = 0, sep = "\t")
    kidMetadata = kidMetadata_N.reset_index ().merge (kidMetadata_T.reset_index (), on = "sample", how = "inner")
    kidMetadata = pd.DataFrame ({"sample": kidMetadata["sample"], "category": kidMetadata["category_x"].str.split ("-", expand = True)[0],
                                 "index_neutrophil": kidMetadata["index_x"], "index_T-cell": kidMetadata["index_y"]})
    if not os.path.exists (args.output):
        os.makedirs (args.output, exist_ok = True)
    # test 1
    os.makedirs (os.path.join (args.output, "test1", "data_set"), exist_ok = True)
    os.makedirs (os.path.join (args.output, "test1", "normalweight"), exist_ok = True)
    os.makedirs (os.path.join (args.output, "test1", "overweight"), exist_ok = True)
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (adultData); rightMtx = np.log2 (kidData_N)
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "adult", right_prefix = "kid",
                 output_directory = os.path.join (args.output, "test1", "data_set"))
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (adultData[adultMetadata.loc[adultMetadata["category"].isin (["underweight", "normalweight"])].index])
        rightMtx = np.log2 (kidData_N[kidMetadata_N.loc[kidMetadata_N["category"].isin (["kNmA-N", "kNmNA-N"])].index])
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "adult", right_prefix = "kid",
                 output_directory = os.path.join (args.output, "test1", "normalweight"))
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (adultData[adultMetadata.loc[adultMetadata["category"].isin (["overweight", "obesity"])].index])
        rightMtx = np.log2 (kidData_N[kidMetadata_N.loc[kidMetadata_N["category"].isin (["kUmA-N", "kUmNA-N"])].index])
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "adult", right_prefix = "kid",
                 output_directory = os.path.join (args.output, "test1", "overweight"))
    # test2
    os.makedirs (os.path.join (args.output, "test2"), exist_ok = True)
    getAnalysis_combinedMtx (adultData, adultMetadata, kidData_N, kidMetadata_N, os.path.join (args.output, "test2"))
    # test 3
    os.makedirs (os.path.join (args.output, "test3"), exist_ok = True)
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (kidData_N[kidMetadata_N.loc[kidMetadata_N["category"] == "kUmA-N"].index])
        rightMtx = np.log2 (kidData_N[kidMetadata_N.loc[kidMetadata_N["category"] != "kUmA-N"].index])
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "kUmA-N", right_prefix = "kXmX-N",
                 output_directory = os.path.join (args.output, "test3"))
    # test 4
    os.makedirs (os.path.join (args.output, "test4"), exist_ok = True)
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (kidData_N[kidMetadata_N.loc[kidMetadata_N["category"].isin (["kUmA-N", "kUmNA-N"])].index])
        rightMtx = np.log2 (kidData_N[kidMetadata_N.loc[kidMetadata_N["category"].isin (["kNmA-N", "kNmNA-N"])].index])
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "kUmX-N", right_prefix = "kNmX-N",
                 output_directory = os.path.join (args.output, "test4"))
    # test 5
    os.makedirs (os.path.join (args.output, "test5"), exist_ok = True)
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (kidData_T[kidMetadata_T.loc[kidMetadata_T["category"] == "kUmA-T"].index])
        rightMtx = np.log2 (kidData_T[kidMetadata_T.loc[kidMetadata_T["category"] != "kUmA-T"].index])
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "kUmA-T", right_prefix = "kXmX-T",
                 output_directory = os.path.join (args.output, "test5"))
    # test 6
    os.makedirs (os.path.join (args.output, "test6"), exist_ok = True)
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (kidData_T[kidMetadata_T.loc[kidMetadata_T["category"].isin (["kUmA-T", "kUmNA-T"])].index])
        rightMtx = np.log2 (kidData_T[kidMetadata_T.loc[kidMetadata_T["category"].isin (["kNmA-T", "kNmNA-T"])].index])
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "kUmX-N", right_prefix = "kNmX-N",
                 output_directory = os.path.join (args.output, "test6"))
    # test 7
    os.makedirs (os.path.join (args.output, "test7", "data_set"), exist_ok = True)
    os.makedirs (os.path.join (args.output, "test7", "normalweight"), exist_ok = True)
    os.makedirs (os.path.join (args.output, "test7", "overweight"), exist_ok = True)
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (kidData_N[kidMetadata["index_neutrophil"]]); rightMtx = np.log2 (kidData_T[kidMetadata["index_T-cell"]])
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "neutrophil", right_prefix = "T-cell",
                 output_directory = os.path.join (args.output, "test7", "data_set"))
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (kidData_N[kidMetadata.loc[kidMetadata["category"].isin (["kNmA", "kNmNA"]), "index_neutrophil"]])
        rightMtx = np.log2 (kidData_T[kidMetadata.loc[kidMetadata["category"].isin (["kNmA", "kNmNA"]), "index_T-cell"]])
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "neutrophil", right_prefix = "T-cell",
                 output_directory = os.path.join (args.output, "test7", "normalweight"))
    with np.errstate (divide = "ignore"):
        leftMtx = np.log2 (kidData_N[kidMetadata.loc[kidMetadata["category"].isin (["kUmA", "kUmNA"]), "index_neutrophil"]])
        rightMtx = np.log2 (kidData_T[kidMetadata.loc[kidMetadata["category"].isin (["kUmA", "kUmNA"]), "index_T-cell"]])
    getAnalysis (leftMtx, rightMtx, noise_cutoff = 6, low_value_percent = 0, bw_factor = 0.75,
                 num_fuzzy_sets = 3, fuzzy_variables = ["LOW", "MID", "HIGH"], width_factor = 1, slope_factor = 0.5,
                 axis_label = "log2 (non-zero count)", left_prefix = "neutrophil", right_prefix = "T-cell",
                 output_directory = os.path.join (args.output, "test7", "overweight"))



if __name__ == "__main__":
    main ()


