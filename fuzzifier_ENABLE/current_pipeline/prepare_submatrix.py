import os
import json
import argparse
import pandas as pd

# python prepare_submatrix.py --data dataDirectory --groups groupList --output outputFile


parser = argparse.ArgumentParser ()
parser.add_argument ("--data", type = str, required = True, help = "Directory with concatenated raw value matrix and metadata")
parser.add_argument ("--groups", type = str, required = True, help = "Comma-separated list of involved groups in the category")
parser.add_argument ("--output", type = str, required = True, help = "Output file path")
args = parser.parse_args ()

matrix = pd.read_csv (os.path.join (args.data, "raw_expression.tsv"), index_col = 0, sep = "\t")
metadata = pd.read_csv (os.path.join (args.data, "metadata.tsv"), index_col = None, sep = "\t")
cats = args.groups.split (",")
if len (cats) == 0:
    raise ValueError ("No entries were found.")
submatrix = matrix[metadata.loc[metadata["category"].isin (cats), "patient"]]

os.makedirs (os.path.dirname (args.output), exist_ok = True)
submatrix.to_csv (args.output, sep = "\t")


