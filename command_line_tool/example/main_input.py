import os
import argparse
import pandas as pd

# python main_input.py --min_pairs minimalNumberOfSamplePairs


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ("--min_pairs", type = int, required = True, help = "Minimal number of sample pairs per cancer type")
    args = parser.parse_args ()

    if not os.path.exists ("./miRNA/data/"):
        os.makedirs ("./miRNA/data/", exist_ok = True)
    metadata = pd.read_csv ("/mnt/extproj/projekte/modelling/interactive_fuzzifier/example_input/metadata_paired.tsv",
                            index_col = None, sep = "\t")[["sample", "miRNA_Tumor", "miRNA_Normal", "context"]]
    metadata = metadata.rename (columns = {"miRNA_Tumor": "tumor", "miRNA_Normal": "normal"}).drop_duplicates (["tumor", "normal", "context"])
    numPairs = metadata.value_counts ("context"); allClusters = sorted (numPairs[numPairs > args.min_pairs].index)
    metadata = metadata.loc[metadata["context"].isin (allClusters)]; print (metadata.shape[0])
    metadata.to_csv ("./miRNA/data/metadata.tsv", index = None, sep = "\t")
    fullMtx = pd.concat ([pd.read_csv (f"/mnt/raidtmp/panc/TCGA_counts/{cluster}_miRNA.tsv", index_col = 0, sep = "\t")
                          for cluster in allClusters],
                         axis = 1)
    fullMtx = fullMtx.astype (int)
    pctExpressed = pd.DataFrame ({f"{C}_{T}": (fullMtx[metadata.loc[metadata["context"] == C, T]] != 0).mean (axis = 1)
                                  for C in allClusters for T in ["tumor", "normal"]})
    allExpressed = list (pctExpressed.index[(pctExpressed > 0.5).all (axis = 1)])
    onlyExpressed = list (pctExpressed.index[((pctExpressed > 0.7).sum (axis = 1) == 1) &
                                             ((pctExpressed < 0.1).sum (axis = 1) == 2 * len (allClusters) - 1)])
    mtx = fullMtx.loc[sorted (allExpressed + onlyExpressed), list (set (metadata["tumor"])) + list (set (metadata["normal"]))]
    mtx.to_csv ("./miRNA/data/raw_counts.tsv", sep = "\t"); print (mtx.shape)

    if not os.path.exists ("./RNA/data/"):
        os.makedirs ("./RNA/data/", exist_ok = True)
    geneType = pd.read_csv ("/home/users/pan/TCGA_benchmark/RNA_rowRanges.tsv", index_col = None, sep = "\t")
    geneType = list (set (geneType.loc[(~geneType["seqnames"].isin (["chrX", "chrY"])) & (geneType["gene_type"] == "protein_coding"), "gene_name"]))
    metadata = pd.read_csv ("/mnt/extproj/projekte/modelling/interactive_fuzzifier/example_input/metadata_paired.tsv",
                            index_col = None, sep = "\t")[["sample", "RNA_Tumor", "RNA_Normal", "context"]]
    metadata = metadata.rename (columns = {"RNA_Tumor": "tumor", "RNA_Normal": "normal"}).drop_duplicates (["tumor", "normal", "context"])
    numPairs = metadata.value_counts ("context"); allClusters = sorted (numPairs[numPairs > args.min_pairs].index)
    metadata = metadata.loc[metadata["context"].isin (allClusters)]; print (metadata.shape[0])
    metadata.to_csv ("./RNA/data/metadata.tsv", index = None, sep = "\t")
    fullMtx = pd.concat ([pd.read_csv (f"/mnt/raidtmp/panc/TCGA_counts/{cluster}_RNA.tsv", index_col = 0, sep = "\t")
                          for cluster in allClusters],
                         axis = 1)
    fullMtx = fullMtx.loc[geneType].astype (int)
    pctExpressed = pd.DataFrame ({f"{C}_{T}": (fullMtx[metadata.loc[metadata["context"] == C, T]] != 0).mean (axis = 1)
                                  for C in allClusters for T in ["tumor", "normal"]})
    allExpressed = list (pctExpressed.index[(pctExpressed > 0.5).all (axis = 1)])
    onlyExpressed = list (pctExpressed.index[((pctExpressed > 0.7).sum (axis = 1) == 1) &
                                             ((pctExpressed < 0.1).sum (axis = 1) == 2 * len (allClusters) - 1)])
    mtx = fullMtx.loc[sorted (allExpressed + onlyExpressed), list (set (metadata["tumor"])) + list (set (metadata["normal"]))]
    mtx.to_csv ("./RNA/data/raw_counts.tsv", sep = "\t"); print (mtx.shape)



if __name__ == "__main__":
    main ()


