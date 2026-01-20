import numpy as np
import pandas as pd


nameList = pd.read_excel ("231122_SuszynskaM_Supplementary_TableS1.xlsx", header = 1, index_col = 1).rename_axis (None, axis = 0)
nameList = nameList[["miRNA ID", "all miRNA precursors/loci (miRBase ID)", "miRNA gene confidence in miRBase "]]
nameList = nameList.rename (columns = {"miRNA ID": "ID", "all miRNA precursors/loci (miRBase ID)": "miRBase_ID",
                                       "miRNA gene confidence in miRBase ": "miRBase_gene_confidence"})

score = pd.read_excel ("231122_SuszynskaM_Supplementary_TableS4.xlsx", header = 1)
score = pd.Series (score["CMC score"].values, index = [nameList.loc[idx, "miRBase_ID"] for idx in score["background miRNA genes"]])

known = pd.read_excel ("231122_SuszynskaM_Supplementary_TableS5.xlsx", header = 2).replace (np.nan, 0)
known = known.set_index ("background miRNA genes").rename_axis (None, axis = 0)
known = known[[col for col in known.columns if "TCGA" in col]]
known = known.rename (index = {idx: nameList.loc[idx, "miRBase_ID"] for idx in known.index},
                      columns = {col: col.split (" ")[3].split ("-")[1] for col in known.columns})
known = known.reset_index (names = "feature").melt (id_vars = "feature", var_name = "cluster", value_name = "is_specific")
known = known.loc[known["is_specific"] == 1, ["feature", "cluster"]]; known["CMC_score"] = score[known["feature"]].values
known.to_csv ("known_cancer-specific.tsv", index = None, sep = "\t")

information = pd.read_excel ("231122_SuszynskaM_Supplementary_TableS9.xlsx", header = 1, index_col = 1).rename_axis (None)
information = information.rename (columns = {"miRNA precursor/locus ID": "miRBase_ID", "miRNA ID": "miRNA_ID",
                                             "oncogene (O)/tumor-suppressor (TS)": "type"})
regulation = information.filter (regex = "^differentially expressed in")
regulation = regulation.rename (columns = {col: col.replace ("differentially expressed in ", "").split (" (")[0].replace ("TCGA-", "")
                                           for col in regulation.columns})
regulation = regulation.reset_index (names = "feature").melt (id_vars = "feature", var_name = "cluster", value_name = "regulation").dropna ()
regulation = regulation.replace ("DOWN", "--").replace ("UP", "++").reset_index (drop = True)
regulation.insert (1, "type", information.loc[regulation["feature"], "type"].values)
regulation.to_csv ("known_CMC_regulation.tsv", index = None, sep = "\t")


