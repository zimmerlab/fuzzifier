import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from evaluation_plots import plotConcept, plotCertaintySummary, plotImpurity


def getMarkers (allFV, itemList, numFuzzySets, indicateValues, clustering, baseLevel, maxNumCluster, minPctMainFS):
    allSets = [0] * len (indicateValues) + list (range (1, 1 + numFuzzySets))
    mainFS = pd.DataFrame (allFV.argmax (axis = 2), index = itemList["feature"], columns = itemList["sample"])
    mainFS = mainFS.replace ({i: allSets[i] for i in range (len (allSets))})
    sampleIdx = pd.Series (range (len (itemList["sample"])), index = itemList["sample"])
    pctMainFS = pd.DataFrame ({"feature": pd.Series (dtype = str), "cluster": pd.Series (dtype = str),
                               "mainFS": pd.Series (dtype = int), "pctMainFS": pd.Series (dtype = float),
                               "avgFV": pd.Series (dtype = float)})
    for cluster in sorted (set (clustering["cluster"])):
        sampleList = clustering[clustering["cluster"] == cluster].index; tmp = mainFS[sampleList]
        avgFV = allFV[:, sampleIdx[sampleList], :]
        avgFV = pd.concat ([pd.DataFrame (avgFV[:, :, :len (indicateValues)].sum (axis = 2).mean (axis = 1),
                                          index = itemList["feature"], columns = [0]),
                            pd.DataFrame (avgFV[:, :, len (indicateValues):].mean (axis = 1), index = itemList["feature"],
                                          columns = range (1, 1 + numFuzzySets))],
                           axis = 1).round (3)
        pct = pd.DataFrame ({idx: tmp.loc[idx].value_counts (normalize = True) for idx in mainFS.index})
        pct = pct.replace (np.nan, 0).T.round (3)
        missing = list (set (range (len (indicateValues) + numFuzzySets)) - set (pct.columns))
        if len (missing) > 0:
            pct = pd.concat ([pct, pd.DataFrame (0, index = pct.index, columns = missing)], axis = 1).sort_index (axis = 1)
        pct = pct.reset_index (names = "feature").melt (id_vars = "feature", var_name = "mainFS", value_name = "pctMainFS")
        pct.insert (1, "cluster", cluster); pct["avgFV"] = avgFV.melt ()["value"]
        pctMainFS = pd.concat ([pctMainFS, pct], axis = 0, ignore_index = True)
    markers = set (); isMarker = dict ()
    mainLevel = pctMainFS.sort_values ("pctMainFS", ascending = False).groupby (["feature", "cluster"]).head (1)
    mainLevel["supported"] = (mainLevel["pctMainFS"] > minPctMainFS) & (mainLevel["avgFV"] > 0.6)
    for idx in range (1 + baseLevel, 1 + numFuzzySets):
        higherLevel = mainLevel.copy (); higherLevel["higher"] = (mainLevel["mainFS"] >= idx).values
        higherLevel["isCandidate"] = higherLevel["higher"] & higherLevel["supported"]
        numSpecific = higherLevel.groupby ("feature")["isCandidate"].sum (); specific = numSpecific[(numSpecific > 0) & (numSpecific <= maxNumCluster)]
        markers = markers.union (set (specific.index)); isMarker[f"FS{idx}"] = list (specific.index)
    for idx in range (1, baseLevel)[::-1]:
        lowerLevel = mainLevel.copy (); lowerLevel["lower"] = ((mainLevel["mainFS"] != 0) & (mainLevel["mainFS"] <= idx)).values
        lowerLevel["isCandidate"] = lowerLevel["lower"] & lowerLevel["supported"]
        numSpecific = lowerLevel.groupby ("feature")["isCandidate"].sum (); specific = numSpecific[(numSpecific > 0) & (numSpecific <= maxNumCluster)]
        markers = markers.union (set (specific.index)); isMarker[f"FS{idx}"] = list (specific.index)
    onlyExpressed = mainLevel.copy (); onlyExpressed["expressed"] = (mainLevel["mainFS"] != 0)
    numSpecific = onlyExpressed.groupby ("feature")["expressed"].sum ()
    specific = numSpecific[(numSpecific > 0) & (numSpecific <= maxNumCluster)]
    specific = set (onlyExpressed.loc[onlyExpressed["feature"].isin (specific.index) & onlyExpressed["supported"], "feature"])
    markers = markers.union (specific); isMarker["FS0"] = list (specific)
    markers = mainLevel.loc[mainLevel["feature"].isin (markers)].copy ()
    markers["mainFS"] = [f"FS{i}" for i in markers["mainFS"]]; markers["isMarker"] = False
    for FS in isMarker.keys ():
        if FS == "FS0":
            markers.loc[markers["feature"].isin (isMarker[FS]) & (markers["mainFS"] != FS) & markers["supported"], "isMarker"] = True
        else:
            markers.loc[markers["feature"].isin (isMarker[FS]) & (markers["mainFS"] == FS) & markers["supported"], "isMarker"] = True
    markers = markers[["feature", "cluster", "mainFS", "pctMainFS", "avgFV", "isMarker"]]
    numClusters = markers.groupby ("feature")["isMarker"].sum ()
    markers = markers.loc[markers["feature"].isin (numClusters[(numClusters > 0) & (numClusters <= maxNumCluster)].index)]
    markers = markers.sort_values (["feature", "cluster"]).reset_index (drop = True)
    return markers



def getCertaintyStats (allFV, itemList, numFuzzySets, indicateValues):
    mainFV = np.partition (allFV, -2)[:, :, -2:]
    secondMainFV = pd.DataFrame (mainFV[:, :, 0], index = itemList["feature"], columns = itemList["sample"])
    mainFV = pd.DataFrame (mainFV[:, :, 1], index = itemList["feature"], columns = itemList["sample"])
    realSets = [f"FS{i}" for i in range (1, 1 + numFuzzySets)]
    allSets = [f"FS0_{x}" for x in indicateValues] + realSets
    mainFS = pd.DataFrame (allFV.argmax (axis = 2), index = itemList["feature"], columns = itemList["sample"])
    mainFS = mainFS.replace ({i: allSets[i] for i in range (len (allSets))})
    diffMainFV = mainFV - secondMainFV
    return mainFV, mainFS, diffMainFV



def getImpurity (allFV, itemList, clustering, nameFuzzySets):
    gini = list ()
    for idx in range (len (itemList["feature"])):
        memberships = pd.concat ([clustering.loc[itemList["sample"]],
                                  pd.DataFrame (allFV[idx, :, :], index = itemList["sample"],
                                                columns = nameFuzzySets)],
                                 axis = 1)
        prob = (memberships.groupby ("cluster").sum () / memberships[nameFuzzySets].sum (axis = 0))
        #gini.append ((1 - (prob ** 2).sum (axis = 0)).to_dict ())
        gini.append (((1 - (prob ** 2).sum (axis = 0)) * (~np.isnan (prob).all (axis = 0))).to_dict ())
    gini = pd.DataFrame.from_dict (gini, orient = "columns").round (3); gini.index = itemList["feature"]
    return gini



def downloadFiles (allFV, items, clusters, labels, concept, numFS, info, typeFS, valueRange, renameDict, colours,
                   sizeCol, baseLevel, maxNumCluster, minPctMainFS, outputDir):
    markers = getMarkers (allFV, items, numFS, labels, clusters, baseLevel, maxNumCluster, minPctMainFS)
    markers["mainFS"] = markers["mainFS"].replace (renameDict)
    markers.to_csv (os.path.join (outputDir, "marker_statistics.tsv"), index = None, sep = "\t")
    mainFV, mainFS, diffMainFV = getCertaintyStats (allFV, items, numFS, labels)
    mainFV.to_csv (os.path.join (outputDir, "main_fuzzy_values.tsv"), sep = "\t")
    mainFS = mainFS.replace (renameDict)
    mainFS.to_csv (os.path.join (outputDir, "main_fuzzy_sets.tsv"), sep = "\t")
    diffMainFV.to_csv (os.path.join (outputDir, "diff_main_fuzzy_values.tsv"), sep = "\t")
    impurity = getImpurity (allFV, items, clusters, list (renameDict.keys ()))
    impurity = impurity.rename (columns = renameDict)
    impurity.to_csv (os.path.join (outputDir, "gini_impurity.tsv"), sep = "\t")
    plotConcept (concept, typeFS, info, valueRange, colours = colours, savePlot = True,
                 savePlotPath = os.path.join (outputDir, "globalConcept.png"))
    nameLabels = [renameDict[key] for key in renameDict.keys () if key.startswith ("FS0_")]
    nameSets = [renameDict[key] for key in renameDict.keys () if not key.startswith ("FS0_")]
    plotCertaintySummary (mainFV, mainFS, diffMainFV, nameLabels, nameSets,
                          savePlot = True, savePlotPath = os.path.join (outputDir, "summaryFV.png"))
    plotImpurity (impurity, nameLabels, nameSets, savePlot = True,
                  savePlotPath = os.path.join (outputDir, "gini_impurity.png"))
    featureList = sorted (set (markers["feature"]))
    allClusters = sorted (set (clusters["cluster"])); allSets = ["FS0"] + nameSets
    colourDict = dict (zip (allSets, ["black"] + colours))
    for i in range (int (np.ceil (len (featureList) / 50))):
        startIdx = 50 * i; endIdx = min (len (featureList), 50 * (i + 1))
        partialList = featureList[startIdx:endIdx]
        pltData = markers.loc[markers["feature"].isin (partialList)]
        pltData = pltData.rename (columns = {"mainFS": "main fuzzy set",
                                             "pctMainFS": "percent of samples",
                                             "avgFV": "average fuzzy value"})
        fig, ax = plt.subplots (1, figsize = (8, 10))
        for i in range (len (allClusters)):
            ax.axvline (i, color = "lightgray", linestyle = "dashed")
        if sizeCol == "avgFV":
            sns.scatterplot (pltData, x = "cluster", y = "feature", size = "average fuzzy value",
                             hue = "main fuzzy set", hue_order = allSets, palette = colourDict, ax = ax)
        elif sizeCol == "pctMain":
            sns.scatterplot (pltData, x = "cluster", y = "feature", size = "percent of samples",
                             hue = "main fuzzy set", hue_order = allSets, palette = colourDict, ax = ax)
        else:
            raise ValueError
        ax.set_xticks (range (len (allClusters)))
        ax.set_xticklabels (allClusters, rotation = 60, ha = "right", size = 7.5)
        ax.set_yticks (range (len (partialList))); ax.set_yticklabels (partialList, size = 7.5)
        ax.legend (loc = (1.05, 0.5), facecolor = "white", fontsize = 10)
        ax.set_xlabel (""); ax.set_ylabel (""); fig.tight_layout ()
        plt.savefig (os.path.join (outputDir, "markers", f"marker_scatter_{startIdx + 1}.png")); plt.close ()


