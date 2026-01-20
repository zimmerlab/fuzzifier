import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


def plot_concept (concept, allSets, colors, xRange, xLabel, outputPath):
    xValues = np.linspace (*xRange, 1000); concept_copy = concept.copy ()
    concept_copy[0][0] = xRange[0] - 1; concept_copy[0][1] = xRange[0] - 1
    concept_copy[-1][2] = xRange[1] + 1; concept_copy[-1][3] = xRange[1] + 1
    fig, ax = plt.subplots (figsize = (6, 3))
    for i in range (len (concept)):
        c = concept_copy[i]
        if len (c) == 2:
            ax.plot (xValues, stats.norm.pdf (xValues, loc = c[0], scale = c[1]) * (c[1] * np.sqrt (2 * np.pi)),
                     color = colors[i], label = allSets[i])
        else:
            ax.plot ((c[0], c[1]), (0, 1), colors[i]); ax.plot ((c[1], c[2]), (1, 1), colors[i])
            ax.plot ((c[2], c[3]), (1, 0), colors[i], label = allSets[i])
    ax.set_xlabel (xLabel, size = 10); ax.set_ylabel ("fuzzy value", size = 10)
    ax.set_xlim (xRange); ax.set_ylim ((0, 1.05)); ax.legend (loc = (1.01, 0.25))
    fig.tight_layout (); plt.savefig (outputPath); plt.close ()



def heatmap_1dim (allFV, allSets, nameDict, clustering, colorDict, plotTitle, outputPath):
    mainFS = pd.DataFrame (allFV.argmax (axis = 2), index = nameDict["feature"], columns = nameDict["sample"])
    pctMainFS = list (); allClusters = sorted (set (clustering))
    for cluster in allClusters:
        tmp = mainFS[clustering[clustering == cluster].index].reset_index ().melt (id_vars = "index")
        tmp = tmp.groupby ("index")["value"].value_counts (normalize = True)
        tmp = tmp.reset_index ().pivot (index = "index", columns = "value", values = "proportion")
        tmp = tmp.rename_axis (None).rename_axis (None, axis = 1).replace (np.nan, 0)
        tmp = tmp.reset_index (names = "feature").melt (id_vars = "feature", var_name = "mainFS", value_name = "pctMainFS")
        tmp["cluster"] = cluster; pctMainFS.append (tmp)
    pctMainFS = pd.concat (pctMainFS, axis = 0, ignore_index = True).sort_values (["feature", "cluster"])
    tmp = pctMainFS.sort_values ("pctMainFS").groupby (["feature", "cluster"]).tail (1)[["feature", "cluster", "mainFS"]]
    pctMainFS = pctMainFS.merge (tmp, on = ["feature", "cluster", "mainFS"], how = "right").sort_values (["feature", "cluster"])
    pltData = pctMainFS.pivot (index = "feature", columns = "cluster", values = "mainFS").rename_axis (None).rename_axis (None, axis = 1)
    labelData = pctMainFS.pivot (index = "feature", columns = "cluster", values = "pctMainFS").rename_axis (None).rename_axis (None, axis = 1)
    palette = sns.color_palette ([colorDict[FS] for FS in allSets], len (allSets))
    fig, ax = plt.subplots (figsize = (8, max (6, len (nameDict["feature"]) / 7.5)))
    sns.heatmap (pltData.astype (float), vmin = 0, vmax = len (allSets), cmap = palette, annot = labelData, fmt = ".2f",
                 linewidth = 0.5, linecolor = "silver", ax = ax)
    ax.set_yticks (ax.get_yticks ()); ax.set_yticklabels (ax.get_yticklabels (), rotation = 0, ha = "right", size = 7)
    ax.set_title (plotTitle, size = 15)
    colorbar = ax.collections[0].colorbar; colorbar.set_ticks (np.arange (len (allSets)) + 0.5)
    colorbar.set_ticklabels (allSets, size = 9)
    fig.tight_layout (); plt.savefig (outputPath); plt.close ()



def heatmap_2aspect (allFV_X, allFV_Y, allSets_X, allSets_Y, nameDict, colorDict, outputPath):
    mainFS_X = pd.DataFrame (allFV_X.argmax (axis = 2), index = nameDict["feature"], columns = nameDict["sample"]).sort_index ()
    mainFV_X = pd.DataFrame (allFV_X.max (axis = 2), index = nameDict["feature"], columns = nameDict["sample"]).loc[mainFS_X.index]
    mainFS_Y = pd.DataFrame (allFV_Y.argmax (axis = 2), index = nameDict["feature"], columns = nameDict["sample"]).sort_index ()
    mainFV_Y = pd.DataFrame (allFV_Y.max (axis = 2), index = nameDict["feature"], columns = nameDict["sample"]).loc[mainFS_Y.index]
    fig, axs = plt.subplots (1, 2, sharex = False, sharey = False, figsize = (18, max (6, np.ceil (len (nameDict["feature"]) / 7.5))))
    palette = sns.color_palette ([colorDict[FS] for FS in allSets_X], len (allSets_X))
    sns.heatmap (mainFS_X.astype (float), vmin = 0, vmax = len (allSets_X), cmap = palette, annot = mainFV_X, fmt = ".2f",
                 linewidth = 0.5, linecolor = "silver", ax = axs[0])
    axs[0].set_yticks (axs[0].get_yticks ()); axs[0].set_yticklabels (axs[0].get_yticklabels (), rotation = 0, ha = "right", size = 8)
    axs[0].set_xlabel (""); axs[0].set_ylabel (""); axs[0].set_title ("DESeq2 log2 fold change", size = 12)
    colorbar = axs[0].collections[0].colorbar; colorbar.set_ticks (np.arange (len (allSets_X)) + 0.5)
    colorbar.set_ticklabels (allSets_X, size = 9)
    palette = sns.color_palette ([colorDict[FS] for FS in allSets_Y], len (allSets_Y))
    sns.heatmap (mainFS_Y.astype (float), vmin = 0, vmax = len (allSets_Y), cmap = palette, annot = mainFV_Y, fmt = ".2f",
                 linewidth = 0.5, linecolor = "silver", ax = axs[1])
    axs[1].set_yticks (axs[1].get_yticks ()); axs[1].set_yticklabels (axs[1].get_yticklabels (), rotation = 0, ha = "right", size = 8)
    axs[1].set_xlabel (""); axs[1].set_ylabel (""); axs[1].set_title ("-log10 (DESeq2 corrected p-value)", size = 12)
    colorbar = axs[1].collections[0].colorbar; colorbar.set_ticks (np.arange (len (allSets_Y)) + 0.5)
    colorbar.set_ticklabels (allSets_Y, size = 9)
    fig.tight_layout (); plt.savefig (outputPath); plt.close ()


