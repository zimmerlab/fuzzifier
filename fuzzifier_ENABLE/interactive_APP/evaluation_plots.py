import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helperFunction import getLines, getCurves



### inputs:
# concept: fuzzy concept
# funcType: type of fuzzy concept
# conceptInfo: dictionary of derivation method and direction of fuzzification
# valueRange: expected range of the x-axis
# colours: list of customize colours
# savePlot: whether to save the plot
# savePlotPath: file path if the plot is to be saved
def plotConcept (concept, funcType, conceptInfo, valueRange, colours, savePlot = False, savePlotPath = "fuzzyConcept.png"):
    if conceptInfo["direction"] == "feature" and funcType == "trap":
        if conceptInfo["method"] == "width":
            xLabel = "percent of crisp value range per feature"
        elif conceptInfo["method"] == "prop":
            xLabel = "percent of crisp value distribution per feature"
    else:
        xLabel = "crisp value"
    fig, ax = plt.subplots (1)
    if funcType == "trap":
        ax.plot (*getLines (concept, colours = colours))
    elif funcType == "gauss":
        xValues = np.linspace (*valueRange, 1000)
        lines = getCurves (concept, valueRange, colours = colours, setPlateau = True)
        for idx in range (concept.shape[0] + 1):
            ax.plot (xValues, lines[idx][0], color = lines[idx][1])
    ax.set_xlabel (xLabel); ax.set_ylabel ("fuzzy value"); ax.set_xlim (valueRange); ax.set_ylim ((0, 1.05))
    fig.tight_layout ()
    if savePlot:
        plt.savefig (savePlotPath)
    else:
        plt.show ()
    plt.close ()



### inputs:
# mainFV: pandas dataframe containing main fuzzy values per feature per sample
# mainFS: pandas dataframe containing main fuzzy set per feature per sample
# diffMainFV: pandas dataframe containing differences between main and second main fuzzy values per feature per sample
# nameIndicatorSets: list of labelling fuzzy variables
# nameFuzzySets: list of fuzzy variables (except labelling fuzzy sets)
# savePlot: whether to save the plot
# savePlotPath: file path if the plot is to be saved
def plotCertaintySummary (mainFV, mainFS, diffMainFV, nameIndicatorSets, nameFuzzySets, savePlot = False, savePlotPath = "mainMembershipSummary.png"):
    pltData = pd.DataFrame ({"mainMembership": mainFV.melt ()["value"], "mainFuzzySet": mainFS.melt ()["value"],
                             "difference": diffMainFV.melt ()["value"]})
    pctMainFuzzySets = pltData.value_counts ("mainFuzzySet", normalize = True)
    pctMainFuzzySets = pd.Series ([pctMainFuzzySets.get (idx, 0) for idx in nameIndicatorSets + nameFuzzySets],
                                  index = nameIndicatorSets + nameFuzzySets)
    fig, axs = plt.subplots (3, sharex = False, sharey = False, figsize = (8, 10))
    axs[0].bar (x = pctMainFuzzySets.index, height = pctMainFuzzySets)
    for idx in range (len (pctMainFuzzySets)):
        axs[0].text (idx, pctMainFuzzySets.iloc[idx] + 0.005, "{:.1%}".format (pctMainFuzzySets.iloc[idx]), ha = "center")
    axs[0].set_ylim ((0, max (pctMainFuzzySets) + 0.05))
    pltData = pltData.loc[pltData["mainFuzzySet"].isin (nameFuzzySets)]; totalDim = pltData.shape[0]
    axs[1].hist (pltData["mainMembership"], bins = 10, weights = np.ones (totalDim) / totalDim)
    axs[1].set_xlabel ("main memberships", size = 10)
    axs[2].hist (pltData["difference"], bins = 10, weights = np.ones (totalDim) / totalDim)
    axs[2].set_xlabel ("difference between main and second main membership", size = 10)
    fig.supylabel ("percent of crisp values", size = 10); fig.tight_layout ()
    if savePlot:
        plt.savefig (savePlotPath)
    else:
        plt.show ()
    plt.close ()



### inputs:
# impurity: pandas dataframe of Gini impurity
# nameLabelSets: list of labelling fuzzy variables
# nameSets: list of fuzzy variables (excluding labelling sets)
# savePlot: whether to save the plots
# savePlotPath: file path if the plots are to be saved
def plotImpurity (impurity, nameLabelSets, nameSets, savePlot = False, savePlotPath = "gini_impurity.png"):
    fig, ax = plt.subplots (1, figsize = (8, 6))
    sns.heatmap (impurity, vmin = 0, vmax = 1, ax = ax)
    ax.set_xticks (np.arange (len (nameLabelSets) + len (nameSets)) + 0.5)
    ax.set_xticklabels (nameLabelSets + nameSets, size = 10, rotation = 60, ha = "right")
    ax.set_yticks (list ()); ax.set_ylabel ("feature", size = 10)
    fig.tight_layout ()
    if savePlot:
        plt.savefig (savePlotPath)
    else:
        plt.show ()
    plt.close ()


