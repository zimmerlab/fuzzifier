import io, os, json, zipfile
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
from jinja2 import Template
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui
from helperFunction import *
from optimizeModes import optimizeGaussian
from fuzzification import fuzzify
from evaluation_item import *
from evaluation_plots import plotConcept, plotCertaintySummary, plotImpurity

sns.set_theme (style = "white", rc = {"axes.facecolor": (0, 0, 0, 0)})


app_ui = ui.page_fluid (
    ui.panel_title (ui.h2 ("1-Dimensional Fuzzifier - Interactive Tool", class_ = "pt-5")),
    ui.accordion (
        ui.accordion_panel (
            "Data Import",
            ui.navset_card_pill (
                ui.nav_panel (
                    "Crisp Value Matrix",
                    ui.row (
                        ui.column (
                            4,
                            ui.card (
                                ui.input_file ("crispMatrix", "Select raw value matrix (.TSV):",
                                               accept = ".tsv", multiple = False, width = "80%"),
                                ui.input_checkbox_group ("specValue", "Select values to label:",
                                                         choices = {"-Inf": "-inf", "+Inf": "+inf", "0": "zero"},
                                                         selected = ("-Inf", "+Inf", "0"), inline = True),
                                ui.input_switch ("addNoise", "Add category for noise?", False),
                                ui.panel_conditional (
                                    "input.addNoise === true",
                                    ui.input_numeric ("minNoiseLevel", "Values no smaller than:",
                                                      min = 0, max = 0, value = 0, step = 0.01),
                                    ui.input_numeric ("maxNoiseLevel", "Values no larger than:",
                                                      min = 0, max = 0, value = 0, step = 0.01)
                                )
                            ),
                            ui.card (
                                ui.input_action_button ("checkInput", "Confirm input and proceed", width = "250px")
                            )
                        ),
                        ui.column (
                            8,
                            ui.navset_card_pill (
                                ui.nav_panel (
                                    "Statistics",
                                    ui.output_data_frame ("summarizeCrispMtx")
                                ),
                                ui.nav_panel (
                                    "Crisp Value Distribution",
                                    ui.layout_columns (
                                        ui.input_slider ("numBins", "Number of bins:", min = 5, max = 100, step = 5,
                                                         value = 50, width = "200px"),
                                        ui.input_slider ("zoom", "Visualize range:", min = 0, max = 0, step = 1,
                                                         value = (0, 0), width = "250px")
                                    ),
                                    ui.div (
                                        ui.output_plot ("crispDistribution", width = "700px", height = "450px"),
                                        style = "display: flex; justify-content: center;"
                                    )
                                )
                            )
                        )
                    )
                ),
                ui.nav_panel (
                    "Metadata",
                    ui.row (
                        ui.column (
                            4,
                            ui.card (
                                ui.input_file ("metadata", "Select metadata (.TSV/.CSV):", accept = [".tsv", ".csv"],
                                               multiple = False, width = "80%"),
                                ui.input_select ("indexCol", "Select column as index:", choices = {"--": "--"}, multiple = False),
                                ui.input_select ("metadataCol", "Select column for clusters:", choices = {"--": "--"}, multiple = False),
                                ui.br (),
                                ui.input_action_button ("checkCluster", "Confirm clustering", width = "250px")
                            )
                        ),
                        ui.column (
                            8,
                            ui.output_data_frame ("showMetadata")
                        )
                    )
                ),
                ui.nav_menu (
                    "Data Overview",
                    ui.nav_panel (
                        "Distrubution per Feature/Sample",
                        ui.layout_columns (
                            ui.div (
                                ui.output_plot ("boxFeature", width = "500px", height = "400px"),
                                style = "display: flex; justify-content: center;"
                            ),
                            ui.div (
                                ui.output_plot ("boxSample", width = "500px", height = "400px"),
                                style = "display: flex; justify-content: center;"
                            )
                        )
                    ),
                    ui.nav_panel (
                        "Distribution per Cluster",
                        ui.div (
                            ui.output_plot ("clusterDistribution", width = "800px", height = "400px"),
                            style = "display: flex; justify-content: center;"
                        )
                    )
                )
            )
        ),
        ui.accordion_panel (
            "Derivation Strategy",
            ui.navset_pill (
                ui.nav_panel (
                    "Fixed Parameters",
                    ui.layout_sidebar (
                        ui.sidebar (
                            ui.card (
                                id = "FS0_fixed"
                            ),
                            width = "300px", position = "left", open = "open"
                        ),
                        ui.card (
                            ui.layout_columns (
                                "Number of fuzzy sets:",
                                ui.input_numeric ("numFS_fixed", "", value = 3, min = 2, max = 10, step = 1),
                                ui.input_action_button ("start_fixed", "Estimate", width = "200px")
                            ),
                            height = "100px"
                        ),
                        ui.card (
                            ui.layout_column_wrap (
                                ui.input_radio_buttons ("typeFS_fixed", "Function type:", choices = {"trap": "trapezoidal", "gauss": "Gaussian"},
                                                        selected = "trap", inline = False),
                                ui.download_button ("download_fixed", "Download concept", width = "200px"),
                                ui.input_action_button ("confirm_fixed", "Fuzzify", width = "200px")
                            ),
                            height = "150px"
                        ),
                        ui.layout_columns (
                            "Select feature for visualization:",
                            ui.input_selectize ("viewFeature_fixed", "", choices = {"ALL": "ALL"}, multiple = False, remove_button = True)
                        ),
                        ui.div (
                            ui.output_plot ("globalDist_fixed", width = "700px", height = "400px"),
                            style = "display: flex; justify-content: center;"
                        ),
                        height = "800px"
                    )
                ),
                ui.nav_panel (
                    "Width per Fuzzy Set",
                    ui.layout_sidebar (
                        ui.sidebar (
                            ui.card (
                                id = "FS0_width"
                            ),
                            width = "300px", position = "left", open = "open"
                        ),
                        ui.card (
                            ui.layout_columns (
                                "Number of fuzzy sets:",
                                ui.input_numeric ("numFS_width", "", value = 3, min = 2, max = 10, step = 1),
                                ui.input_action_button ("start_width", "Estimate", width = "200px")
                            ),
                            height = "100px"
                        ),
                        ui.card (
                            ui.layout_column_wrap (
                                ui.input_radio_buttons ("typeFS_width", "Function type:", choices = {"trap": "trapezoidal", "gauss": "Gaussian"},
                                                        selected = "trap", inline = False),
                                ui.input_radio_buttons ("fuzzyBy_width", "Direction:", choices = {"feature": "per feature", "dataset": "per matrix"},
                                                        selected = "feature", inline = False),
                                ui.download_button ("download_width", "Download concept", width = "200px"),
                                ui.div (),
                                ui.div (),
                                ui.input_action_button ("confirm_width", "Fuzzify", width = "200px"),
                                width = 1 / 3
                            ),
                            height = "150px"
                        ),
                        ui.layout_columns (
                            "Select feature for visualization:",
                            ui.input_selectize ("viewFeature_width", "", choices = {"ALL": "ALL"}, multiple = False, remove_button = True)
                        ),
                        ui.div (
                            ui.output_plot ("globalDist_width", width = "700px", height = "400px"),
                            style = "display: flex; justify-content: center;"
                        ),
                        height = "800px"
                    )
                ),
                ui.nav_panel (
                    "Proportion per Fuzzy Set",
                    ui.layout_sidebar (
                        ui.sidebar (
                            ui.card (
                                id = "FS0_prop"
                            ),
                            width = "300px", position = "left", open = "open"
                        ),
                        ui.card (
                            ui.layout_columns (
                                "Number of fuzzy sets:",
                                ui.input_numeric ("numFS_prop", "", value = 3, min = 2, max = 10, step = 1),
                                ui.input_action_button ("start_prop", "Estimate", width = "200px")
                            ),
                            height = "100px"
                        ),
                        ui.card (
                            ui.layout_column_wrap (
                                ui.input_radio_buttons ("typeFS_prop", "Function type:", choices = {"trap": "trapezoidal", "gauss": "Gaussian"},
                                                        selected = "trap", inline = False),
                                ui.input_radio_buttons ("fuzzyBy_prop", "Direction:", choices = {"feature": "per feature", "dataset": "per matrix"},
                                                        selected = "feature", inline = False),
                                ui.download_button ("download_prop", "Download concept", width = "200px"),
                                ui.div (),
                                ui.div (),
                                ui.input_action_button ("confirm_prop", "Fuzzify", width = "200px"),
                                width = 1 / 3
                            ),
                            height = "150px"
                        ),
                        ui.layout_columns (
                            "Select feature for visualization:",
                            ui.input_selectize ("viewFeature_prop", "", choices = {"ALL": "ALL"}, multiple = False, remove_button = True)
                        ),
                        ui.div (
                            ui.output_plot ("globalDist_prop", width = "700px", height = "400px"),
                            style = "display: flex; justify-content: center;"
                        ),
                        height = "800px"
                    )
                ),
                ui.nav_panel (
                    "Mode Derivation",
                    ui.card (
                        ui.card_header ("Density Maxima Estimation"),
                        ui.card (
                            ui.input_radio_buttons ("defMode", "", selected = "custom", inline = True,
                                                    choices = {"custom": "Customize modes",
                                                               "fit": "Estimate by Boostrapping"})
                        ),
                        ui.panel_conditional (
                            "input.defMode === 'custom'",
                            ui.row (
                                ui.column (
                                    4,
                                    ui.card (
                                        ui.layout_columns (
                                            "Number of curves to fit:",
                                            ui.input_numeric ("numModes_custom", "", value = 3, min = 1, max = 5, step = 1)
                                        ),
                                        ui.input_action_button ("getMode", "Get modes", width = "200px"),
                                        id = "custom0"
                                    )
                                ),
                                ui.column (
                                    6,
                                    ui.div (
                                        ui.output_plot ("globalModes_custom", width = "800px", height = "500px"),
                                        style = "display: flex; justify-content: center;"
                                    )
                                )
                            ),
                            height = "650px"
                        ),
                        ui.panel_conditional (
                            "input.defMode === 'fit'",
                            ui.row (
                                ui.column (
                                    4,
                                    ui.input_numeric ("seed", "Random seed for Bootstrapping:",
                                                      value = 1, min = 1, max = 100, step = 1),
                                    ui.br (),
                                    ui.input_action_button ("randomSeed", "Change seed", width = "250px"),
                                    ui.br (),
                                    ui.br (),
                                    ui.input_numeric ("numValues", "Number of raw values per iteration:",
                                                      min = 100, max = 10000, value = 1000, step = 100),
                                    ui.input_numeric ("numIteration", "Number of iterations:",
                                                      value = 100, min = 100, max = 1000, step = 100),
                                    ui.br (),
                                    ui.input_action_button ("estimate", "Estimate modes", width = "250px")
                                ),
                                ui.column (
                                    6,
                                    ui.div (
                                        ui.output_plot ("globalModes_fit", width = "800px", height = "500px"),
                                        style = "display: flex; justify-content: center;"
                                    )
                                )
                            ),
                            height = "650px"
                        )
                    ),
                    ui.card (
                        ui.input_action_button ("proceedMode", "Proceed with selected modes", width = "500px")
                    ),
                    ui.card (
                        ui.card_header ("Final Fuzzy Concept Derivation"),
                        ui.layout_sidebar (
                            ui.sidebar (
                                ui.card (
                                    ui.input_slider ("pctOverlap", "Percent of slope:", min = 0, max = 1,
                                                     value = 0.5, step = 0.05),
                                    id = "PFC0"
                                ),
                                width = "300px", position = "left", open = "open", heihgt = "1250px"
                            ),
                            ui.card (
                                ui.layout_column_wrap (
                                    ui.download_button ("download_mode", "Download concept", width = "200px"),
                                    ui.input_action_button ("confirm_mode", "Fuzzify", width = "200px")
                                ),
                                height = "100px"
                            ),
                            ui.card (
                                ui.card_header ("Partial Fuzzy Concepts"),
                                ui.div (
                                    ui.output_plot ("partialConcepts", width = "600px", height = "600px"),
                                    style = "display: flex; justify-content: center;"
                                )
                            ),
                            ui.layout_columns (
                                "Select feature for visualization:",
                                ui.input_selectize ("viewFeature_mode", "", choices = {"ALL": "ALL"}, multiple = False, remove_button = True)
                            ),
                            ui.card (
                                ui.card_header ("Merged Fuzzy Concept"),
                                ui.div (
                                    ui.output_plot ("mergedConcept", width = "750px", height = "350px"),
                                    style = "display: flex; justify-content: center;"
                                )
                            )
                        ),
                        height = "1400px"
                    )
                )
            )
        ),
        ui.accordion_panel (
            "Customization and Download Section",
            ui.row (
                ui.column (
                    6,
                    ui.card (
                        id = "rename_PH"
                    )
                ),
                ui.column (
                    1
                ),
                ui.column (
                    4,
                    ui.card (
                        ui.card_header ("Download"),
                        ui.input_select ("downloadDirection", "", selected = "set", multiple = False, width = "200px",
                                         choices = {"feature": "Per feature",
                                                    "sample": "Per sample",
                                                    "set": "Per fuzzy set"}),
                        ui.download_button ("saveFuzzy", "Download results", width = "200px", height = "200px")
                    ),
                    ui.br (),
                    ui.card (
                        ui.card_header ("Evaluation"),
                        ui.input_select ("base", "Select fuzzy set as base level for marker selection:",
                                         choices = {0: "FS0"}, multiple = False),
                        ui.input_slider ("maxSpecific", "Select maximal number of clusters for markers:",
                                         min = 1, max = 1, value = 1, step = 1),
                        ui.input_slider ("minPercent", "Select minimal percentage of samples per main fuzzy set:",
                                         min = 0, max = 1, value = 0.5, step = 0.01),
                        ui.input_select ("sizeCol", "Select parameter for significance of cluster-specificity:",
                                         choices = {"avgFV": "average fuzzy value", "pctMain": "percentage of samples"},
                                         selected = "pctMain", multiple = False),
                        ui.download_button ("saveEvaluation", "Download report", width = "200px")
                    )
                )
            )
        )
    )
)



def server (input, output, session):
    matrix = reactive.value (pd.DataFrame ())
    tempMatrix = reactive.value (pd.DataFrame ())
    itemList = reactive.value ({"feature": list (), "sample": list ()})
    plotRangeGlobal = reactive.value (list ())
    rangeGlobal = reactive.value (list ())
    addNoiseLeft = reactive.value (False)
    addNoiseRight = reactive.value (False)
    labelValues = reactive.value (list ())
    pctWidth = reactive.value (pd.DataFrame (dtype = float))
    pctProp = reactive.value (pd.DataFrame (dtype = float))
    clustering = reactive.value (pd.Series (dtype = str))
    tempConcepts_fixed = reactive.value (dict ())
    tempConcepts_width = reactive.value (dict ())
    tempConcepts_prop = reactive.value (dict ())
    globalMeans = reactive.value (pd.Series (dtype = float))
    globalStds = reactive.value (pd.Series (dtype = float))
    numCards_fixed = reactive.value (0)
    numCards_width = reactive.value (0)
    numCards_prop = reactive.value (0)
    numCards_custom = reactive.value (0)
    numCards_mode = reactive.value (0)
    centerGlobal_custom = reactive.value (list ())
    widthGlobal_custom = reactive.value (list ())
    centerGlobal_fit = reactive.value (list ())
    widthGlobal_fit = reactive.value (list ())
    useModeMethod = reactive.value ("custom")
    proceed = reactive.value (False)
    tempPartialConcepts = reactive.value (list ())
    tempMergedConcept = reactive.value (np.array (list ()))
    allConcepts = reactive.value (dict ())
    globalConcept = reactive.value (np.array (list ()))
    numFuzzySets = reactive.value (0)
    nameFuzzySets = reactive.value (list ())
    fuzzyValues = reactive.value (np.array (list ()))
    idRenameCards = reactive.value (list ())
    conceptInfo = reactive.value ({"method": "", "direction": ""})


    @reactive.effect
    def _ ():
        file = input.crispMatrix ()
        if file is None:
            mtx = pd.DataFrame ()
        else:
            with ui.Progress () as p:
                p.set (message = "Importing Matrix")
                mtx = pd.read_csv (file[0]["datapath"], index_col = 0, sep = "\t").astype (float)
            ui.notification_show ("Import Successful", type = "message", duration = 1.5)
            xMin = np.floor (mtx.replace (-np.inf, np.nan).min (axis = None, skipna = True)) - 1
            xMax = np.ceil (mtx.replace (np.inf, np.nan).max (axis = None, skipna = True)) + 1
            step = estimateStep (xMin, xMax); plotRangeGlobal.set ([xMin, xMax])
            ui.update_numeric ("minNoiseLevel", min = xMin, max = xMax, value = xMin, step = step)
            ui.update_numeric ("maxNoiseLevel", min = xMin, max = xMax, value = xMax, step = step)
            ui.update_numeric ("zoom", min = xMin, max = xMax, value = (xMin, xMax), step = step)
        matrix.set (mtx)
        itemList.set ({"feature": list (mtx.index), "sample": list (mtx.columns)})


    @reactive.effect
    def _ ():
        mtx = matrix.get ()
        if (mtx.empty) or (not input.addNoise ()) or (len (plotRangeGlobal.get ()) != 2):
            tempMatrix.set (pd.DataFrame ())
            return
        if input.addNoise ():
            noiseRepLeft, noiseRepRight = plotRangeGlobal.get ()
            minLevel = input.minNoiseLevel (); minLevel = noiseRepLeft if minLevel is None else minLevel
            maxLevel = input.maxNoiseLevel (); maxLevel = noiseRepRight if maxLevel is None else maxLevel
            if minLevel > noiseRepLeft:
                tempMatrix.set (mtx.mask ((mtx <= minLevel) & np.isfinite (mtx) & (mtx != 0), noiseRepLeft))
                addNoiseLeft.set (True)
            else:
                addNoiseLeft.set (False)
            if maxLevel < noiseRepRight:
                tempMatrix.set (mtx.mask ((mtx >= maxLevel) & np.isfinite (mtx) & (mtx != 0), noiseRepRight))
                addNoiseRight.set (True)
            else:
                addNoiseRight.set (False)
        else:
            addNoiseLeft.set (False); addNoiseRight.set (False)


    @render.data_frame
    def summarizeCrispMtx ():
        if input.addNoise () and (not tempMatrix.get ().empty):
            mtx = tempMatrix.get ()
        else:
            mtx = matrix.get ()
        try:
            noiseRep = plotRangeGlobal.get ()
        except:
            noiseRep = None
        summary = getMtxSummary (mtx, labelValues.get (), noiseRep = noiseRep)
        return render.DataGrid (summary, width = "100%", styles = {"style": {"height": "50px"}})


    @reactive.effect
    def _ ():
        labels = [float (x) for x in input.specValue ()] + [np.nan]
        if matrix.get ().empty:
            return
        if input.addNoise () and (not tempMatrix.get ().empty):
            if addNoiseLeft.get ():
                labels.append (plotRangeGlobal.get ()[0])
            if addNoiseRight.get ():
                labels.append (plotRangeGlobal.get ()[1])
            mtx = tempMatrix.get ().replace (labels + [-np.inf, np.inf], np.nan)
        else:
            mtx = matrix.get ().replace (labels + [-np.inf, np.inf], np.nan)
        xMin = np.floor (mtx.min (axis = None, skipna = True)) - 1
        xMax = np.ceil (mtx.max (axis = None, skipna = True)) + 1
        labelValues.set (labels); rangeGlobal.set ([xMin, xMax])


    @render.plot
    def crispDistribution ():
        visualRange = input.zoom ()
        if matrix.get ().empty or visualRange[0] == visualRange[1]:
            return
        labels = labelValues.get ()
        if input.addNoise () and (not tempMatrix.get ().empty):
            mtx = tempMatrix.get ().replace (labels + [-np.inf, np.inf], np.nan)
        else:
            mtx = matrix.get ().replace (labels + [-np.inf, np.inf], np.nan)
        mtx = mtx.melt ()["value"].dropna ()
        mtx = mtx.loc[(mtx >= visualRange[0]) & (mtx <= visualRange[1])]
        fig, ax = plt.subplots (1, figsize = (15, 6))
        ax.hist (mtx, bins = input.numBins ())
        ax.set_xlim (input.zoom ())
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        return fig


    @reactive.effect
    @reactive.event (input.checkInput)
    def _ ():
        if matrix.get ().empty:
            message = ui.modal ("Please upload one raw matrix (.TSV).", title = "No Crisp Matrix Available",
                                easy_close = True)
            ui.modal_show (message)
        else:
            if input.addNoise ():
                matrix.set (tempMatrix.get ())
            mtx = matrix.get (); tempMatrix.set (pd.DataFrame ()); valueRange = rangeGlobal.get ()
            tmp = mtx.replace (labelValues.get () + [-np.inf, np.inf], np.nan)
            widthTicks = pd.DataFrame ({"min": np.floor (tmp.min (axis = 1, skipna = True)) - 1,
                                        "max": np.ceil (tmp.max (axis = 1, skipna = True)) + 1})
            widthTicks.loc["ALL"] = {"min": valueRange[0], "max": valueRange[1]}
            widthTicks.loc[np.isnan (widthTicks["min"]), "min"] = valueRange[0]
            widthTicks.loc[np.isnan (widthTicks["max"]), "max"] = valueRange[1]
            widthTicks = widthTicks.apply (lambda x: np.linspace (x["min"], x["max"], 1001), axis = 1, result_type = "expand")
            widthTicks = widthTicks.round (3).rename (columns = {widthTicks.columns[i]: i for i in range (1001)})
            propTicks = tmp.quantile (np.linspace (0, 1, 1001), axis = 1, numeric_only = True).T
            propTicks.loc["ALL"] = tmp.melt ()["value"].dropna ().quantile (np.linspace (0, 1, 1001))
            propTicks = propTicks.round (3).rename (columns = {propTicks.columns[i]: i for i in range (1001)})
            pctWidth.set (widthTicks); pctProp.set (propTicks)
            means = tmp.mean (axis = 1, skipna = True); means["ALL"] = tmp.melt ()["value"].mean (skipna = True)
            stds = tmp.std (axis = 1, skipna = True); stds["ALL"] = tmp.melt ()["value"].std (skipna = True)
            globalMeans.set (means.round (3)); globalStds.set (stds.round (3))
            featureList = itemList.get ()["feature"]
            ui.update_selectize ("viewFeature_fixed", choices = ["ALL"] + featureList)
            ui.update_selectize ("viewFeature_width", choices = ["ALL"] + featureList)
            ui.update_selectize ("viewFeature_prop", choices = ["ALL"] + featureList)
            ui.update_selectize ("viewFeature_mode", choices = ["ALL"] + featureList)
            ui.notification_show ("Crisp Value Matrix Done", type = "message", duration = 1.5)


    @render.data_frame
    def showMetadata ():
        file = input.metadata ()
        if file is None:
            metadata = pd.DataFrame ()
        else:
            sep = "," if file[0]["datapath"].split (".")[-1] == "csv" else "\t"
            metadata = pd.read_csv (file[0]["datapath"], index_col = None, sep = sep)
            if metadata.columns[0] == "Unnamed: 0":
                metadata = metadata.rename (columns = {"Unnamed: 0": "index"})
            ui.update_select ("indexCol", choices = list (metadata.columns))
            ui.update_select ("metadataCol", choices = list (metadata.columns))
        return render.DataGrid (metadata, width = "100%", height = "450px", filters = False,
                                editable = False, styles = {"style": {"height": "50px"}})


    @reactive.effect
    def _ ():
        metadata = showMetadata.data_view ()
        indexCol = input.indexCol (); clusterCol = input.metadataCol ()
        if (indexCol == clusterCol) or (indexCol not in metadata.columns) or (clusterCol not in metadata.columns):
            return
        clustering.set (pd.Series (metadata[clusterCol].values, index = metadata[indexCol].values))


    @reactive.effect
    @reactive.event (input.checkCluster)
    def _ ():
        if clustering.get ().empty:
            message = ui.modal ("Please upload metadata (.TSV/.CSV) or select columns for clustering.",
                                title = "No Clustering Available", easy_close = True)
            ui.modal_show (message)
        else:
            ui.update_slider ("maxSpecific", max = len (set (clustering.get ())))
            ui.notification_show ("Clustering Done", type = "message", duration = 1.5)


    @render.plot
    def boxFeature ():
        mtx = matrix.get ()
        if mtx.empty:
            return
        labels = labelValues.get (); mtx = mtx.replace (labels + [-np.inf, np.inf], np.nan)
        ordered = mtx.mean (axis = 1, skipna = True).sort_values ()
        percentiles = mtx.quantile ([0.25, 0.5, 0.75], axis = 1, numeric_only = True)[ordered.index]
        fig, ax = plt.subplots (1, figsize = (5, 8))
        for q in [0.25, 0.5, 0.75]:
            ax.scatter (range (mtx.shape[0]), percentiles.loc[q], s = 3, label = f"{q:.0%}")
        ax2 = ax.twinx (); ax2.plot (ordered, color = "black")
        ax.set_xticks (list ()); ax2.set_yticks (list ()); ax2.set_yticks (list ())
        ax.tick_params (axis = "y", which = "major", labelsize = 8)
        ax.set_xlabel ("sorted by average raw value per feature", size = 10)
        ax.set_ylabel ("quantile", size = 10); ax2.set_ylabel ("average raw value per feature", size = 10)
        ax.legend (loc = "upper left", facecolor = "white"); fig.tight_layout ()
        return fig


    @render.plot
    def boxSample ():
        mtx = matrix.get ()
        if mtx.empty:
            return
        labels = labelValues.get (); mtx = mtx.replace (labels + [-np.inf, np.inf], np.nan)
        ordered = mtx.mean (axis = 0, skipna = True).sort_values ()
        percentiles = mtx.quantile ([0.25, 0.5, 0.75], axis = 0, numeric_only = True)[ordered.index]
        fig, ax = plt.subplots (1, figsize = (5, 8))
        for q in [0.25, 0.5, 0.75]:
            ax.scatter (range (mtx.shape[1]), percentiles.loc[q], s = 3, label = f"{q:.0%}")
        ax2 = ax.twinx (); ax2.plot (ordered, color = "black")
        ax.set_xticks (list ()); ax2.set_yticks (list ()); ax2.set_yticks (list ())
        ax.tick_params (axis = "y", which = "major", labelsize = 8)
        ax.set_xlabel ("sorted by average raw value per sample", size = 10)
        ax.set_ylabel ("quantile", size = 10); ax2.set_ylabel ("average raw value per sample", size = 10)
        ax.legend (loc = "upper left", facecolor = "white"); fig.tight_layout ()
        return fig


    @render.plot
    def clusterDistribution ():
        mtx = matrix.get (); clusters = clustering.get ()
        if mtx.empty or clusters.empty:
            return
        mtx = mtx.replace (labelValues.get () + [-np.inf, np.inf], np.nan)
        allClusters = sorted (set (clusters))
        pltData = pd.DataFrame ({"value": mtx.mean (axis = 0, skipna = True),
                                 "cluster": clusters.loc[mtx.columns]}).sort_values ("cluster")
        fig, ax = plt.subplots (1, figsize = (8, 5))
        sns.violinplot (pltData, x = "cluster", y = "value", ax = ax)
        ax.set_xticks (range (len (allClusters)))
        ax.set_xticklabels (allClusters, rotation = 60, ha = "right")
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel (""); ax.set_ylabel ("average raw value per sample", size = 10)
        fig.tight_layout ()
        return fig


    @reactive.effect
    @reactive.event (input.start_fixed)
    def _ ():
        mtx = matrix.get (); labels = labelValues.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_fixed (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame (mtx.melt ()["value"].replace (labels + [-np.inf, np.inf], np.nan).dropna ()).T
            cutoff = estimateCutoff (dummy, percents).loc["value"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)], 3).T,
                   "gauss": np.round (cutoff.tolist ()[1:-1], 3)}
            tempConcepts_fixed.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix.get (); concepts = tempConcepts_fixed.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]
        num = numCards_fixed.get (); currNum = trap.shape[0]; xMin, xMax = rangeGlobal.get ()
        width = xMax - xMin; step = estimateStep (xMin, xMax)
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_fixed", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"intersection{idx}_fixed", value = trap[i, 0])
            ui.update_numeric (f"slope{idx}_fixed", value = trap[i, 1])
            ui.update_numeric (f"cutoff{idx}_fixed", value = gauss[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.panel_conditional (
                        "input.typeFS_fixed === 'trap'",
                        ui.layout_columns (
                            "Curoff:",
                            ui.input_numeric (f"intersection{idx}_fixed", "", step = step, min = xMin, max = xMax, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Slope:",
                            ui.input_numeric (f"slope{idx}_fixed", "", step = step, min = 0, max = width, value = trap[i, 1])
                        )
                    ),
                    ui.panel_conditional (
                        "input.typeFS_fixed === 'gauss'",
                        ui.layout_columns (
                            "Cutoff:",
                            ui.input_numeric (f"cutoff{idx}_fixed", "", step = step, min = xMin, max = xMax, value = gauss[i])
                        )
                    ),
                    id = f"FS{idx}_fixed"
                ),
                selector = f"#FS{i}_fixed", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_fixed.set (currNum)


    @render.plot
    def globalDist_fixed ():
        mtx = matrix.get (); feature = input.viewFeature_fixed ()
        if mtx.empty or len (plotRangeGlobal.get ()) != 2:
            return
        mtx = mtx.replace (labelValues.get () + [-np.inf, np.inf], np.nan)
        fig, ax = plt.subplots (1, figsize = (8, 5))
        if feature == "ALL":
            pltData = mtx.melt ()["value"]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
            ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
            del pltData
        else:
            try:
                pltData = mtx.loc[feature]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
                ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
                del pltData
            except KeyError:
                pctUnlabelled = "0.0%"
        ax.set_xlim (plotRangeGlobal.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_fixed.get (); valueRange = rangeGlobal.get ()
        if num > 0 and len (valueRange) == 2:
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_ylabel ("fuzzy value", size = 10)
            if input.typeFS_fixed () == "trap":
                try:
                    concept = np.array ([[input[f"intersection{i}_fixed"] (), input[f"slope{i}_fixed"] ()]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "trap", valueRange)
                    ax2.plot (*getLines (concept, colours = list ()))
                except (KeyError, TypeError):
                    pass
            elif input.typeFS_fixed () == "gauss":
                try:
                    concept = np.array ([input[f"cutoff{i}_fixed"] () for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", valueRange)
                    xValues = np.linspace (*valueRange, 1000)
                    lines = getCurves (concept, valueRange, colours = list (), setPlateau = True)
                    for idx in range (num + 1):
                        ax2.plot (xValues, lines[idx][0], color = lines[idx][1])
                    ax2.plot ((valueRange[0], valueRange[0]), (0, 1), lines[0][1])
                    ax2.plot ((valueRange[1], valueRange[1]), (1, 0), lines[-1][1])
                except (KeyError, TypeError):
                    pass
            else:
                raise ValueError
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_fixed_parameters.json")
    def download_fixed ():
        num = numCards_fixed.get (); valueRange = pctWidth.get ()[[0, 1000]]
        globalRange = rangeGlobal.get ()
        if input.typeFS_fixed () == "trap":
            concept = np.array ([[input[f"intersection{i}_fixed"] (), input[f"slope{i}_fixed"] ()]
                                 for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "trap", globalRange)
            concepts = dict ()
            for feature in itemList.get ()["feature"]:
                tmp = concept.copy ()
                left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
                right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
                tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
                concepts[feature] = tmp
        elif input.typeFS_fixed () == "gauss":
            concept = np.array ([input[f"cutoff{i}_fixed"] () for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", globalRange)
            concepts = {feature: concept for feature in itemList.get ()["feature"]}
        else:
            raise ValueError
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
            tmp = {feature: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t]
                             for t in concepts[feature]] for feature in concepts.keys ()}
            outputStr = json.dumps (tmp, indent = 4)
            yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.confirm_fixed)
    def _ ():
        mtx = matrix.get (); labels = labelValues.get (); globalRange = rangeGlobal.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        num = numCards_fixed.get (); valueRange = pctWidth.get ()[[0, 1000]]
        if input.typeFS_fixed () == "trap":
            concept = np.array ([[input[f"intersection{i}_fixed"] (), input[f"slope{i}_fixed"] ()]
                                 for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "trap", globalRange)
            concepts = dict ()
            for feature in itemList.get ()["feature"]:
                tmp = concept.copy ()
                left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
                right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
                tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
                concepts[feature] = tmp
        elif input.typeFS_fixed () == "gauss":
            concept = np.array ([input[f"cutoff{i}_fixed"] () for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", globalRange)
            concepts = {feature: concept for feature in itemList.get ()["feature"]}
        else:
            raise ValueError
        allFV = list (); featureList = itemList.get ()["feature"]; fuzzyValues.set (np.array (list ()))
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts.set (concepts); globalConcept.set (concept)
        numFuzzySets.set (concept.shape[0]); fuzzyValues.set (np.array (allFV))
        noiseRep = plotRangeGlobal.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets.set (tmp); conceptInfo.set ({"method": "fixed", "direction": "dataset"})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.start_width)
    def _ ():
        mtx = matrix.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_width (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame ({"percents": range (101)}).T
            cutoff = estimateCutoff (dummy, percents).loc["percents"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T,
                   "gauss": np.round (cutoff.tolist ()[1:-1])}
            tempConcepts_width.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix.get (); concepts = tempConcepts_width.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]
        num = numCards_width.get (); currNum = trap.shape[0]
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_width", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"intersection{idx}_width", value = trap[i, 0])
            ui.update_numeric (f"slope{idx}_width", value = trap[i, 1])
            ui.update_numeric (f"cutoff{idx}_width", value = gauss[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.panel_conditional (
                        "input.typeFS_width === 'trap'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"intersection{idx}_width", "", step = 0.1, min = 0, max = 100, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Slope (%):",
                            ui.input_numeric (f"slope{idx}_width", "", step = 0.1, min = 0, max = 100, value = trap[i, 1])
                        )
                    ),
                    ui.panel_conditional (
                        "input.typeFS_width === 'gauss'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"cutoff{idx}_width", "", step = 0.1, min = 0, max = 100, value = gauss[i])
                        )
                    ),
                    id = f"FS{idx}_width"
                ),
                selector = f"#FS{i}_width", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_width.set (currNum)


    @render.plot
    def globalDist_width ():
        mtx = matrix.get (); feature = input.viewFeature_width ()
        if mtx.empty or len (plotRangeGlobal.get ()) != 2:
            return
        mtx = mtx.replace (labelValues.get () + [-np.inf, np.inf], np.nan)
        fig, ax = plt.subplots (1, figsize = (8, 5))
        if feature == "ALL":
            pltData = mtx.melt ()["value"]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
            ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
            del pltData
        else:
            try:
                pltData = mtx.loc[feature]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
                ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
                del pltData
            except KeyError:
                pctUnlabelled = "0.0%"
        ax.set_xlim (plotRangeGlobal.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_width.get (); valueRange = rangeGlobal.get (); ticks = pctWidth.get ()
        if num > 0 and len (valueRange) == 2:
            feature = "ALL" if input.fuzzyBy_width () == "dataset" else feature
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            if input.typeFS_width () == "trap":
                try:
                    concept = np.array ([[input[f"intersection{i}_width"] (), input[f"slope{i}_width"] ()]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "trap", [0, 100])
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in params] for params in concept])
                    concept[0, 0] = ticks.loc[feature, 0]; concept[0, 1] = ticks.loc[feature, 0]
                    concept[-1, 2] = ticks.loc[feature, 1000]; concept[-1, 3] = ticks.loc[feature, 1000]
                    ax2.plot (*getLines (concept, colours = list ()))
                except (KeyError, TypeError):
                    pass
            elif input.typeFS_width () == "gauss":
                try:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_width"] ())]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", valueRange)
                    xValues = np.linspace (*valueRange, 1000)
                    lines = getCurves (concept, valueRange, colours = list (), setPlateau = True)
                    for idx in range (num + 1):
                        ax2.plot (xValues, lines[idx][0], color = lines[idx][1])
                    ax2.plot ((valueRange[0], valueRange[0]), (0, 1), lines[0][1])
                    ax2.plot ((valueRange[1], valueRange[1]), (1, 0), lines[-1][1])
                except (KeyError, TypeError):
                    pass
            else:
                raise ValueError
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_width.json")
    def download_width ():
        num = numCards_width.get (); ticks = pctWidth.get ()
        concepts = dict ()
        if input.typeFS_width () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_width"] (), input[f"slope{i}_width"] ()]
                                    for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_width () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), ticks.loc[feature, 0])
                    right = max (np.ceil (concept[-1, 1]), ticks.loc[feature, 1000])
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
            if input.fuzzyBy_width () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
        elif input.typeFS_width () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_width"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            if input.fuzzyBy_width () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_width"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_width () == "dataset":
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
            else:
                raise ValueError
        else:
            raise ValueError
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
            tmp = {feature: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t]
                             for t in concepts[feature]] for feature in concepts.keys ()}
            outputStr = json.dumps (tmp, indent = 4)
            yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.confirm_width)
    def _ ():
        mtx = matrix.get (); labels = labelValues.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        num = numCards_width.get (); ticks = pctWidth.get ()
        concepts = dict ()
        if input.typeFS_width () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_width"] (), input[f"slope{i}_width"] ()]
                                    for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_width () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), ticks.loc[feature, 0])
                    right = max (np.ceil (concept[-1, 1]), ticks.loc[feature, 1000])
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
                globalConcept.set (pctConcept)
            if input.fuzzyBy_width () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
                globalConcept.set (concept)
        elif input.typeFS_width () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_width"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            globalConcept.set (concept)
            if input.fuzzyBy_width () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_width"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_width () == "dataset":
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
            else:
                raise ValueError
        else:
            raise ValueError
        allFV = list (); featureList = itemList.get ()["feature"]; fuzzyValues.set (np.array (list ()))
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts.set (concepts); numFuzzySets.set (concept.shape[0]); fuzzyValues.set (np.array (allFV))
        noiseRep = plotRangeGlobal.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets.set (tmp); conceptInfo.set ({"method": "width", "direction": input.fuzzyBy_width ()})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.start_prop)
    def _ ():
        mtx = matrix.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_prop (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame ({"percents": range (101)}).T
            cutoff = estimateCutoff (dummy, percents).loc["percents"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T,
                   "gauss": np.round (cutoff.tolist ()[1:-1])}
            tempConcepts_prop.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix.get (); concepts = tempConcepts_prop.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]
        num = numCards_prop.get (); currNum = trap.shape[0]
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_prop", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"intersection{idx}_prop", value = trap[i, 0])
            ui.update_numeric (f"slope{idx}_prop", value = trap[i, 1])
            ui.update_numeric (f"cutoff{idx}_prop", value = gauss[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.panel_conditional (
                        "input.typeFS_prop === 'trap'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"intersection{idx}_prop", "", step = 0.1, min = 0, max = 100, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Slope (%):",
                            ui.input_numeric (f"slope{idx}_prop", "", step = 0.1, min = 0, max = 100, value = trap[i, 1])
                        )
                    ),
                    ui.panel_conditional (
                        "input.typeFS_prop === 'gauss'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"cutoff{idx}_prop", "", step = 0.1, min = 0, max = 100, value = gauss[i])
                        )
                    ),
                    id = f"FS{idx}_prop"
                ),
                selector = f"#FS{i}_prop", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_prop.set (currNum)


    @render.plot
    def globalDist_prop ():
        mtx = matrix.get (); feature = input.viewFeature_prop ()
        if mtx.empty or len (plotRangeGlobal.get ()) != 2:
            return
        mtx = mtx.replace (labelValues.get () + [-np.inf, np.inf], np.nan)
        fig, ax = plt.subplots (1, figsize = (8, 5))
        if feature == "ALL":
            pltData = mtx.melt ()["value"]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
            ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
            del pltData
        else:
            try:
                pltData = mtx.loc[feature]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
                ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
                del pltData
            except KeyError:
                pctUnlabelled = "0.0%"
        ax.set_xlim (plotRangeGlobal.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_prop.get (); valueRange = rangeGlobal.get (); ticks = pctProp.get ()
        if num > 0 and len (valueRange) == 2:
            feature = "ALL" if input.fuzzyBy_prop () == "dataset" else feature
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            if input.typeFS_prop () == "trap":
                try:
                    concept = np.array ([[input[f"intersection{i}_prop"] (), input[f"slope{i}_prop"] ()]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "trap", [0, 100])
                    concept = np.array ([[ticks.loc[feature, 10 * i] for i in params] for params in concept])
                    concept[0, 0] = np.floor (ticks.loc[feature, 0]) - 1
                    concept[0, 1] = np.floor (ticks.loc[feature, 0]) - 1
                    concept[-1, 2] = np.ceil (ticks.loc[feature, 1000]) + 1
                    concept[-1, 3] = np.ceil (ticks.loc[feature, 1000]) + 1
                    ax2.plot (*getLines (concept, colours = list ()))
                except (KeyError, TypeError):
                    pass
            elif input.typeFS_prop () == "gauss":
                try:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_prop"] ())]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", valueRange)
                    xValues = np.linspace (*valueRange, 1000)
                    lines = getCurves (concept, valueRange, colours = list (), setPlateau = True)
                    for idx in range (num + 1):
                        ax2.plot (xValues, lines[idx][0], color = lines[idx][1])
                    ax2.plot ((valueRange[0], valueRange[0]), (0, 1), lines[0][1])
                    ax2.plot ((valueRange[1], valueRange[1]), (1, 0), lines[-1][1])
                except (KeyError, TypeError):
                    pass
            else:
                raise ValueError
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_proportion.json")
    def download_prop ():
        num = numCards_prop.get (); ticks = pctProp.get ()
        concepts = dict ()
        if input.typeFS_prop () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_prop"] (), input[f"slope{i}_prop"] ()]
                                    for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_prop () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), np.floor (ticks.loc[feature, 0]) - 1)
                    right = max (np.ceil (concept[-1, 1]), np.ceil (ticks.loc[feature, 1000]) + 1)
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
            if input.fuzzyBy_prop () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
        elif input.typeFS_prop () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_prop"] ())]
                                 for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            if input.fuzzyBy_prop () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_prop"] ())]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_prop () == "dataset":
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
            else:
                raise ValueError
        else:
            raise ValueError
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
            tmp = {feature: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t]
                             for t in concepts[feature]] for feature in concepts.keys ()}
            outputStr = json.dumps (tmp, indent = 4)
            yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.confirm_prop)
    def _ ():
        mtx = matrix.get (); labels = labelValues.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        num = numCards_prop.get (); ticks = pctProp.get ()
        concepts = dict ()
        if input.typeFS_prop () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_prop"] (), input[f"slope{i}_prop"] ()]
                                    for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_prop () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), np.floor (ticks.loc[feature, 0]) - 1)
                    right = max (np.ceil (concept[-1, 1]), np.ceil (ticks.loc[feature, 1000]) + 1)
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
                globalConcept.set (pctConcept)
            if input.fuzzyBy_prop () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
                globalConcept.set (concept)
        elif input.typeFS_prop () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_prop"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            globalConcept.set (concept)
            if input.fuzzyBy_prop () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_prop"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_prop () == "dataset":
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
            else:
                raise ValueError
        else:
            raise ValueError
        allFV = list (); featureList = itemList.get ()["feature"]; fuzzyValues.set (np.array (list ()))
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts.set (concepts); numFuzzySets.set (concept.shape[0]); fuzzyValues.set (np.array (allFV))
        noiseRep = plotRangeGlobal.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets.set (tmp); conceptInfo.set ({"method": "prop", "direction": input.fuzzyBy_prop ()})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.getMode)
    def _ ():
        if len (rangeGlobal.get ()) != 2:
            return
        num = numCards_custom.get (); currNum = input.numModes_custom ()
        xMin, xMax = rangeGlobal.get (); step = estimateStep (xMin, xMax); decimal = int (max (0, -np.log10 (step)))
        modes = list (np.round (np.linspace (xMin, xMax, currNum + 2)[1:-1], decimal))
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (f"#custom{idx}", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"mode{idx}_custom", value = modes[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.layout_columns (
                        "Mode:",
                        ui.input_numeric (f"mode{idx}_custom", "", step = step, min = xMin, max = xMax, value = modes[i])
                    ),
                    id = f"custom{idx}"
                ),
                selector = f"#custom{i}", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_custom.set (currNum)


    @reactive.effect
    def _ ():
        if numCards_custom.get () != input.numModes_custom ():
            return
        elif numCards_custom.get () == 0 and input.numModes_custom () == 0:
            return
        num = numCards_custom.get (); valueRange = rangeGlobal.get ()
        modes = [input[f"mode{i + 1}_custom"] () for i in range (num)]
        width = np.ediff1d ([valueRange[0]] + modes + [valueRange[1]]) / (2 * np.sqrt (2 * np.log (2)))
        width = np.round (np.array ([[width[i], width[i + 1]] for i in range (len (modes))]), 3)
        centerGlobal_custom.set (modes); widthGlobal_custom.set (width)


    @render.plot
    def globalModes_custom ():
        mtx = matrix.get (); labels = labelValues.get (); valueRange = rangeGlobal.get ()
        modes = centerGlobal_custom.get (); width = widthGlobal_custom.get ()
        if mtx.empty or len (modes) == 0 or len (width) == 0 or len (modes) != len (width):
            return
        params = np.array ([[modes[i], width[i, 0], width[i, 1]] for i in range (len (modes))])
        xValues = np.linspace (*valueRange, 1000)
        with np.errstate (divide = "ignore", invalid = "ignore"):
            leftFunc = lambda x, mean, std: (x <= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
            rightFunc = lambda x, mean, std: (x >= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
        fig, ax = plt.subplots (figsize = (8, 5)); ax2 = ax.twinx ()
        ax.hist (mtx.melt ()["value"].replace (labels, np.nan).dropna (), bins = 50)
        for p in params:
            yValues = leftFunc (xValues, p[0], p[1]) + rightFunc (xValues, p[0], p[2])
            ax2.plot (xValues, yValues, color = "red")
            ax2.axvline (p[0], color = "black", linestyle = "dashed")
        ax.set_xlim (plotRangeGlobal.get ())
        ax2.set_xlim (plotRangeGlobal.get ()); ax2.set_ylim ((0, 1.05))
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of values", size = 10)
        ax2.set_ylabel ("fitted fuzzy value", size = 10)
        fig.tight_layout ()
        return fig


    @reactive.effect
    @reactive.event (input.randomSeed)
    def _ ():
        seed = np.random.choice (range (1, 101))
        ui.update_numeric ("seed", value = int (seed))


    @reactive.effect
    @reactive.event (input.estimate)
    def _ ():
        label = labelValues.get (); valueRange = rangeGlobal.get ()
        crisp = matrix.get ().melt ()["value"].replace (label, np.nan).dropna ()
        numValues = input.numValues (); numIteration = input.numIteration ()
        np.random.seed (input.seed ())
        with ui.Progress (min = 0, max = numIteration - 1) as p:
            p.set (message = "Estimating Modes", detail = "This will take a while...")
            allModes = list ()
            furtherParams = {"addIndicator": len (label) != 0, "indicateValue": label}
            for numIter in range (numIteration):
                p.set (numIter, message = "Processing")
                values = pd.Series (np.random.choice (crisp, size = numValues, replace = True),
                                    index = [f"sample{i}" for i in range (numValues)])
                modes = getDensityMaxima (values)
                modes = round (modes.loc[modes["density"] >= 5e-4].drop_duplicates (), 3).to_numpy ()
                try:
                    optFC = optimizeGaussian (values, modes, fuzzyParams = furtherParams,
                                              mergeOverlapFS = True, maxIteration = np.inf)
                    allModes.append (modes.tolist ()); allModes.append (optFC.tolist ())
                except ValueError:
                    numIter -= 1
                    continue
            params = pd.concat ([pd.DataFrame (modes, columns = ["mean", "std"]) for modes in allModes],
                                axis = 0, ignore_index = True)
            modes = getDensityMaxima (params["mean"])
            modes = round (modes.drop_duplicates (), 3)["crisp"].sort_values ().tolist ()
            width = np.ediff1d ([valueRange[0]] + modes + [valueRange[1]]) / (2 * np.sqrt (2 * np.log (2)))
            width = np.round (np.array ([[width[i], width[i + 1]] for i in range (len (modes))]), 3)
            centerGlobal_fit.set (modes); widthGlobal_fit.set (width)
            ui.notification_show ("Estimation Completed", type = "message", duration = 2)


    @render.plot
    @reactive.event (input.estimate)
    def globalModes_fit ():
        mtx = matrix.get (); labels = labelValues.get (); valueRange = rangeGlobal.get ()
        modes = centerGlobal_fit.get (); width = widthGlobal_fit.get ()
        if mtx.empty or len (modes) == 0 or len (width) == 0 or len (modes) != len (width):
            return
        params = np.array ([[modes[i], width[i, 0], width[i, 1]] for i in range (len (modes))])
        xValues = np.linspace (*valueRange, 1000)
        leftFunc = lambda x, mean, std: (x <= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
        rightFunc = lambda x, mean, std: (x >= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
        fig, ax = plt.subplots (figsize = (8, 5)); ax2 = ax.twinx ()
        ax.hist (mtx.melt ()["value"].replace (labels, np.nan).dropna (), bins = 50)
        for p in params:
            yValues = leftFunc (xValues, p[0], p[1]) + rightFunc (xValues, p[0], p[2])
            ax2.plot (xValues, yValues, color = "red")
            ax2.axvline (p[0], color = "black", linestyle = "dashed")
        ax.set_xlim (plotRangeGlobal.get ())
        ax2.set_xlim (plotRangeGlobal.get ()); ax2.set_ylim ((0, 1.05))
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of values", size = 10)
        ax2.set_ylabel ("scaled density", size = 10)
        fig.tight_layout ()
        return fig


    @reactive.effect
    @reactive.event (input.proceedMode)
    def _ ():
        useModeMethod.set (input.defMode ())
        proceed.set (True)


    @reactive.effect
    @reactive.event (input.proceedMode)
    def _ ():
        if useModeMethod.get () == "custom":
            center = centerGlobal_custom.get ()
        elif useModeMethod.get () == "fit":
            center = centerGlobal_fit.get ()
        else:
            raise ValueError
        num = numCards_mode.get ()
        if len (center) == 0:
            return
        for i in range (len (center), num):
            idx = i + 1
            ui.remove_ui (selector = f"#PFC{idx}", multiple = False, immediate = False)
        for i in range (num, len (center)):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Concept {idx}"),
                    ui.input_slider (f"numSet{idx}", "Number of fuzzy sets:", value = 3, min = 0, max = 5, step = 1),
                    ui.input_slider (f"fctWidth{idx}", "Scaling factor for concept width:", value = 1, min = 0, max = 2, step = 0.05),
                    id = f"PFC{idx}"
                ),
                selector = f"#PFC{i}", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_mode.set (len (center))


    @reactive.effect ()
    def _ ():
        if not proceed.get ():
            return
        if useModeMethod.get () == "custom":
            center = centerGlobal_custom.get (); width = widthGlobal_custom.get ()
        elif useModeMethod.get () == "fit":
            center = centerGlobal_fit.get (); width = widthGlobal_fit.get ()
        else:
            raise ValueError
        tempPC = list ()
        if len (center) == 0 or len (width) == 0 or len (center) != len (width):
            return
        for idx in range (len (center)):
            miu = center[idx]; sigma = width[idx]; pct = input.pctOverlap ()
            num = input[f"numSet{idx + 1}"] (); fct = input[f"fctWidth{idx + 1}"] ()
            if num == 0 or fct == 0:
                partialConcept = np.array (list ())
            else:
                if num % 2 == 0:
                    leftCoords = [miu + fct * (k + overlap) * sigma[0] for k in np.linspace (-num, -2, int (num / 2))
                                  for overlap in [-pct, pct]]
                    middleCoords = [miu - fct * pct * min (sigma), miu + fct * pct * min (sigma)]
                    rightCoords = [miu + fct * (k + overlap) * sigma[1] for k in np.linspace (2, num, int (num / 2))
                                   for overlap in [-pct, pct]]
                    coords = leftCoords + middleCoords + rightCoords
                else:
                    leftCoords = [miu + fct * (k + overlap) * sigma[0] for k in np.linspace (-num, -1, int (num / 2 + 1))
                                  for overlap in [-pct, pct]]
                    rightCoords = [miu + fct * (k + overlap) * sigma[1] for k in np.linspace (1, num, int (num / 2 + 1))
                                   for overlap in [-pct, pct]]
                    coords = leftCoords + rightCoords
                partialConcept = np.round ([coords[(2 * i - 2):(2 * i + 2)] for i in range (1, num + 1)], 3)
            tempPC.append (partialConcept.tolist ())
        tempPartialConcepts.set (tempPC)


    @render.plot
    def partialConcepts ():
        if not proceed.get ():
            return
        tempPC = tempPartialConcepts.get (); valueRange = rangeGlobal.get ()
        if len (tempPC) == 0 or len (tempPC) != numCards_mode.get ():
            return
        if useModeMethod.get () == "custom":
            center = centerGlobal_custom.get (); width = widthGlobal_custom.get ()
        elif useModeMethod.get () == "fit":
            center = centerGlobal_fit.get (); width = widthGlobal_fit.get ()
        else:
            raise ValueError
        xValues = np.linspace (*valueRange, 1000)
        leftFunc = lambda x, mean, std: (x <= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
        rightFunc = lambda x, mean, std: (x >= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
        fig, axs = plt.subplots (len (center), sharex = True, sharey = True, figsize = (8, 6))
        for idx in range (len (center)):
            miu = center[idx]; sigma = width[idx]
            yValues = leftFunc (xValues, miu, sigma[0]) + rightFunc (xValues, miu, sigma[1])
            axs[idx].plot (xValues, yValues, color = "black", linestyle = "dashed")
            axs[idx].set_title (f"mean {round (miu, 3)}, std {round (sigma[0], 3)} {round (sigma[1], 3)}", size = 12)
            axs[idx].set_xlim (plotRangeGlobal.get ()); axs[idx].set_ylim ((0, 1.05))
            axs[idx].tick_params (axis = "both", which = "major", labelsize = 8)
            if len (tempPC[idx]) != 0:
                axs[idx].plot (*getLines (np.array (tempPC[idx]), colours = list ()))
        fig.supxlabel ("raw value", size = 10); fig.supylabel ("fuzzy value", size = 10)
        fig.tight_layout ()
        return fig


    @render.plot
    def mergedConcept ():
        if not proceed.get ():
            return
        tempPC = [p for params in tempPartialConcepts.get () for p in params]
        if len (tempPC) == 0:
            return
        mtx = matrix.get ().replace (labelValues.get () + [-np.inf, np.inf], np.nan)
        if mtx.empty or len (rangeGlobal.get ()) != 2:
            return
        xMin, xMax = rangeGlobal.get (); xMin += 1; xMax -= 1
        concept = list (); lastSlope = [xMin, xMin]
        for idx in range (len (tempPC)):
            params = tempPC[idx]
            if len (params) == 0:
                continue
            if params[1] > xMax:
                break
            if params[2] < xMin or params[2] < lastSlope[1]:
                continue
            if params[0] >= lastSlope[1]:
                concept.append (lastSlope + params[:2]); lastSlope = params[:2]
            concept.append (lastSlope + params[2:]); lastSlope = params[2:]
        if concept[-1][3] < xMax:
            concept.append (lastSlope + [xMax, xMax])
        valueRange = rangeGlobal.get ()
        concept[0][0] = valueRange[0]; concept[0][1] = valueRange[0]
        concept[-1][2] = valueRange[1]; concept[-1][3] = valueRange[1]
        concept = np.array (concept); tempMergedConcept.set (concept)
        feature = input.viewFeature_mode ()
        fig, ax = plt.subplots (figsize = (9, 5)); ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05))
        if feature == "ALL":
            pltData = mtx.melt ()["value"]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
            ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
            del pltData
        else:
            try:
                pltData = mtx.loc[feature]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
                ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
                del pltData
            except KeyError:
                pctUnlabelled = "0.0%"
        ax2.plot (*getLines (concept, colours = list ()))
        ax.set_xlim (plotRangeGlobal.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax2.set_xlim (plotRangeGlobal.get ()); ax2.set_ylim ((0, 1.05))
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of values", size = 10)
        ax2.set_ylabel ("fuzzy value", size = 10)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax2.tick_params (axis = "y", which = "major", labelsize = 8)
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_gaussian_mode.json")
    def download_mode ():
        concept = tempMergedConcept.get (); valueRange = pctWidth.get ()[[0, 1000]]
        concepts = dict ()
        for feature in itemList.get ()["feature"]:
            tmp = concept.copy ()
            left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
            right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
            tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
            concepts[feature] = tmp
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
            tmp = {feature: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t]
                             for t in concepts[feature]] for feature in concepts.keys ()}
            outputStr = json.dumps (tmp, indent = 4)
            yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.confirm_mode)
    def _ ():
        mtx = matrix.get (); labels = labelValues.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        concept = tempMergedConcept.get (); valueRange = pctWidth.get ()[[0, 1000]]
        concepts = dict ()
        for feature in itemList.get ()["feature"]:
            tmp = concept.copy ()
            left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
            right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
            tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
            concepts[feature] = tmp
        allFV = list (); featureList = itemList.get ()["feature"]; fuzzyValues.set (np.array (list ()))
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts.set (concepts); globalConcept.set (concept)
        numFuzzySets.set (concept.shape[0]); fuzzyValues.set (np.array (allFV))
        noiseRep = plotRangeGlobal.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets.set (tmp); conceptInfo.set ({"method": "mode", "direction": "dataset"})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        if len (nameFuzzySets.get ()) == len (idRenameCards.get ()):
            return
        allSets = ["PH"] + nameFuzzySets.get (); labels = labelValues.get ()
        allColours = ["#FFFFFF", "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                      "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
                      "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
                      "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"]
        allNames = ["/", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
                    "light:blue", "light:orange", "light:green", "light:red", "light:purple",
                    "light:brown", "light:pink", "light:gray", "light:olive", "light:cyan"]
        colourDict = dict (zip (allColours, allNames))
        allColours = ["#FFFFFF"] * len (labels) + allColours; allNames = ["/"] * len (labels) + allNames
        currCards = idRenameCards.get (); newCards = list ()
        for idx in range (1, len (allSets)):
            prevName = allSets[idx - 1]; prevID = prevName.replace ("-", "_").replace (".", "_")
            currName = allSets[idx]; currID = currName.replace ("-", "_").replace (".", "_")
            if f"rename_{currID}" in currCards:
                ui.update_text (f"new_{currID}", value = currName)
                ui.update_select (f"colour_{currID}", selected = allColours[idx])
            else:
                ui.insert_ui (
                    ui.card (
                        ui.layout_columns (
                            currName,
                            ui.input_text (f"new_{currID}", "", value = currName, spellcheck = False, width = "200px"),
                            "Colour:",
                            ui.input_select (f"colour_{currID}", "", choices = colourDict, selected = allColours[idx], multiple = False)
                        ),
                        id = f"rename_{currID}"
                    ),
                    selector = f"#rename_{prevID}", where = "afterEnd",
                    multiple = False, immediate = False
                )
            newCards.append (f"rename_{currID}")
        for ID in set (currCards) - set (newCards):
            ui.remove_ui (selector = f"#{ID}", multiple = False, immediate = False)
        if currCards != newCards:
            idRenameCards.set (newCards)


    @reactive.effect
    def _ ():
        names = ["FS0"] + [input[f"new_{N.replace ("-", "_").replace (".", "_")}"] ()
                           for N in nameFuzzySets.get () if not N.startswith ("FS0_")]
        ui.update_select ("base", choices = dict (zip (range (len (names)), names)))


    @render.download (filename = "fuzzy_results.zip")
    def saveFuzzy ():
        allFV = fuzzyValues.get (); items = itemList.get (); labels = labelValues.get ()
        concepts = allConcepts.get ()
        if addNoiseLeft.get ():
            labels = ["noiseLeft" if x == plotRangeGlobal.get ()[0] else x for x in labels]
        if addNoiseRight.get ():
            labels = ["noiseRight" if x == plotRangeGlobal.get ()[1] else x for x in labels]
        if input.downloadDirection () == "feature":
            maxNum = allFV.shape[0] + 2
        elif input.downloadDirection () == "sample":
            maxNum = allFV.shape[1] + 2
        else:
            maxNum = allFV.shape[2] + 2
        with ui.Progress (min = 0, max = maxNum) as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            with io.BytesIO () as buf:
                with zipfile.ZipFile (buf, "w") as zf:
                    p.set (0, message = "Downloading")
                    constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
                    with open ("fuzzyConcepts.json", "w", encoding = "utf-8") as f:
                        tmp = {feature: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t]
                                         for t in concepts[feature]] for feature in concepts.keys ()}
                        json.dump (tmp, f, indent = 4)
                    zf.write ("fuzzyConcepts.json"); os.remove ("fuzzyConcepts.json")
                    p.set (1, "Downloading")
                    defaultNames = nameFuzzySets.get ()
                    newNames = [input[f"new_{N.replace ("-", "_").replace (".", "_")}"] () for N in defaultNames]
                    colours = [input[f"colour_{N.replace ("-", "_").replace (".", "_")}"] () for N in defaultNames]
                    summaryDF = pd.DataFrame ({"default name": defaultNames, "new name": newNames, "colour": colours})
                    summaryDF.to_csv ("fuzzy_set_summary.tsv", index = None, sep = "\t")
                    zf.write ("fuzzy_set_summary.tsv"); os.remove ("fuzzy_set_summary.tsv")
                    if input.downloadDirection () == "feature":
                        nameList = items["feature"]
                        for idx in range (allFV.shape[0]):
                            p.set (idx + 2, message = "Downloading"); name = nameList[idx]
                            output = pd.DataFrame (allFV[idx, :, :], index = items["sample"], columns = newNames)
                            output.to_csv (f"fuzzyValues_{name}.tsv", sep = "\t")
                            zf.write (f"fuzzyValues_{name}.tsv"); os.remove (f"fuzzyValues_{name}.tsv")
                    elif input.downloadDirection () == "sample":
                        nameList = items["sample"]
                        for idx in range (allFV.shape[1]):
                            p.set (idx + 1, message = "Downloading"); name = nameList[idx]
                            output = pd.DataFrame (allFV[:, idx, :], index = items["feature"], columns = newNames)
                            output.to_csv (f"fuzzyValues_{name}.tsv", sep = "\t")
                            zf.write (f"fuzzyValues_{name}.tsv"); os.remove (f"fuzzyValues_{name}.tsv")
                    else:
                        for idx in range (allFV.shape[2]):
                            p.set (idx + 1, message = "Downloading"); name = newNames[idx]
                            output = pd.DataFrame (allFV[:, :, idx], index = items["feature"], columns = items["sample"])
                            output.to_csv (f"fuzzyValues_{name}.tsv", sep = "\t")
                            zf.write (f"fuzzyValues_{name}.tsv"); os.remove (f"fuzzyValues_{name}.tsv")
                yield buf.getvalue ()
        ui.notification_show ("Download Completed", type = "message", duration = 2)


    @render.download (filename = "fuzzy_evaluation.zip")
    def saveEvaluation ():
        print (globalConcept.get ())
        items = itemList.get (); labels = labelValues.get ()
        if addNoiseLeft.get ():
            labels = ["noiseLeft" if x == plotRangeGlobal.get ()[0] else x for x in labels]
        if addNoiseRight.get ():
            labels = ["noiseRight" if x == plotRangeGlobal.get ()[1] else x for x in labels]
        concepts = allConcepts.get (); numFS = concepts[items["feature"][0]].shape[0]
        allFV = fuzzyValues.get (); clusters = clustering.get ()
        baseLevel = int (input.base ()); maxNumCluster = input.maxSpecific (); minPctMainFS = input.minPercent ()
        defaultNames = nameFuzzySets.get ()
        newNames = [input[f"new_{N.replace ("-", "_").replace (".", "_")}"] () for N in defaultNames]
        colours = [input[f"colour_{N.replace ("-", "_").replace (".", "_")}"] () for N in defaultNames if not N.startswith ("FS0_")]
        renameDict = dict (zip (defaultNames, newNames))
        if clusters.empty:
            clusters = pd.Series ("TOTAL", index = items["sample"])
            clustering.set (clusters)
        clusters = pd.DataFrame ({"cluster": clusters.values}, index = clusters.index).loc[items["sample"]]
        with ui.Progress (min = 0, max = 9) as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            with io.BytesIO () as buf:
                tmpDir = f"{date.today().isoformat()}-{np.random.randint(100, 999)}"; os.mkdir (tmpDir)
                currentDir = os.getcwd (); os.chdir (tmpDir)
                os.mkdir ("evaluation"); os.mkdir ("./evaluation/markers/")
                with zipfile.ZipFile (buf, "w") as zf:
                    p.set (0, message = "Searching for markers...")
                    markers = getMarkers (allFV, items, numFS, labels, clusters, baseLevel, maxNumCluster, minPctMainFS)
                    markers["mainFS"] = markers["mainFS"].replace (renameDict)
                    markers.to_csv ("./evaluation/marker_statistics.tsv", index = None, sep = "\t")
                    zf.write ("./evaluation/marker_statistics.tsv"); os.remove ("./evaluation/marker_statistics.tsv")
                    p.set (1, message = "Calculating main fuzzy values...")
                    mainFV, mainFS, diffMainFV = getCertaintyStats (allFV, items, numFS, labels)
                    mainFV.to_csv ("./evaluation/main_fuzzy_values.tsv", sep = "\t")
                    zf.write ("./evaluation/main_fuzzy_values.tsv"); os.remove ("./evaluation/main_fuzzy_values.tsv")
                    mainFS = mainFS.replace (renameDict)
                    mainFS.to_csv ("./evaluation/main_fuzzy_sets.tsv", sep = "\t")
                    zf.write ("./evaluation/main_fuzzy_sets.tsv"); os.remove ("./evaluation/main_fuzzy_sets.tsv")
                    diffMainFV.to_csv ("./evaluation/diff_main_fuzzy_values.tsv", sep = "\t")
                    zf.write ("./evaluation/diff_main_fuzzy_values.tsv"); os.remove ("./evaluation/diff_main_fuzzy_values.tsv")
                    p.set (2, message = "Generating Gini impurity...")
                    impurity = getImpurity (allFV, items, clusters, defaultNames)
                    impurity = impurity.rename (columns = renameDict)
                    impurity.to_csv ("./evaluation/gini_impurity.tsv", sep = "\t")
                    zf.write ("./evaluation/gini_impurity.tsv"); os.remove ("./evaluation/gini_impurity.tsv")
                    p.set (3, message = "Visualizing...")
                    concept = globalConcept.get (); info = conceptInfo.get ()
                    method = info["method"]; fuzzyBy = info["direction"]
                    typeFS = input[f"typeFS_{method}"] () if method != "mode" else "trap"
                    if fuzzyBy == "feature" and typeFS == "trap":
                        valueRange = [0, 100]
                    else:
                        valueRange = rangeGlobal.get ()
                    plotConcept (concept, typeFS, info, valueRange, colours = colours, savePlot = True,
                                 savePlotPath = "./evaluation/globalConcept.png")
                    nameLabels = [renameDict[key] for key in defaultNames if key.startswith ("FS0_")]
                    nameSets = [renameDict[key] for key in defaultNames if not key.startswith ("FS0_")]
                    plotCertaintySummary (mainFV, mainFS, diffMainFV, nameLabels, nameSets, savePlot = True,
                                          savePlotPath = "./evaluation/summaryFV.png")
                    plotImpurity (impurity, nameLabels, nameSets, savePlot = True, savePlotPath = "./evaluation/gini_impurity.png")
                    featureList = sorted (set (markers["feature"]))
                    allClusters = sorted (set (clusters["cluster"])); allSets = ["FS0"] + nameSets
                    colours = dict (zip (allSets, ["black"] + colours))
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
                        if input.sizeCol () == "avgFV":
                            sns.scatterplot (pltData, x = "cluster", y = "feature", size = "average fuzzy value",
                                             hue = "main fuzzy set", hue_order = allSets, palette = colours, ax = ax)
                        elif input.sizeCol () == "pctMain":
                            sns.scatterplot (pltData, x = "cluster", y = "feature", size = "percent of samples",
                                             hue = "main fuzzy set", hue_order = allSets, palette = colours, ax = ax)
                        else:
                            raise ValueError
                        ax.set_xticks (range (len (allClusters)))
                        ax.set_xticklabels (allClusters, rotation = 60, ha = "right", size = 7.5)
                        ax.set_yticks (range (len (partialList))); ax.set_yticklabels (partialList, size = 7.5)
                        ax.legend (loc = (1.05, 0.5), facecolor = "white", fontsize = 10)
                        ax.set_xlabel (""); ax.set_ylabel (""); fig.tight_layout ()
                        plt.savefig (f"./evaluation/markers/marker_scatter_{startIdx + 1}.png"); plt.close ()
                    p.set (4, message = "Generating report...")
                    valueRange = rangeGlobal.get ()
                    derivation = {"fixed": "by fixed parameters", "width": "by percent of width in raw value range",
                                  "prop": "by proportion of raw values per fuzzy set", "mode": "by estimated density maxima"}
                    direction = {"feature": "per feature", "dataset": "per matrix"}
                    data = {"numRows": len (items["feature"]),
                            "numCols": len (items["sample"]),
                            "addNoise": input.addNoise (),
                            "cutoffLeft": addNoiseLeft.get (),
                            "cutoffRight": addNoiseRight.get (),
                            "minNoise": input.minNoiseLevel (),
                            "maxNoise": input.maxNoiseLevel (),
                            "labelStr": ", ".join ([str (x) for x in labels]),
                            "crispStats": list ((summarizeCrispMtx.data_view ().iloc[4:, [0, 2]].to_dict (orient = "index").values ())),
                            "pctCompleted": (np.abs (allFV.sum (axis = 2) - 1) <= 1e-3 + 1e-10).mean (axis = None),
                            "conceptStats": [{"statement": "additional fuzzy sets", "value": len (labels)},
                                             {"statement": "fuzzy sets", "value": len (nameSets)},
                                             {"statement": "derivation method", "value": derivation[method]},
                                             {"statement": "fuzzificaion direction", "value": direction[fuzzyBy]}],
                            "conceptPlot": "./evaluation/globalConcept.png",
                            "summaryFV": "./evaluation/summaryFV.png",
                            "pctClear": (impurity[nameSets].max (axis = 1) > 0.5).mean (),
                            "impurityPlot": "./evaluation/gini_impurity.png",
                            "numSpecific": int (markers.shape[0] / len (allClusters)),
                            "baseLevel": nameSets[baseLevel - 1],
                            "maxNum": int (input.maxSpecific ()),
                            "minPct": input.minPercent (),
                            "firstDot": "./evaluation/markers/marker_scatter_1.png"}
                    with open (os.path.join (os.path.dirname (os.path.realpath (__file__)), "template_1dimensional.html"), "r") as f:
                        template = "".join (f.readlines ()); f.close ()
                    content = Template (template).render (**data)
                    with open ("report.html", "w") as f:
                        f.write (content); f.close ()
                    zf.write ("report.html"); os.remove ("report.html")
                    zf.write ("./evaluation/globalConcept.png"); os.remove ("./evaluation/globalConcept.png")
                    zf.write ("./evaluation/summaryFV.png"); os.remove ("./evaluation/summaryFV.png")
                    zf.write ("./evaluation/gini_impurity.png"); os.remove ("./evaluation/gini_impurity.png")
                    for i in range (int (np.ceil (len (featureList) / 50))):
                        zf.write (f"./evaluation/markers/marker_scatter_{50 * i + 1}.png")
                        os.remove (f"./evaluation/markers/marker_scatter_{50 * i + 1}.png")
                os.chdir (currentDir); os.rmdir (f"./{tmpDir}/evaluation/markers/")
                os.rmdir (f"./{tmpDir}/evaluation"); os.rmdir (tmpDir)
                yield buf.getvalue ()
        ui.notification_show ("Download Completed", type = "message", duration = 2, close_button = False)



app = App (app_ui, server)


