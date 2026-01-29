import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui
from helperFunction_1dim import *

sns.set_theme (style = "white", rc = {"axes.facecolor": (0, 0, 0, 0)})


app_ui = ui.page_fluid (
    ui.panel_title (ui.h2 ("1-Dimensional Fuzzifier - Interactive Tool", class_ = "pt-5")),
    ui.accordion (
        ui.accordion_panel (
            "Data Import",
            ui.row (
                ui.column (
                    4,
                    ui.card (
                        ui.input_file ("crispMatrix", "Select raw value matrix (.TSV):", accept = ".tsv", multiple = False, width = "80%"),
                        ui.input_checkbox_group ("specValue", "Select values to label:", choices = {"-Inf": "-inf", "+Inf": "+inf", "0": "zero"},
                                                 selected = ("-Inf", "+Inf", "0"), inline = True),
                        ui.input_switch ("addNoise", "Add category for noise?", False),
                        ui.panel_conditional (
                            "input.addNoise === true",
                            ui.input_numeric ("minNoiseLevel", "Values no smaller than:", min = 0, max = 0, value = 0, step = 0.01),
                            ui.input_numeric ("maxNoiseLevel", "Values no larger than:", min = 0, max = 0, value = 0, step = 0.01)
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
                                ui.input_slider ("numBins", "Number of bins:", min = 5, max = 100, step = 5, value = 50, width = "200px"),
                                ui.input_slider ("zoom", "Visualize range:", min = 0, max = 0, step = 1, value = (0, 0), width = "250px")
                            ),
                            ui.div (
                                ui.output_plot ("crispDistribution", width = "700px", height = "450px"),
                                style = "display: flex; justify-content: center;"
                            )
                        ),
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
                            width = "400px", position = "left", open = "open"
                        ),
                        ui.card (
                            ui.layout_column_wrap (
                                "Number of fuzzy sets:",
                                ui.input_numeric ("numFS_fixed", "", value = 3, min = 2, max = 10, step = 1),
                                ui.input_action_button ("start_fixed", "Estimate", width = "200px"),
                                ui.div (),
                                ui.div (),
                                ui.download_button ("download_fixed", "Download concept", width = "200px"),
                                width = 1 / 3
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
                            width = "400px", position = "left", open = "open"
                        ),
                        ui.card (
                            ui.layout_column_wrap (
                                "Number of fuzzy sets:",
                                ui.input_numeric ("numFS_width", "", value = 3, min = 2, max = 10, step = 1),
                                ui.input_action_button ("start_width", "Estimate", width = "200px"),
                                "Direction:",
                                ui.input_radio_buttons ("fuzzyBy_width", "", choices = {"feature": "per feature", "dataset": "per matrix"},
                                                        selected = "feature", inline = False),
                                ui.download_button ("download_width", "Download concept", width = "200px"),
                                width = 1 / 3
                            ),
                            height = "150px",
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
                            width = "400px", position = "left", open = "open"
                        ),
                        ui.card (
                            ui.layout_column_wrap (
                                "Number of fuzzy sets:",
                                ui.input_numeric ("numFS_prop", "", value = 3, min = 2, max = 10, step = 1),
                                ui.input_action_button ("start_prop", "Estimate", width = "200px"),
                                "Direction:",
                                ui.input_radio_buttons ("fuzzyBy_prop", "", choices = {"feature": "per feature", "dataset": "per matrix"},
                                                        selected = "feature", inline = False),
                                ui.download_button ("download_prop", "Download concept", width = "200px"),
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
                    "Default Fuzzification",
                    ui.layout_sidebar (
                        ui.sidebar (
                            ui.card (
                                ui.layout_columns (
                                    "Band width factor:",
                                    ui.input_numeric ("bwFactor", "", value = 1, min = 0, max = 2, step = 0.05)
                                ),
                                id = "FS0_default"
                            ),
                            width = "400px", position = "left", open = "open"
                        ),
                        ui.card (
                            ui.layout_column_wrap (
                                "Number of fuzzy sets on left/right side:",
                                ui.input_numeric ("numFS_default", "", value = 3, min = 1, max = 3, step = 1),
                                ui.input_action_button ("start_default", "Estimate", width = "200px"),
                                "Direction:",
                                ui.input_radio_buttons ("fuzzyBy_default", "", choices = {"feature": "per feature", "dataset": "per matrix"},
                                                        selected = "feature", inline = False),
                                ui.download_button ("download_default", "Download concept", width = "200px"),
                                width = 1 / 3
                            ),
                            height = "150px"
                        ),
                        ui.layout_columns (
                            "Select feature for visualization:",
                            ui.input_selectize ("viewFeature_default", "", choices = {"ALL": "ALL"}, multiple = False, remove_button = True)
                        ),
                        ui.div (
                            ui.output_plot ("globalDist_default", width = "700px", height = "400px"),
                            style = "display: flex; justify-content: center;"
                        ),
                        height = "800px"
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
    noiseCutoffLeft = reactive.value (-np.inf)
    addNoiseRight = reactive.value (False)
    noiseCutoffRight = reactive.value (np.inf)
    labelValues = reactive.value (list ())
    pctWidth = reactive.value (pd.DataFrame (dtype = float))
    pctProp = reactive.value (pd.DataFrame (dtype = float))
    concepts_fixed = reactive.value (dict ())
    concepts_width = reactive.value (dict ())
    concepts_prop = reactive.value (dict ())
    numCards_fixed = reactive.value (0)
    numCards_width = reactive.value (0)
    numCards_prop = reactive.value (0)
    nameFuzzySets = reactive.value ({"label": list (), "fixed": list (), "width": list (), "prop": list ()})
    idRenameCards = reactive.value (list ())
    defaultColors = reactive.value (["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                                     "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
                                     "blue", "orange", "green", "red", "purple",
                                     "brown", "pink", "gray", "olive", "cyan"])
    defaultColorCodes = reactive.value (["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                                         "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
                                         "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
                                         "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"])


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
                addNoiseLeft.set (True); noiseCutoffLeft.set (minLevel)
            else:
                addNoiseLeft.set (False); noiseCutoffLeft.set (-np.inf)
            if maxLevel < noiseRepRight:
                tempMatrix.set (mtx.mask ((mtx >= maxLevel) & np.isfinite (mtx) & (mtx != 0), noiseRepRight))
                addNoiseRight.set (True); noiseCutoffRight.set (maxLevel)
            else:
                addNoiseRight.set (False); noiseCutoffRight.set (np.inf)
        else:
            addNoiseLeft.set (False); noiseCutoffLeft.set (np.inf)
            addNoiseRight.set (False); noiseCutoffRight.set (np.inf)


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
        labels = [float (x) for x in input.specValue ()]
        if matrix.get ().empty:
            return
        if np.isnan (matrix.get ()).any (axis = None):
            labels.append (np.nan)
        noiseRep = plotRangeGlobal.get (); names = [f"FS0_{val}" for val in labels]
        if input.addNoise () and (not tempMatrix.get ().empty):
            if addNoiseLeft.get ():
                labels.append (noiseRep[0]); names.append ("FS0_MIN-NOISE")
            if addNoiseRight.get ():
                labels.append (noiseRep[1]); names.append ("FS0_MAX-NOISE")
            mtx = tempMatrix.get ().replace (labels + [-np.inf, np.inf], np.nan)
        else:
            mtx = matrix.get ().replace (labels + [-np.inf, np.inf], np.nan)
        xMin = np.floor (mtx.min (axis = None, skipna = True)) - 1
        xMax = np.ceil (mtx.max (axis = None, skipna = True)) + 1
        labelValues.set (labels); rangeGlobal.set ([xMin, xMax])
        nameFS = nameFuzzySets.get (); nameFS["label"] = names; nameFuzzySets.set (nameFS)


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
            mtx = matrix.get ().replace (labelValues.get () + [-np.inf, np.inf], np.nan); tempMatrix.set (pd.DataFrame ())
            widthTicks = pd.DataFrame ({"min": mtx.min (axis = 1, skipna = True), "max": mtx.max (axis = 1, skipna = True)})
            widthTicks.loc["ALL"] = {"min": mtx.min (axis = None, skipna = True), "max": mtx.max (axis = None, skipna = True)}
            widthTicks.loc[np.isnan (widthTicks["min"]), "min"] = widthTicks.loc["ALL", "min"]
            widthTicks.loc[np.isnan (widthTicks["max"]), "max"] = widthTicks.loc["ALL", "max"]
            widthTicks = widthTicks.apply (lambda x: np.linspace (x["min"], x["max"], 1001), axis = 1, result_type = "expand")
            widthTicks = widthTicks.round (3).rename (columns = {widthTicks.columns[i]: i for i in range (1001)})
            propTicks = mtx.quantile (np.linspace (0, 1, 1001), axis = 1, numeric_only = True).T
            propTicks.loc["ALL"] = mtx.melt ()["value"].dropna ().quantile (np.linspace (0, 1, 1001))
            propTicks = propTicks.round (3).rename (columns = {propTicks.columns[i]: i for i in range (1001)})
            pctWidth.set (widthTicks); pctProp.set (propTicks)
            featureList = itemList.get ()["feature"]
            ui.update_selectize ("viewFeature_fixed", choices = ["ALL"] + featureList)
            ui.update_selectize ("viewFeature_width", choices = ["ALL"] + featureList)
            ui.update_selectize ("viewFeature_prop", choices = ["ALL"] + featureList)
            ui.update_selectize ("viewFeature_mode", choices = ["ALL"] + featureList)
            ui.notification_show ("Crisp Value Matrix Done", type = "message", duration = 1.5)


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


    @reactive.effect
    @reactive.event (input.start_fixed)
    def _ ():
        mtx = matrix.get (); labels = labelValues.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_fixed (); percents = [1 / numFS] * numFS; xRange = rangeGlobal.get ()
            dummy = pd.DataFrame (mtx.melt ()["value"].replace (labels + [-np.inf, np.inf], np.nan).dropna ()).T
            cutoff = estimateCutoff (dummy, percents).loc["value"]; slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": getFinalConcept (np.array ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T, "trap", xRange),
                   "gauss": getFinalConcept (np.array (cutoff.tolist ()[1:-1]), "gauss", xRange)}
            concepts_fixed.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix.get (); concepts = concepts_fixed.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]; colorCode = defaultColorCodes.get ()
        colorDict = dict (zip (colorCode, defaultColors.get ()))
        num = numCards_fixed.get (); currNum = trap.shape[0]; xMin, xMax = rangeGlobal.get ()
        width = xMax - xMin; step = estimateStep (xMin, xMax)
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_fixed", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_text (f"name{idx}_fixed", value = f"FS{idx}"),
            ui.update_select (f"typeFS{idx}_fixed", selected = "trap")
            ui.update_numeric (f"coord{idx}_a_fixed", value = trap[i, 0]); ui.update_numeric (f"coord{idx}_b_fixed", value = trap[i, 1])
            ui.update_numeric (f"coord{idx}_c_fixed", value = trap[i, 2]); ui.update_numeric (f"coord{idx}_d_fixed", value = trap[i, 3])
            ui.update_numeric (f"center{idx}_fixed", value = gauss[i, 0]); ui.update_numeric (f"width{idx}_fixed", value = gauss[i, 1])
            ui.update_select (f"color{idx}_fixed", selected = colorCode[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.input_text (f"name{idx}_fixed", "", value = f"FS{idx}"),
                    ui.input_select (f"typeFS{idx}_fixed", "", choices = {"trap": "trapezoidal", "gauss": "Gaussian"}, selected = "trapezoidal",
                                     multiple = False),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_fixed === 'trap'",
                        ui.layout_columns (
                            "Left end:",
                            ui.input_numeric (f"coord{idx}_a_fixed", "", step = step, min = xMin, max = xMax, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Left corner:",
                            ui.input_numeric (f"coord{idx}_b_fixed", "", step = step, min = 0, max = width, value = trap[i, 1])
                        ),
                        ui.layout_columns (
                            "Right corner:",
                            ui.input_numeric (f"coord{idx}_c_fixed", "", step = step, min = xMin, max = xMax, value = trap[i, 2])
                        ),
                        ui.layout_columns (
                            "Right end:",
                            ui.input_numeric (f"coord{idx}_d_fixed", "", step = step, min = 0, max = width, value = trap[i, 3])
                        )
                    ),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_fixed === 'gauss'",
                        ui.layout_columns (
                            "Mu:",
                            ui.input_numeric (f"center{idx}_fixed", "", step = step, min = xMin, max = xMax, value = gauss[i, 0])
                        ),
                        ui.layout_columns (
                            "Sigma:",
                            ui.input_numeric (f"width{idx}_fixed", "", step = step, min = 0, max = width, value = gauss[i, 1])
                        )
                    ),
                    ui.input_select (f"color{idx}_fixed", "", choices = colorDict, selected = colorCode[i], multiple = False),
                    id = f"FS{idx}_fixed"
                ),
                selector = f"#FS{i}_fixed", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_fixed.set (currNum)
        nameFS = nameFuzzySets.get (); nameFS["fixed"] = [f"FS{i}" for i in range (1, currNum + 1)]; nameFuzzySets.set (nameFS)


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
        xDummy = np.linspace (*valueRange, 1000)
        if num > 0 and len (valueRange) == 2:
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_ylabel ("fuzzy value", size = 10)
            for idx in range (1, num + 1):
                if input[f"typeFS{idx}_fixed"] () == "trap":
                    try:
                        a = input[f"coord{idx}_a_fixed"] (); b = input[f"coord{idx}_b_fixed"] ()
                        c = input[f"coord{idx}_c_fixed"] (); d = input[f"coord{idx}_d_fixed"] ()
                        if idx == 1:
                            params = [valueRange[0], valueRange[0], c, d]
                        elif idx == num:
                            params = [a, b, valueRange[1], valueRange[1]]
                        else:
                            params = [a, b, c, d]
                        lines = [(params[0], params[1]), (0, 1), input[f"color{idx}_fixed"] (),
                                 (params[1], params[2]), (1, 1), input[f"color{idx}_fixed"] (),
                                 (params[2], params[3]), (1, 0), input[f"color{idx}_fixed"] ()]
                        ax2.plot (*lines, linewidth = 2)
                    except TypeError:
                        pass
                else:
                    try:
                        mu = input[f"center{idx}_fixed"] (); sigma = input[f"width{idx}_fixed"] ()
                        ax2.plot (xDummy, np.exp (-(xDummy - mu) ** 2 / (2 * sigma ** 2)), color = input[f"color{idx}_fixed"] (), linewidth = 2)
                    except TypeError:
                        pass
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_fixed_parameters.json")
    def download_fixed ():
        num = numCards_fixed.get (); xRange = rangeGlobal.get ()
        constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
        basicInfo = {"number_fuzzy_sets": numCards_fixed.get (), "define_concept_per": "matrix",
                     "label_values": [constRev.get (x, x) if not np.isnan (x) else "NA" for x in set (labelValues.get ()) - set (plotRangeGlobal.get ())],
                     "left_noise_cutoff": constRev.get (noiseCutoffLeft.get (), noiseCutoffLeft.get ()),
                     "right_noise_cutoff": constRev.get (noiseCutoffRight.get (), noiseCutoffRight.get ())}
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            concept = dict ()
            for idx in range (1, num + 1):
                if input[f"typeFS{idx}_fixed"] () == "trap":
                    coords = [input[f"coord{idx}_a_fixed"] (), input[f"coord{idx}_b_fixed"] (),
                              input[f"coord{idx}_c_fixed"] (), input[f"coord{idx}_d_fixed"] ()]
                    if idx == 1:
                        left = min (xRange[0], np.floor (coords[2]) - 1)
                        coords[0] = left; coords[1] = left
                    elif idx == num:
                        right = max (xRange[1], np.ceil (coords[1]) + 1)
                        coords[2] = right; coords[3] = right
                    concept[input[f"name{idx}_fixed"] ()] = [[constRev.get (x, x) if not np.isnan (x) else "NA" for x in coords],
                                                             "trapezoidal", input[f"color{idx}_fixed"] ()]
                else:
                    mu = input[f"center{idx}_fixed"] (); sigma = input[f"width{idx}_fixed"] ()
                    concept[input[f"name{idx}_fixed"] ()] = [[constRev.get (mu, mu) if not np.isnan (mu) else "NA",
                                                              constRev.get (sigma, sigma) if not np.isnan (sigma) else "NA"],
                                                             "Gaussian", input[f"color{idx}_fixed"] ()]
            output = {"basic_information": basicInfo, "fuzzy_concepts": {"ALL": concept}}
            outputStr = json.dumps (output, indent = 4)
            yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)


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
            cutoff = estimateCutoff (dummy, percents).loc["percents"]; slope = cutoff.diff ().iloc[1:].min () / 4
            trap = getFinalConcept (np.array ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T, "trap", [0, 100])
            gauss = getFinalConcept (np.array (cutoff.tolist ()[1:-1]), "gauss", [0, 100])
            concepts_width.set ({"trap": trap, "gauss": gauss[:, 0]})
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix.get (); concepts = concepts_width.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]; colorCode = defaultColorCodes.get ()
        colorDict = dict (zip (colorCode, defaultColors.get ()))
        num = numCards_width.get (); currNum = trap.shape[0]
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_width", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_text (f"name{idx}_width", value = f"FS{idx}"),
            ui.update_select (f"typeFS{idx}_width", selected = "trap")
            ui.update_numeric (f"coord{idx}_a_width", value = trap[i, 0]); ui.update_numeric (f"coord{idx}_b_width", value = trap[i, 1])
            ui.update_numeric (f"coord{idx}_c_width", value = trap[i, 2]); ui.update_numeric (f"coord{idx}_d_width", value = trap[i, 3])
            ui.update_numeric (f"center{idx}_width", value = gauss[i]); ui.update_numeric (f"width{idx}_width", value = 1)
            ui.update_select (f"color{idx}_width", selected = colorCode[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.input_text (f"name{idx}_width", "", value = f"FS{idx}"),
                    ui.input_select (f"typeFS{idx}_width", "", choices = {"trap": "trapezoidal", "gauss": "Gaussian"}, selected = "trapezoidal",
                                     multiple = False),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_width === 'trap'",
                        ui.layout_columns (
                            "Left end (%):",
                            ui.input_numeric (f"coord{idx}_a_width", "", step = 0.1, min = 0, max = 100, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Left corner (%):",
                            ui.input_numeric (f"coord{idx}_b_width", "", step = 0.1, min = 0, max = 100, value = trap[i, 1])
                        ),
                        ui.layout_columns (
                            "Right corner (%)",
                            ui.input_numeric (f"coord{idx}_c_width", "", step = 0.1, min = 0, max = 100, value = trap[i, 2])
                        ),
                        ui.layout_columns (
                            "Right end (%)",
                            ui.input_numeric (f"coord{idx}_d_width", "", step = 0.1, min = 0, max = 100, value = trap[i, 3])
                        )
                    ),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_width === 'gauss'",
                        ui.layout_columns (
                            "Mu (%):",
                            ui.input_numeric (f"center{idx}_width", "", step = 0.1, min = 0, max = 100, value = gauss[i])
                        ),
                        ui.layout_columns (
                            "Sigma scaling factor:",
                            ui.input_numeric (f"width{idx}_width", "", step = 0.1, min = 0, max = 2, value = 1)
                        )
                    ),
                    ui.input_select (f"color{idx}_width", "", choices = colorDict, selected = colorCode[i], multiple = False),
                    id = f"FS{idx}_width"
                ),
                selector = f"#FS{i}_width", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_width.set (currNum)
        nameFS = nameFuzzySets.get (); nameFS["width"] = [f"FS{i}" for i in range (1, currNum + 1)]; nameFuzzySets.set (nameFS)


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
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_width.get (); valueRange = rangeGlobal.get (); ticks = pctWidth.get ()
        xDummy = np.linspace (*valueRange, 1000)
        if num > 0 and len (valueRange) == 2:
            feature = "ALL" if input.fuzzyBy_width () == "dataset" else feature
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            if any ([input[f"typeFS{i}_width"] () == "gauss" for i in range (1, num + 1)]):
                centers = list ()
                for idx in range (1, num + 1):
                    if input[f"typeFS{idx}_width"] () == "trap":
                        if idx == 1:
                            coords = ticks.loc[feature, [0, 0, int (10 * input[f"coord{idx}_c_width"] ()), int (10 * input[f"coord{idx}_d_width"] ())]]
                        elif idx == num:
                            coords = ticks.loc[feature, [int (10 * input[f"coord{idx}_a_width"] ()), int (10 * input[f"coord{idx}_b_width"] ()), 1000, 1000]]
                        else:
                            coords = ticks.loc[feature, [int (10 * input[f"coord{idx}_a_width"] ()), int (10 * input[f"coord{idx}_b_width"] ()),
                                                         int (10 * input[f"coord{idx}_c_width"] ()), int (10 * input[f"coord{idx}_d_width"] ())]]
                        centers.append (coords.mean ())
                    else:
                        centers.append (ticks.loc[feature, int (10 * input[f"center{idx}_width"] ())])
                widths = estimateSigma (centers, valueRange)
            for idx in range (1, num + 1):
                if input[f"typeFS{idx}_width"] () == "trap":
                    try:
                        a = input[f"coord{idx}_a_width"] (); b = input[f"coord{idx}_b_width"] ()
                        c = input[f"coord{idx}_c_width"] (); d = input[f"coord{idx}_d_width"] ()
                        if idx == 1:
                            params = [valueRange[0], valueRange[0], ticks.loc[feature, int (10 * c)], ticks.loc[feature, int (10 * d)]]
                        elif idx == num:
                            params = [ticks.loc[feature, int (10 * a)], ticks.loc[feature, int (10 * b)], valueRange[1], valueRange[1]]
                        else:
                            params = [ticks.loc[feature, int (10 * a)], ticks.loc[feature, int (10 * b)],
                                      ticks.loc[feature, int (10 * c)], ticks.loc[feature, int (10 * d)]]
                        lines = [(params[0], params[1]), (0, 1), input[f"color{idx}_width"] (),
                                 (params[1], params[2]), (1, 1), input[f"color{idx}_width"] (),
                                 (params[2], params[3]), (1, 0), input[f"color{idx}_width"] ()]
                        ax2.plot (*lines, linewidth = 2)
                    except (KeyError, TypeError):
                        pass
                else:
                    try:
                        mu = ticks.loc[feature, int (10 * input[f"center{idx}_width"] ())]; sigma = input[f"width{idx}_width"] () * widths[idx]
                        ax2.plot (xDummy, np.exp (-(xDummy - mu) ** 2 / (2 * sigma ** 2)), color = input[f"color{idx}_width"] (), linewidth = 2)
                    except (KeyError, TypeError):
                        pass
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_width.json")
    def download_width ():
        num = numCards_width.get (); ticks = pctWidth.get (); concepts = dict ()
        if input.fuzzyBy_width () == "feature":
            featureList = itemList.get ()["feature"]
        if input.fuzzyBy_width () == "dataset":
            featureList = ["ALL"]
        names = list (); colors = list (); pctConcept = list (); gaussIdx = list ()
        for idx in range (1, num + 1):
            names.append (input[f"name{idx}_width"] ()); colors.append (input[f"color{idx}_width"] ())
            if input[f"typeFS{idx}_width"] () == "trap":
                pctConcept.append ([int (10 * input[f"coord{idx}_a_width"] ()), int (10 * input[f"coord{idx}_b_width"] ()),
                                    int (10 * input[f"coord{idx}_c_width"] ()), int (10 * input[f"coord{idx}_d_width"] ())])
            else:
                gaussIdx.append (idx)
                pctConcept.append ([10 * input[f"center{idx}_width"] (), input[f"width{idx}_width"] ()])
        constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
        basicInfo = {"number_fuzzy_sets": numCards_width.get (), "define_concept_per": input.fuzzyBy_width (),
                     "label_values": [constRev.get (x, x) if not np.isnan (x) else "NA" for x in set (labelValues.get ()) - set (plotRangeGlobal.get ())],
                     "left_noise_cutoff": constRev.get (noiseCutoffLeft.get (), noiseCutoffLeft.get ()),
                     "right_noise_cutoff": constRev.get (noiseCutoffRight.get (), noiseCutoffRight.get ())}
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            for feature in featureList:
                centers = list (); concept = list ()
                xMin = np.floor (ticks.loc[feature, 0]) - 1; xMax = np.ceil (ticks.loc[feature, 1000]) + 1
                for i in range (num):
                    if input[f"typeFS{i + 1}_width"] () == "trap":
                        coords = ticks.loc[feature, pctConcept[i]].tolist ()
                        if i == 0:
                            coords[0] = xMin; coords[1] = xMin
                        elif i == num - 1:
                            coords[2] = xMax; coords[3] = xMax
                        centers.append (sum (coords) / 4)
                        concept.append ([[constRev.get (x, x) if not np.isnan (x) else "NA" for x in coords], "trapezoidal", colors[i]])
                    else:
                        centers.append (ticks.loc[feature, int (pctConcept[i][0])])
                        concept.append ([[constRev.get (centers[-1], centers[-1]) if not np.isnan (centers[-1]) else "NA", 1],
                                         "Gaussian", colors[i]])
                widths = estimateSigma (centers, [xMin, xMax])
                for idx in gaussIdx:
                    sigma = round (pctConcept[idx - 1][1] * widths[idx], 3)
                    concept[idx - 1][0][1] = constRev.get (sigma, sigma) if not np.isnan (sigma) else "NA"
                concepts[feature] = dict (zip (names, concept))
            outputStr = json.dumps ({"basic_information": basicInfo, "fuzzy_concepts": concepts}, indent = 4)
            yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)


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
            trap = getFinalConcept (np.array ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T, "trap", [0, 100])
            gauss = getFinalConcept (np.array (cutoff.tolist ()[1:-1]), "gauss", [0, 100])
            concepts_prop.set ({"trap": trap, "gauss": gauss[:, 0]})
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix.get (); concepts = concepts_prop.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]; colorCode = defaultColorCodes.get ()
        colorDict = dict (zip (colorCode, defaultColors.get ()))
        num = numCards_prop.get (); currNum = trap.shape[0]
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_prop", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_text (f"name{idx}_prop", value = f"FS{i}")
            ui.update_select (f"typeFS{idx}_prop", selected = "trap")
            ui.update_numeric (f"coord{idx}_a_prop", value = trap[i, 0]); ui.update_numeric (f"coord{idx}_b_prop", value = trap[i, 1])
            ui.update_numeric (f"coord{idx}_c_prop", value = trap[i, 2]); ui.update_numeric (f"coord{idx}_d_prop", value = trap[i, 3])
            ui.update_numeric (f"center{idx}_prop", value = gauss[i]); ui.update_numeric (f"width{idx}_prop", value = 1)
            ui.update_select (f"color{idx}_prop", selected = colorCode[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.input_text (f"name{idx}_prop", "", value = f"FS{idx}"),
                    ui.input_select (f"typeFS{idx}_prop", "", choices = {"trap": "trapezoidal", "gauss": "Gaussian"}, selected = "trap",
                                     multiple = False),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_prop === 'trap'",
                        ui.layout_columns (
                            "Left end (%):",
                            ui.input_numeric (f"coord{idx}_a_prop", "", step = 0.1, min = 0, max = 100, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Left corner (%):",
                            ui.input_numeric (f"coord{idx}_b_prop", "", step = 0.1, min = 0, max = 100, value = trap[i, 1])
                        ),
                        ui.layout_columns (
                            "Right corner (%):",
                            ui.input_numeric (f"coord{idx}_c_prop", "", step = 0.1, min = 0, max = 100, value = trap[i, 2])
                        ),
                        ui.layout_columns (
                            "Right end (%):",
                            ui.input_numeric (f"coord{idx}_d_prop", "", step = 0.1, min = 0, max = 100, value = trap[i, 3])
                        )
                    ),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_prop === 'gauss'",
                        ui.layout_columns (
                            "Mu (%):",
                            ui.input_numeric (f"center{idx}_prop", "", step = 0.1, min = 0, max = 100, value = gauss[i])
                        ),
                        ui.layout_columns (
                            "Sigma scaling factor:",
                            ui.input_numeric (f"width{idx}_prop", "", step = 0.1, min = 0, max = 2, value = 1)
                        )
                    ),
                    ui.input_select (f"color{idx}_prop", "", choices = colorDict, selected = colorCode[i], multiple = False),
                    id = f"FS{idx}_prop"
                ),
                selector = f"#FS{i}_prop", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_prop.set (currNum)
        nameFS = nameFuzzySets.get (); nameFS["prop"] = [f"FS{i}" for i in range (1, currNum + 1)]; nameFuzzySets.set (nameFS)


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
        xDummy = np.linspace (*valueRange, 1000)
        if num > 0 and len (valueRange) == 2:
            feature = "ALL" if input.fuzzyBy_prop () == "dataset" else feature
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            if any ([input[f"typeFS{i}_prop"] () == "gauss" for i in range (1, num + 1)]):
                centers = list ()
                for idx in range (1, num + 1):
                    if input[f"typeFS{idx}_prop"] () == "trap":
                        if idx == 1:
                            coords = ticks.loc[feature, [int (10 * input[f"coord{idx}_c_prop"] ()), int (10 * input[f"coord{idx}_d_prop"] ())]]
                            centers.append ((valueRange[0] + coords.sum ()) / 3)
                        elif idx == num:
                            coords = ticks.loc[feature, [int (10 * input[f"coord{idx}_a_prop"] ()), int (10 * input[f"coord{idx}_b_prop"] ())]]
                            centers.append ((coords.sum () + valueRange[1]) / 3)
                        else:
                            coords = ticks.loc[feature, [int (10 * input[f"coord{idx}_a_prop"] ()), int (10 * input[f"coord{idx}_b_prop"] ()),
                                                         int (10 * input[f"coord{idx}_c_prop"] ()), int (10 * input[f"coord{idx}_d_prop"] ())]]
                            centers.append (coords.mean ())
                    else:
                        centers.append (ticks.loc[feature, int (10 * input[f"center{idx}_prop"] ())])
                widths = estimateSigma (centers, valueRange)
            for idx in range (1, num + 1):
                if input[f"typeFS{idx}_prop"] () == "trap":
                    try:
                        a = input[f"coord{idx}_a_prop"] (); b = input[f"coord{idx}_b_prop"] ()
                        c = input[f"coord{idx}_c_prop"] (); d = input[f"coord{idx}_d_prop"] ()
                        if idx == 1:
                            params = [valueRange[0], valueRange[0], ticks.loc[feature, int (10 * c)], ticks.loc[feature, int (10 * d)]]
                        elif idx == num:
                            params = [ticks.loc[feature, int (10 * a)], ticks.loc[feature, int (10 * b)], valueRange[1], valueRange[1]]
                        else:
                            params = [ticks.loc[feature, int (10 * a)], ticks.loc[feature, int (10 * b)],
                                      ticks.loc[feature, int (10 * c)], ticks.loc[feature, int (10 * d)]]
                        lines = [(params[0], params[1]), (0, 1), input[f"color{idx}_prop"] (),
                                 (params[1], params[2]), (1, 1), input[f"color{idx}_prop"] (),
                                 (params[2], params[3]), (1, 0), input[f"color{idx}_prop"] ()]
                        ax2.plot (*lines, linewidth = 2)
                    except (KeyError, TypeError):
                        pass
                else:
                    try:
                        mu = ticks.loc[feature, int (10 * input[f"center{idx}_prop"] ())]; sigma = input[f"width{idx}_prop"] () * widths[idx]
                        ax2.plot (xDummy, np.exp (-(xDummy - mu) ** 2 / (2 * sigma ** 2)), color = input[f"color{idx}_prop"] (), linewidth = 2)
                    except (KeyError, TypeError):
                        pass
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_proportion.json")
    def download_prop ():
        num = numCards_prop.get (); ticks = pctProp.get (); concepts = dict ()
        if input.fuzzyBy_prop () == "feature":
            featureList = itemList.get ()["feature"]
        if input.fuzzyBy_prop () == "dataset":
            featureList = ["ALL"]
        names = list (); colors = list (); pctConcept = list (); gaussIdx = list ()
        for idx in range (1, num + 1):
            names.append (input[f"name{idx}_prop"] ()); colors.append (input[f"color{idx}_prop"] ())
            if input[f"typeFS{idx}_prop"] () == "trap":
                pctConcept.append ([int (10 * input[f"coord{idx}_a_prop"] ()), int (10 * input[f"coord{idx}_b_prop"] ()),
                                    int (10 * input[f"coord{idx}_c_prop"] ()), int (10 * input[f"coord{idx}_d_prop"] ())])
            else:
                gaussIdx.append (idx)
                pctConcept.append ([10 * input[f"center{idx}_prop"] (), input[f"width{idx}_prop"] ()])
        constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
        basicInfo = {"number_fuzzy_sets": numCards_prop.get (), "define_concept_per": input.fuzzyBy_prop (),
                     "label_values": [constRev.get (x, x) if not np.isnan (x) else "NA" for x in set (labelValues.get ()) - set (plotRangeGlobal.get ())],
                     "left_noise_cutoff": constRev.get (noiseCutoffLeft.get (), noiseCutoffLeft.get ()),
                     "right_noise_cutoff": constRev.get (noiseCutoffRight.get (), noiseCutoffRight.get ())}
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            for feature in featureList:
                centers = list (); concept = list ()
                xMin = np.floor (ticks.loc[feature, 0]) - 1; xMax = np.ceil (ticks.loc[feature, 1000]) + 1
                for i in range (num):
                    if input[f"typeFS{i + 1}_prop"] () == "trap":
                        coords = ticks.loc[feature, pctConcept[i]].tolist ()
                        if i == 0:
                            coords[0] = xMin; coords[1] = xMin
                        elif i == num - 1:
                            coords[2] = xMax; coords[3] = xMax
                        centers.append (sum (coords) / 4)
                        concept.append ([[constRev.get (x, x) if not np.isnan (x) else "NA" for x in coords], "trapezoidal", colors[i]])
                    else:
                        centers.append (ticks.loc[feature, int (pctConcept[i][0])])
                        concept.append ([[constRev.get (centers[-1], centers[-1]) if not np.isnan (centers[-1]) else "NA", 1], "Gaussian", colors[i]])
                widths = estimateSigma (centers, [xMin, xMax])
                for idx in gaussIdx:
                    sigma = round (pctConcept[idx - 1][1] * widths[idx - 1], 3)
                    concept[idx - 1][0][1] = constRev.get (sigma, sigma) if not np.isnan (sigma) else "NA"
                concepts[feature] = dict (zip (names, concept))
            output = {"basic_information": basicInfo, "fuzzy_concepts": concepts}
            outputStr = json.dumps (output, indent = 4)
            yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)



app = App (app_ui, server)


