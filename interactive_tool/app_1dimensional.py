import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shiny import App, reactive, render, ui
from helperFunction_1dim import getMtxSummary, estimateStep, estimateCutoff, getFinalConcept, \
    fitMode, getDefaultConcept, getLines, getPercentage, \
        generateOutputFromConstraint, generateOutputFromDefault

sns.set_theme (style = "white", rc = {"axes.facecolor": (0, 0, 0, 0)})


app_ui = ui.page_fluid (
    ui.panel_title (ui.h2 ("1-Dimensional Fuzzifier - Interactive Tool", class_ = "pt-5")),
    ui.accordion (
        ui.accordion_panel (
            "Initial Fuzzy Concept Definition",
            ui.card (
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
                            ui.layout_columns (
                                ui.input_action_button ("invertMtx", "Invert", width = "150px"),
                                ui.input_action_button ("checkInput", "Confirm and proceed", width = "150px")
                            )
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
                                    ui.input_slider ("numBins", "Number of bins:", min = 0, max = 100, step = 5, value = 50, width = "300px", ticks = True),
                                    ui.input_slider ("zoom", "Visualize range:", min = 0, max = 0, step = 1, value = (0, 0), width = "300px")
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
            ui.card (
                "Fuzzy Concept Derivation",
                ui.navset_pill (
                    ui.nav_panel (
                        "Fuzzy Concepts from Constraints",
                        ui.layout_sidebar (
                            ui.sidebar (
                                ui.card (
                                    ui.div (
                                        ui.input_action_button ("val2pct", "Value to percentile", width = "250px"),
                                        style = "display: flex; justify-content: center;"
                                    ),
                                    id = "FS0_cons"
                                ),
                                width = "450px", position = "left", open = "open"
                            ),
                            ui.card (
                                ui.layout_column_wrap (
                                    "Number of fuzzy sets:",
                                    ui.input_numeric ("numFS_cons", "", value = 3, min = 2, max = 10, step = 1),
                                    ui.input_action_button ("start_cons", "Estimate", width = "200px"),
                                    "Direction:",
                                    ui.input_select ("fuzzyBy_cons", "", selected = "feature", multiple = False,
                                                     choices = {"feature": "per feature", "dataset": "per matrix"}),
                                    ui.div (),
                                    width = 1 / 3
                                ),
                                height = "150px"
                            ),
                            ui.layout_columns (
                                "Select feature for visualization:",
                                ui.input_selectize ("viewFeature_cons", "", choices = {"ALL": "ALL"}, multiple = False, remove_button = True),
                                ui.div ()
                            ),
                            ui.layout_columns (
                                "Number of bins:",
                                ui.input_slider ("numBins_cons", "", min = 0, max = 100, step = 5, value = 50, width = "300px", ticks = True),
                                ui.div ()
                            ),
                            ui.div (
                                ui.output_plot ("globalDist_cons", width = "700px", height = "400px"),
                                style = "display: flex; justify-content: center;"
                            ),
                            height = "900px"
                        )
                    ),
                    ui.nav_panel (
                        "Fuzzy Concepts from Fitting",
                        ui.layout_sidebar (
                            ui.sidebar (
                                ui.card (
                                    ui.layout_columns (
                                        "Number of fuzzy sets on left/right side:",
                                        ui.input_numeric ("numFS_default", "", value = 2, min = 1, max = 3, step = 1)
                                    ),
                                    id = "FS0_default"
                                ),
                                width = "400px", position = "left", open = "open"
                            ),
                            ui.card (
                                ui.layout_column_wrap (
                                    "Band width factor:",
                                    ui.input_numeric ("bwFactor", "", value = 1, min = 0, max = 2, step = 0.05),
                                    ui.input_action_button ("start_default", "Estimate", width = "200px"),
                                    "Direction:",
                                    ui.input_select ("fuzzyBy_default", "", selected = "feature", multiple = False,
                                                     choices = {"feature": "per feature", "dataset": "per matrix"}),
                                    ui.div (),
                                    width = 1 / 3
                                ),
                                height = "150px"
                            ),
                            ui.layout_columns (
                                "Select feature for visualization:",
                                ui.input_selectize ("viewFeature_default", "", choices = {"ALL": "ALL"}, multiple = False, remove_button = True),
                                ui.div ()
                            ),
                            ui.layout_columns (
                                "Number of bins:",
                                ui.input_slider ("numBins_default", "", min = 0, max = 100, step = 5, value = 50, width = "300px", ticks = True),
                                ui.div ()
                            ),
                            ui.div (
                                ui.output_plot ("globalDist_default", width = "700px", height = "400px"),
                                style = "display: flex; justify-content: center;"
                            ),
                            height = "800px"
                        )
                    )
                )
            ),
            ui.card (
                "Download Fuzzy Concepts",
                ui.div (
                    height = "20px"
                ),
                ui.layout_column_wrap (
                    "Download fuzzy concepts defined by:",
                    ui.input_radio_buttons ("downloadOption", "", choices = {"cons": "constraints", "default": "default fuzzification"},
                                            selected = "cons", inline = True),
                    ui.div (),
                    "Default name for matrix-wise fuzzy concept:",
                    ui.input_text ("defaultName", "", value = "ALL", placeholder = "ALL", spellcheck = False),
                    ui.input_action_button ("uniqueName", "Check existence", width = "200px"),
                    width = 1 / 3
                ),
                ui.hr (),
                ui.layout_column_wrap (
                    ui.div (
                        ui.download_button ("downloadConstraints", "Download constraints", width = "250px"),
                        style = "display: flex; justify-content: center;"
                    ),
                    ui.div (
                        ui.download_button ("downloadConcepts", "Download concepts", width = "250px"),
                        style = "display: flex; justify-content: center;"
                    ),
                    width = 1 / 2
                )
            )
        ),
        ui.accordion_panel (
            "Fuzzy Concepts Visualization and Comparison",
            ui.card (
                "Data Import",
                ui.row (
                    ui.column (
                        5,
                        ui.card (
                            ui.input_radio_buttons ("useMatrix", "", choices = {"original": "Use original matrix", "upload": "Upload new matrix"},
                                                    selected = "upload", inline = True),
                            ui.panel_conditional (
                                "input.useMatrix === 'upload'",
                                ui.input_file ("crispMatrix_visual", "Select raw value matrix (.TSV):", accept = ".tsv", multiple = False, width = "80%")
                            ),
                            height = "200px"
                        )
                    ),
                    ui.column (
                        2
                    ),
                    ui.column (
                        5,
                        ui.card (
                            ui.input_radio_buttons ("useConcept", "", choices = {"original": "Use original concepts", "upload": "Upload new concepts"},
                                                    selected = "upload", inline = True),
                            ui.panel_conditional (
                                "input.useConcept === 'original'",
                                ui.input_radio_buttons ("definedBy", "Use fuzzy concepts defined by:", choices = {"cons": "constraints", "default": "default fuzzification"},
                                                        selected = "cons", inline = True)
                            ),
                            ui.panel_conditional (
                                "input.useConcept === 'upload'",
                                ui.input_file ("concept_visual", "Select fuzzy concepts (.JSON):", accept = ".json", multiple = False, width = "80%")
                            ),
                            height = "200px"
                        )
                    )
                ),
                ui.div (
                    ui.input_action_button ("checkInput_visual", "Confirm input and proceed", width = "300px"),
                    style = "display: flex; justify-content: center;"
                ),
                height = "400px"
            ),
            ui.card (
                "Visualization",
                ui.layout_sidebar (
                    ui.sidebar (
                        "Select feature from raw value matrix:",
                        ui.br (),
                        ui.input_selectize ("viewFeature_raw_visual", "", choices = {"ALL": "ALL"}, selected = "ALL", multiple = False, remove_button = True),
                        ui.br (),
                        "Select minimal and maximal value range:",
                        ui.input_slider ("zoom_visual", "", min = 0, max = 0, step = 1, value = (0, 0), width = "300px"),
                        ui.input_slider ("numBins_visual", "Number of bins:", min = 5, max = 100, step = 5, value = 50, width = "300px", ticks = True),
                        "Select feature from fuzzy concepts:",
                        ui.br (),
                        ui.input_selectize ("viewFeature_concept_visual", "", choices = {"--": "--"}, selected = "--", multiple = False, remove_button = True),
                        ui.br (),
                        ui.input_slider ("deviationCutoff", "Maximal deviation:", min = 0, max = 1, step = 0.05, value = 0.2, width = "300px", ticks = True),
                        width = "400px", position = "left", open = "open"
                    ),
                    ui.div (
                        ui.output_plot ("globalDist_visual", width = "700px", height = "400px"),
                        style = "display: flex; justify-content: center;"
                    ),
                    ui.div (
                        ui.output_plot ("deviation_visual", width = "700px", height = "400px"),
                        style = "display: flex; justify-content: center;"
                    )
                ),
                height = "1000px"
            )
        )
    )
)



def server (input, output, session):
    matrix = reactive.value (pd.DataFrame ())
    itemList = reactive.value ({"feature": list (), "sample": list ()})
    plotRangeGlobal = reactive.value (list ())
    rangeGlobal = reactive.value (list ())
    addNoiseLeft = reactive.value (False)
    noiseCutoffLeft = reactive.value (-np.inf)
    addNoiseRight = reactive.value (False)
    noiseCutoffRight = reactive.value (np.inf)
    labelValues = reactive.value (list ())
    pctProp = reactive.value (pd.DataFrame (dtype = float))
    allStd = reactive.value (pd.Series (dtype = float))
    curveFit = reactive.value (pd.DataFrame (dtype = float))
    numCards_cons = reactive.value (0)
    numCards_default = reactive.value (0)
    defaultColors = reactive.value (["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                                     "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
                                     "blue", "orange", "green", "red", "purple",
                                     "brown", "pink", "gray", "olive", "cyan"])
    defaultColorCodes = reactive.value (["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                                         "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
                                         "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
                                         "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"])
    matrix_visual = reactive.value (pd.DataFrame ())
    plotRangeGlobal_visual = reactive.value (list ())
    concepts_visual = reactive.value (dict ())


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
            ui.update_numeric ("zoom", min = xMin + 1, max = xMax - 1, value = (xMin + 1, xMax - 1), step = step)
        matrix.set (mtx)


    @reactive.effect
    def _ ():
        if len (plotRangeGlobal.get ()) != 2:
            addNoiseLeft.set (False); noiseCutoffLeft.set (-np.inf)
            addNoiseRight.set (False); noiseCutoffRight.set (np.inf)
            return
        if input.addNoise ():
            noiseRepLeft, noiseRepRight = plotRangeGlobal.get ()
            minLevel = input.minNoiseLevel (); minLevel = noiseRepLeft if minLevel is None else minLevel
            maxLevel = input.maxNoiseLevel (); maxLevel = noiseRepRight if maxLevel is None else maxLevel
            if minLevel > noiseRepLeft:
                addNoiseLeft.set (True); noiseCutoffLeft.set (minLevel)
            else:
                addNoiseLeft.set (False); noiseCutoffLeft.set (-np.inf)
            if maxLevel < noiseRepRight:
                addNoiseRight.set (True); noiseCutoffRight.set (maxLevel)
            else:
                addNoiseRight.set (False); noiseCutoffRight.set (np.inf)
        else:
            addNoiseLeft.set (False); noiseCutoffLeft.set (-np.inf)
            addNoiseRight.set (False); noiseCutoffRight.set (np.inf)


    @render.data_frame
    def summarizeCrispMtx ():
        mtx = matrix.get (); minLevel = noiseCutoffLeft.get (); maxLevel = noiseCutoffRight.get ()
        labels = labelValues.get (); noiseRep = plotRangeGlobal.get ()
        if len (noiseRep) == 2:
            if addNoiseLeft.get ():
                mtx = mtx.mask (mtx.replace (labels + [-np.inf], np.nan) <= minLevel, noiseRep[0])
            if addNoiseRight.get ():
                mtx = mtx.mask (mtx.replace (labels + [np.inf], np.nan) >= maxLevel, noiseRep[1])
        else:
            noiseRep = None
        summary = getMtxSummary (mtx, labels, noiseRep = noiseRep)
        return render.DataGrid (summary, width = "100%", styles = {"style": {"height": "50px"}})


    @render.plot
    def crispDistribution ():
        visualRange = input.zoom ()
        if matrix.get ().empty or visualRange[0] == visualRange[1]:
            return
        mtx = matrix.get ().replace (labelValues.get () + [-np.inf, np.inf], np.nan).melt ()["value"].dropna ()
        mtx = mtx[(mtx >= visualRange[0]) & (mtx <= visualRange[1])]
        minLevel = noiseCutoffLeft.get (); maxLevel = noiseCutoffRight.get ()
        fig, ax = plt.subplots (1, figsize = (15, 6))
        if input.numBins () != 0:
            ax.hist (mtx, bins = input.numBins ())
        ax.set_xlim (input.zoom ())
        if minLevel > input.zoom ()[0]:
            ax.axvline (minLevel, color = "black", linestyle = "dashed")
        if maxLevel < input.zoom ()[1]:
            ax.axvline (maxLevel, color = "black", linestyle = "dashed")
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of unlabelled values", size = 10)
        return fig
    

    @reactive.effect
    def _ ():
        labels = [float (x) for x in input.specValue ()]
        if matrix.get ().empty:
            return
        if np.isnan (matrix.get ()).any (axis = None):
            labels.append (np.nan)
        labelValues.set (labels)


    @reactive.effect
    @reactive.event (input.invertMtx)
    def _ ():
        if matrix.get ().empty:
            return
        matrix.set (matrix.get ().T)


    @reactive.effect
    @reactive.event (input.checkInput)
    def _ ():
        if matrix.get ().empty:
            message = ui.modal ("Please upload one raw matrix (.TSV).", title = "No Crisp Matrix Available",
                                easy_close = True)
            ui.modal_show (message)
        else:
            mtx = matrix.get ().replace (labelValues.get () + [-np.inf, np.inf], np.nan)
            if addNoiseLeft.get ():
                mtx = mtx.mask (mtx <= noiseCutoffLeft.get ())
            if addNoiseRight.get ():
                mtx = mtx.mask (mtx >= noiseCutoffRight.get ())
            xMin = np.floor (mtx.min (axis = None, skipna = True)) - 1; xMax = np.ceil (mtx.max (axis = None, skipna = True)) + 1
            rangeGlobal.set ([xMin, xMax])
            propTicks = mtx.quantile (np.linspace (0, 1, 1001), axis = 1, numeric_only = True).T
            propTicks.loc["ALL"] = mtx.melt ()["value"].dropna ().quantile (np.linspace (0, 1, 1001))
            propTicks = propTicks.round (3).rename (columns = {propTicks.columns[i]: i for i in range (1001)})
            widths = mtx.std (axis = 1, skipna = True); widths["ALL"] = mtx.melt ()["value"].dropna ().std ()
            pctProp.set (propTicks); allStd.set (widths.round (3))
            featureList = list (mtx.index); itemList.set ({"feature": featureList, "sample": list (mtx.columns)})
            ui.update_selectize ("viewFeature_cons", choices = ["ALL"] + featureList)
            ui.update_selectize ("viewFeature_default", choices = ["ALL"] + featureList)
            ui.notification_show ("Crisp Value Matrix Done", type = "message", duration = 1.5)


    @render.plot
    def boxFeature ():
        mtx = matrix.get ()
        if mtx.empty:
            return
        labels = labelValues.get (); mtx = mtx.replace (labels + [-np.inf, np.inf], np.nan)
        if addNoiseLeft:
            mtx = mtx.mask (mtx <= noiseCutoffLeft.get ())
        if addNoiseRight:
            mtx = mtx.mask (mtx >= noiseCutoffRight.get ())
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
        if addNoiseLeft:
            mtx = mtx.mask (mtx <= noiseCutoffLeft.get ())
        if addNoiseRight:
            mtx = mtx.mask (mtx >= noiseCutoffRight.get ())
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
    @reactive.event (input.start_cons)
    def _ ():
        ticks = pctProp.get ()
        if ticks.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_cons (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame ({"percents": range (101)}).T
            cutoff = estimateCutoff (dummy, percents).loc["percents"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            trap = getFinalConcept (np.array ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T, "trap", [0, 100])
            gauss = getFinalConcept (np.array (cutoff.tolist ()[1:-1]), "gauss", [0, 100])[:, 0]
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)
        colorCode = defaultColorCodes.get (); colorDict = dict (zip (colorCode, defaultColors.get ()))
        num = numCards_cons.get (); currNum = trap.shape[0]; xMin, xMax = rangeGlobal.get ()
        step = estimateStep (xMin, xMax); widths = allStd.get ()
        feature = input.viewFeature_cons (); feature = feature if feature in ticks.index else "ALL"
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_cons", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_text (f"name{idx}_cons", value = f"FS{idx}")
            ui.update_select (f"typeFS{idx}_cons", selected = "trap")
            ui.update_numeric (f"coord{idx}_a_cons", value = trap[i, 0]); ui.update_numeric (f"coord{idx}_b_cons", value = trap[i, 1])
            ui.update_numeric (f"coord{idx}_c_cons", value = trap[i, 2]); ui.update_numeric (f"coord{idx}_d_cons", value = trap[i, 3])
            ui.update_numeric (f"center{idx}_cons", value = gauss[i]); ui.update_numeric (f"width{idx}_cons", value = round (1 if currNum == 0 else 1 / currNum, 2))
            ui.update_select (f"color{idx}_cons", selected = colorCode[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.input_text (f"name{idx}_cons", "", value = f"FS{idx}"),
                    ui.input_select (f"typeFS{idx}_cons", "", choices = {"trap": "trapezoidal", "gauss": "Gaussian"}, selected = "trap",
                                     multiple = False),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_cons === 'trap'",
                        ui.layout_columns (
                            "a",
                            ui.input_numeric (f"coord{idx}_a_cons", "", step = 0.1, min = 0, max = 100, value = trap[i, 0]),
                            "%",
                            "\u2192",
                            ui.input_numeric (f"coord{idx}_a_cons_val", "", step = step, min = xMin, max = xMax,
                                              value = ticks.loc[feature, int (10 * trap[i, 0])]),
                            col_widths = {"sm": (2, 4, 1, 1, 4)}
                        ),
                        ui.layout_columns (
                            "b",
                            ui.input_numeric (f"coord{idx}_b_cons", "", step = 0.1, min = 0, max = 100, value = trap[i, 1]),
                            "%",
                            "\u2192",
                            ui.input_numeric (f"coord{idx}_b_cons_val", "", step = step, min = xMin, max = xMax,
                                              value = ticks.loc[feature, int (10 * trap[i, 1])]),
                            col_widths = {"sm": (2, 4, 1, 1, 4)}
                        ),
                        ui.layout_columns (
                            "c",
                            ui.input_numeric (f"coord{idx}_c_cons", "", step = 0.1, min = 0, max = 100, value = trap[i, 2]),
                            "%",
                            "\u2192",
                            ui.input_numeric (f"coord{idx}_c_cons_val", "", step = step, min = xMin, max = xMax,
                                              value = ticks.loc[feature, int (10 * trap[i, 2])]),
                            col_widths = {"sm": (2, 4, 1, 1, 4)}
                        ),
                        ui.layout_columns (
                            "d",
                            ui.input_numeric (f"coord{idx}_d_cons", "", step = 0.1, min = 0, max = 100, value = trap[i, 3]),
                            "%",
                            "\u2192",
                            ui.input_numeric (f"coord{idx}_d_cons_val", "", step = step, min = xMin, max = xMax,
                                              value = ticks.loc[feature, int (10 * trap[i, 3])]),
                            col_widths = {"sm": (2, 4, 1, 1, 4)}
                        )
                    ),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_cons === 'gauss'",
                        ui.layout_columns (
                            "Mu",
                            ui.input_numeric (f"center{idx}_cons", "", step = 0.1, min = 0, max = 100, value = gauss[i]),
                            "%",
                            "\u2192",
                            ui.input_numeric (f"center{idx}_cons_val", "", step = step, min = xMin, max = xMax,
                                              value = ticks.loc[feature, int (10 * gauss[i])]),
                            col_widths = {"sm": (3, 4, 1, 1, 3)}
                        ),
                        ui.layout_columns (
                            "x*std",
                            ui.input_numeric (f"width{idx}_cons", "", step = 0.1, min = 0, max = 2, value = round (1 if currNum == 0 else 1 / currNum, 2)),
                            "*",
                            ui.input_text (f"std{idx}_cons", "", value = "{:.3f}".format (widths[feature])),
                            col_widths = {"sm": (3, 4, 2, 3)}
                        )
                    ),
                    ui.input_select (f"color{idx}_cons", "", choices = colorDict, selected = colorCode[i], multiple = False),
                    id = f"FS{idx}_cons"
                ),
                selector = f"#FS{i}_cons", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_cons.set (currNum)


    @reactive.effect
    def _ ():
        feature = input.viewFeature_cons () if input.fuzzyBy_cons () == "feature" else "ALL"
        width = allStd.get ().get (feature, 0)
        for idx in range (1, numCards_cons.get ()):
            ui.update_text (f"std{idx}_cons", value = "{:.3f}".format (width))


    @reactive.effect
    def _ ():
        ticks = pctProp.get (); widths = allStd.get ()
        if ticks.empty or widths.empty:
            return
        feature = input.viewFeature_cons () if input.fuzzyBy_cons () == "feature" else "ALL"
        num = numCards_cons.get (); ticks = ticks.loc[feature]
        for idx in range (1, num + 1):
            for pos in ["a", "b", "c", "d"]:
                if idx == 1 and pos in ["a", "b"]:
                    ui.update_numeric (f"coord{idx}_{pos}_cons", value = 0)
                    ui.update_numeric (f"coord{idx}_{pos}_cons_val", value = ticks[0])
                elif idx == num and pos in ["c", "d"]:
                    ui.update_numeric (f"coord{idx}_{pos}_cons", value = 100)
                    ui.update_numeric (f"coord{idx}_{pos}_cons_val", value = ticks[1000])
                else:
                    ui.update_numeric (f"coord{idx}_{pos}_cons_val", value = ticks[int (10 * input[f"coord{idx}_{pos}_cons"] ())])
                    ui.update_text (f"std{idx}_cons", value = "{:.3f}".format (widths[feature]))
            ui.update_numeric (f"center{idx}_cons_val", value = ticks[int (10 * input[f"center{idx}_cons"] ())])


    @reactive.effect
    @reactive.event (input.val2pct)
    def _ ():
        ticks = pctProp.get ()
        if ticks.empty:
            return
        feature = input.viewFeature_cons () if input.fuzzyBy_cons () == "feature" else "ALL"
        num = numCards_cons.get (); ticks = ticks.loc[feature]
        for idx in range (1, num + 1):
            for pos in ["a", "b", "c", "d"]:
                pct = max (0, (ticks <= input[f"coord{idx}_{pos}_cons_val"] ()).sum () - 1) / 10
                ui.update_numeric (f"coord{idx}_{pos}_cons", value = pct)
            pct = max (0, (ticks <= input[f"center{idx}_cons_val"] ()).sum () - 1) / 10
            ui.update_numeric (f"center{idx}_cons", value = max (0, pct))


    @render.plot
    def globalDist_cons ():
        mtx = matrix.get (); xRange = plotRangeGlobal.get (); valueRange = rangeGlobal.get ()
        if mtx.empty or len (xRange) != 2 or len (valueRange) != 2:
            return
        mtx = mtx.replace (labelValues.get () + [-np.inf, np.inf], np.nan)
        minLevel = noiseCutoffLeft.get (); maxLevel = noiseCutoffRight.get ()
        widths = allStd.get (); feature = input.viewFeature_cons ()
        fig, ax = plt.subplots (figsize = (8, 5))
        if feature == "ALL":
            pltData = mtx.melt ()["value"]
            if input.numBins_cons () != 0:
                ax.hist (pltData.dropna (), bins = input.numBins_cons (), color = "lightgray")
            pctUnlabelled = len (pltData.mask ((pltData <= minLevel) | (pltData >= maxLevel)).dropna ()) / len (pltData)
            del pltData
        else:
            try:
                pltData = mtx.loc[feature]
                if input.numBins_cons () != 0:
                    ax.hist (pltData.dropna (), bins = input.numBins_cons (), color = "lightgray")
                pctUnlabelled = len (pltData.mask ((pltData <= minLevel) | (pltData >= maxLevel)).dropna ()) / len (pltData)
                del pltData
            except KeyError:
                pctUnlabelled = 0
        ax.set_xlim (xRange); ax.set_title (f"unlabelled values - {pctUnlabelled:.1%}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        if minLevel > xRange[0]:
            ax.axvline (minLevel, color = "black", linestyle = "dashed")
        if maxLevel < xRange[1]:
            ax.axvline (maxLevel, color = "black", linestyle = "dashed")
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_cons.get (); ticks = pctProp.get ()
        if ticks.empty:
            fig.tight_layout ()
            return fig
        if num > 0 and len (valueRange) == 2:
            feature = "ALL" if input.fuzzyBy_cons () == "dataset" else feature
            ax2 = ax.twinx ()
            ax2.set_xlim (xRange); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            if ticks.empty:
                fig.tight_layout (); return fig
            ticks = ticks.loc[feature]; names = list (); colors = list (); concept = list (); handles = list ()
            for idx in range (1, num + 1):
                names.append (input[f"name{idx}_cons"] ()); colors.append (input[f"color{idx}_cons"] ())
                handles.append (Line2D ([0], [0], color = colors[-1], linewidth = 2))
                if input[f"typeFS{idx}_cons"] () == "trap":
                    a = input[f"coord{idx}_a_cons"] (); b = input[f"coord{idx}_b_cons"] ()
                    c = input[f"coord{idx}_c_cons"] (); d = input[f"coord{idx}_d_cons"] ()
                    try:
                        if idx == 1:
                            params = [valueRange[0], valueRange[0], ticks[int (10 * c)], ticks[int (10 * d)]]
                        elif idx == num:
                            params = [ticks[int (10 * a)], ticks[int (10 * b)], valueRange[1], valueRange[1]]
                        else:
                            params = [ticks[int (10 * a)], ticks[int (10 * b)], ticks[int (10 * c)], ticks[int (10 * d)]]
                        concept.append (params)
                    except (KeyError, TypeError):
                        pass
                else:
                    try:
                        mu = ticks[int (10 * input[f"center{idx}_cons"] ())]; sigma = input[f"width{idx}_cons"] () * widths[feature]
                        concept.append ([mu, sigma])
                    except (KeyError, TypeError):
                        pass
                lines, curves = getLines (concept, [max (minLevel, xRange[0]), min (maxLevel, xRange[1])], colors)
                ax2.plot (*lines, linewidth = 2)
                for curve in curves:
                    ax2.plot (curve[0], curve[1], c = curve[2], linewidth = 2)
            ax2.legend (handles, names, facecolor = "white")
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_constraints.json")
    def download_cons ():
        num = numCards_cons.get ()
        if input.fuzzyBy_cons () == "feature":
            featureList = itemList.get ()["feature"]
        if input.fuzzyBy_cons () == "dataset":
            featureList = ["ALL"]
        typeList = list (); names = list (); colors = list (); pctConcept = list ()
        for idx in range (1, num + 1):
            typeList.append (input[f"typeFS{idx}_cons"] ()); names.append (input[f"name{idx}_cons"] ()); colors.append (input[f"color{idx}_cons"] ())
            if input[f"typeFS{idx}_cons"] () == "trap":
                pctConcept.append ([int (10 * input[f"coord{idx}_a_cons"] ()), int (10 * input[f"coord{idx}_b_cons"] ()),
                                    int (10 * input[f"coord{idx}_c_cons"] ()), int (10 * input[f"coord{idx}_d_cons"] ())])
            else:
                pctConcept.append ([10 * input[f"center{idx}_cons"] (), input[f"width{idx}_cons"] ()])
        constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}; labels = set (labelValues.get ()) - set (plotRangeGlobal.get ())
        basicInfo = {"number_fuzzy_sets": num, "label_values": [constRev.get (x, x) if not np.isnan (x) else "NA" for x in labels]}
        if np.isfinite (noiseCutoffLeft.get ()):
            basicInfo["MIN-NOISE"] = noiseCutoffLeft.get ()
        if np.isfinite (noiseCutoffRight.get ()):
            basicInfo["MAX-NOISE"] = noiseCutoffRight.get ()
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            output = generateOutputFromConstraint (featureList, pctConcept, pctProp.get (), allStd.get (), input.fuzzyBy_cons (), basicInfo,
                                                   typeList, names, colors)
            outputStr = json.dumps (output, indent = 4)
            yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.start_default)
    def _ ():
        mtx = matrix.get ().replace (labelValues.get (), np.nan); bwFct = input.bwFactor ()
        if mtx.empty:
            return
        if addNoiseLeft.get ():
            mtx = mtx.mask (mtx <= noiseCutoffLeft.get ())
        if addNoiseRight.get ():
            mtx = mtx.mask (mtx >= noiseCutoffRight.get ())
        with ui.Progress () as p:
            p.set (message = "Deriving", detail = "This will take a while...")
            fit = pd.DataFrame (columns = ["mu", "sigma"], dtype = float)
            for feature in mtx.index:
                fit.loc[feature] = dict (zip (["mu", "sigma"], fitMode (mtx.loc[feature], bwFct = bwFct, useFit = (bwFct > 0))))
            mtx = mtx.melt ()["value"].dropna ()
            fit.loc["ALL"] = dict (zip (["mu", "sigma"], fitMode (mtx, bwFct = bwFct, useFit = False)))
            curveFit.set (fit.round (3))
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)
    

    @reactive.effect
    def _ ():
        if matrix.get ().empty:
            return
        colorCode = defaultColorCodes.get (); colorDict = dict (zip (colorCode, defaultColors.get ()))
        num = numCards_default.get (); numSide = input.numFS_default (); currNum = 2 * numSide + 1
        trap, gauss = getDefaultConcept (numSide)
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_default", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_text (f"name{idx}_default", value = f"FS{idx}")
            ui.update_select (f"typeFS{idx}_default", choices = {"trap": "trapezoidal", "gauss": "Gaussian"}, selected = "trap")
            ui.update_numeric (f"coord{idx}_a_default", value = trap[i, 0]); ui.update_numeric (f"coord{idx}_b_default", value = trap[i, 1])
            ui.update_numeric (f"coord{idx}_c_default", value = trap[i, 2]); ui.update_numeric (f"coord{idx}_d_default", value = trap[i, 3])
            ui.update_numeric (f"center{idx}_default", step = 0.1, min = -10, max = 10, value = gauss[i])
            ui.update_numeric (f"width{idx}_default", value = 1)
            ui.update_select (f"color{idx}_default", selected = colorCode[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.input_text (f"name{idx}_default", "", value = f"FS{idx}"),
                    ui.input_select (f"typeFS{idx}_default", "", choices = {"trap": "trapezoidal", "gauss": "Gaussian"}, selected = "trap",
                                     multiple = False),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_default === 'trap'",
                        ui.layout_columns (
                            "(a - mu) / sigma:",
                            ui.input_numeric (f"coord{idx}_a_default", "", step = 0.1, min = -10, max = 10, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "(b - mu) / sigma:",
                            ui.input_numeric (f"coord{idx}_b_default", "", step = 0.1, min = -10, max = 10, value = trap[i, 1])
                        ),
                        ui.layout_columns (
                            "(c - mu) / sigma:",
                            ui.input_numeric (f"coord{idx}_c_default", "", step = 0.1, min = -10, max = 10, value = trap[i, 2])
                        ),
                        ui.layout_columns (
                            "(d - mu) / sigma:",
                            ui.input_numeric (f"coord{idx}_d_default", "", step = 0.1, min = -10, max = 10, value = trap[i, 3])
                        )
                    ),
                    ui.panel_conditional (
                        f"input.typeFS{idx}_default === 'gauss'",
                        ui.layout_columns (
                            "(center - mu) / sigma:",
                            ui.input_numeric (f"center{idx}_default", "", step = 0.1, min = -10, max = 10, value = gauss[i])
                        ),
                        ui.layout_columns (
                            "Width scaling factor:",
                            ui.input_numeric (f"width{idx}_default", "", step = 0.1, min = 0, max = 2, value = 1)
                        )
                    ),
                    ui.input_select (f"color{idx}_default", "", choices = colorDict, selected = colorCode[i], multiple = False),
                    id = f"FS{idx}_default"
                ),
                selector = f"#FS{i}_default", where = "afterEnd", multiple = False, immediate = False
            )
        ui.update_select (f"typeFS{numSide + 1}_default", choices = {"gauss": "Gaussian"}, selected = "gauss")
        ui.update_numeric (f"center{numSide + 1}_default", min = gauss[numSide], max = gauss[numSide], step = 0)
        numCards_default.set (currNum)


    @render.plot
    def globalDist_default ():
        mtx = matrix.get (); xRange = plotRangeGlobal.get (); num = numCards_default.get ()
        if mtx.empty or len (xRange) != 2:
            return
        mtx = mtx.replace (labelValues.get () + [-np.inf, np.inf], np.nan)
        minLevel = noiseCutoffLeft.get (); maxLevel = noiseCutoffRight.get ()
        fit = curveFit.get (); feature = input.viewFeature_default ()
        fig, ax = plt.subplots (figsize = (8, 5))
        if feature == "ALL":
            pltData = mtx.melt ()["value"]
            if input.numBins_default () != 0:
                ax.hist (pltData.dropna (), bins = input.numBins_default (), color = "lightgray")
            pctUnlabelled = len (pltData.mask ((pltData <= minLevel) | (pltData >= maxLevel)).dropna ()) / len (pltData)
            del pltData
        else:
            try:
                pltData = mtx.loc[feature]
                if input.numBins_default () != 0:
                    ax.hist (pltData.dropna (), bins = input.numBins_default (), color = "lightgray")
                pctUnlabelled = len (pltData.mask ((pltData <= minLevel) | (pltData >= maxLevel)).dropna ()) / len (pltData)
                del pltData
            except KeyError:
                pctUnlabelled = 0
        ax.set_xlim (xRange); ax.set_title (f"unlabelled values - {pctUnlabelled:.1%}", size = 15)
        if minLevel > xRange[0]:
            ax.axvline (minLevel, color = "black", linestyle = "dashed")
        if maxLevel < xRange[1]:
            ax.axvline (maxLevel, color = "black", linestyle = "dashed")
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of unlabelled values", size = 10)
        if (not fit.empty) and num > 0:
            feature = "ALL" if input.fuzzyBy_default () == "dataset" else feature
            mu, sigma = fit.loc[feature]; names = list (); colors = list (); concept = list (); handles = list ()
            ax2 = ax.twinx (); ax2.set_xlim (xRange); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            for idx in range (1, num + 1):
                names.append (input[f"name{idx}_default"] ()); colors.append (input[f"color{idx}_default"] ())
                handles.append (Line2D ([0], [0], color = colors[-1], linewidth = 2))
                if input[f"typeFS{idx}_default"] () == "trap":
                    try:
                        a = mu + sigma * input[f"coord{idx}_a_default"] (); b = mu + sigma * input[f"coord{idx}_b_default"] ()
                        c = mu + sigma * input[f"coord{idx}_c_default"] (); d = mu + sigma * input[f"coord{idx}_d_default"] ()
                        if idx == 1:
                            xMin = min (xRange[0], np.floor (c) - 1)
                            concept.append ([xMin, xMin, c, d])
                        elif idx == num:
                            xMax = max (xRange[1], np.ceil (b) + 1)
                            concept.append ([a, b, xMax, xMax])
                        else:
                            concept.append ([a, b, c, d])
                    except (TypeError, KeyError):
                        pass
                else:
                    try:
                        center = mu + sigma * input[f"center{idx}_default"] (); width = input[f"width{idx}_default"] () * sigma
                        concept.append ([center, width])
                    except (TypeError, KeyError):
                        pass
            lines, curves = getLines (concept, [max (minLevel, xRange[0]), min (maxLevel, xRange[1])], colors)
            ax2.plot (*lines, linewidth = 2)
            for curve in curves:
                ax2.plot (curve[0], curve[1], c = curve[2], linewidth = 2)
            ax2.legend (handles, names, facecolor = "white")
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_default.json")
    def download_default ():
        num = numCards_default.get (); fit = curveFit.get ()
        if input.fuzzyBy_default () == "feature":
            featureList = itemList.get ()["feature"]
        if input.fuzzyBy_default () == "dataset":
            featureList = ["ALL"]
        typeList = list (); names = list (); colors = list (); zConcept = list ()
        for idx in range (1, num + 1):
            typeList.append (input[f"typeFS{idx}_default"] ()); names.append (input[f"name{idx}_default"] ()); colors.append (input[f"color{idx}_default"] ())
            if input[f"typeFS{idx}_default"] () == "trap":
                zConcept.append ([input[f"coord{idx}_a_default"] (), input[f"coord{idx}_b_default"] (),
                                  input[f"coord{idx}_c_default"] (), input[f"coord{idx}_d_default"] ()])
            else:
                zConcept.append ([input[f"center{idx}_default"] (), input[f"width{idx}_default"] ()])
        constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}; labels = set (labelValues.get ()) - set (plotRangeGlobal.get ())
        basicInfo = {"number_fuzzy_sets": num, "label_values": [constRev.get (x, x) if not np.isnan (x) else "NA" for x in labels]}
        if np.isfinite (noiseCutoffLeft.get ()):
            basicInfo["MIN-NOISE"] = noiseCutoffLeft.get ()
        if np.isfinite (noiseCutoffRight.get ()):
            basicInfo["MAX-NOISE"] = noiseCutoffRight.get ()
        allRanges = matrix.get ().replace (labelValues.get (), np.nan)
        if addNoiseLeft.get ():
            allRanges = allRanges.mask (allRanges <= noiseCutoffLeft.get ())
        if addNoiseRight.get ():
            allRanges = allRanges.mask (allRanges >= noiseCutoffRight.get ())
        allRanges = pd.DataFrame ({"min": allRanges.min (axis = 1, skipna = True), "max": allRanges.max (axis = 1, skipna = True)})
        allRanges = allRanges.replace (np.nan, 0); allRanges.loc["ALL"] = dict (zip (["min", "max"], rangeGlobal.get ()))
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            output = generateOutputFromDefault (featureList, zConcept, fit, allRanges, basicInfo, typeList, names, colors)
            outputStr = json.dumps (output, indent = 4)
            yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.uniqueName)
    def _ ():
        defaultName = input.defaultName () if input.defaultName () != "" else "ALL"
        featureList = set (itemList.get ().get ("feature", list ()))
        if {defaultName}.issubset (featureList):
            message = ui.modal ("The default name already exists as index in the input matrix.", title = "Check Failed",
                                easy_close = True)
            ui.modal_show (message)
        elif len (featureList) != 0:
            ui.notification_show ("The default name is unique.", type = "message", duration = 2)


    @render.download (filename = "concept_constraint.json")
    def downloadConstraints ():
        if numCards_cons.get () == 0 and numCards_default.get () == 0:
            return
        constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
        option = input.downloadOption (); labels = set (labelValues.get ()) - set (plotRangeGlobal.get ())
        if option == "cons":
            num = numCards_cons.get ()
            content = {"derivation_method": "percentiles", "number_fuzzy_sets": num,
                       "label_values": [constRev.get (x, x) if not np.isnan (x) else "NA" for x in labels]}
        elif option == "default":
            num = numCards_default.get ()
            content = {"derivation_method": "default", "number_fuzzy_sets": num,
                       "label_values": [constRev.get (x, x) if not np.isnan (x) else "NA" for x in labels]}
        else:
            raise ValueError
        for idx in range (1, num + 1):
            typeFS = input[f"typeFS{idx}_{option}"] (); color = input[f"color{idx}_{option}"] ()
            if typeFS == "trap":
                params = [input[f"coord{idx}_a_{option}"] (), input[f"coord{idx}_b_{option}"] (),
                          input[f"coord{idx}_c_{option}"] (), input[f"coord{idx}_d_{option}"] ()]
            else:
                params = [input[f"center{idx}_{option}"] (), input[f"width{idx}_{option}"] ()]
            content[input[f"name{idx}_{option}"] ()] = [params, typeFS, color]
        print (content)
        outputStr = json.dumps (content, indent = 4)
        yield outputStr
        ui.notification_show ("Download Completed", type = "message", duration = 2)


    @render.download (filename = "concept_detailed.json")
    def downloadConcepts ():
        if numCards_cons.get () == 0 and numCards_default.get () == 0:
            return
        option = input.downloadOption (); defaultName = input.defaultName () if input.defaultName () != "" else "ALL"
        featureList = [defaultName] + itemList.get ()["feature"]
        constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}; labels = set (labelValues.get ()) - set (plotRangeGlobal.get ())
        if option == "cons":
            num = numCards_cons.get (); ticks = pctProp.get ().rename (index = {"ALL": defaultName})
            typeList = list (); names = list (); colors = list (); pctConcept = list ()
            for idx in range (1, num + 1):
                typeList.append (input[f"typeFS{idx}_cons"] ()); names.append (input[f"name{idx}_cons"] ()); colors.append (input[f"color{idx}_cons"] ())
                if input[f"typeFS{idx}_cons"] () == "trap":
                    pctConcept.append ([int (10 * input[f"coord{idx}_a_cons"] ()), int (10 * input[f"coord{idx}_b_cons"] ()),
                                        int (10 * input[f"coord{idx}_c_cons"] ()), int (10 * input[f"coord{idx}_d_cons"] ())])
                else:
                    pctConcept.append ([10 * input[f"center{idx}_cons"] (), input[f"width{idx}_cons"] ()])
            basicInfo = {"number_fuzzy_sets": num, "label_values": [constRev.get (x, x) if not np.isnan (x) else "NA" for x in labels]}
            if np.isfinite (noiseCutoffLeft.get ()):
                basicInfo["MIN-NOISE"] = noiseCutoffLeft.get ()
            if np.isfinite (noiseCutoffRight.get ()):
                basicInfo["MAX-NOISE"] = noiseCutoffRight.get ()
            with ui.Progress () as p:
                p.set (message = "Download Running", detail = "This will take a while...")
                output = generateOutputFromConstraint (featureList, pctConcept, ticks, allStd.get (), basicInfo, typeList, names, colors)
                outputStr = json.dumps (output, indent = 4)
                yield outputStr
            ui.notification_show ("Download Completed", type = "message", duration = 2)
        elif option == "default":
            num = numCards_default.get (); fit = curveFit.get ().rename (index = {"ALL": defaultName})
            typeList = list (); names = list (); colors = list (); zConcept = list ()
            for idx in range (1, num + 1):
                typeList.append (input[f"typeFS{idx}_default"] ()); names.append (input[f"name{idx}_default"] ()); colors.append (input[f"color{idx}_default"] ())
                if input[f"typeFS{idx}_default"] () == "trap":
                    zConcept.append ([input[f"coord{idx}_a_default"] (), input[f"coord{idx}_b_default"] (),
                                      input[f"coord{idx}_c_default"] (), input[f"coord{idx}_d_default"] ()])
                else:
                    zConcept.append ([input[f"center{idx}_default"] (), input[f"width{idx}_default"] ()])
            basicInfo = {"number_fuzzy_sets": num, "label_values": [constRev.get (x, x) if not np.isnan (x) else "NA" for x in labels]}
            if np.isfinite (noiseCutoffLeft.get ()):
                basicInfo["MIN-NOISE"] = noiseCutoffLeft.get ()
            if np.isfinite (noiseCutoffRight.get ()):
                basicInfo["MAX-NOISE"] = noiseCutoffRight.get ()
            allRanges = matrix.get ().replace (labelValues.get (), np.nan)
            if addNoiseLeft.get ():
                allRanges = allRanges.mask (allRanges <= noiseCutoffLeft.get ())
            if addNoiseRight.get ():
                allRanges = allRanges.mask (allRanges >= noiseCutoffRight.get ())
            allRanges = pd.DataFrame ({"min": allRanges.min (axis = 1, skipna = True), "max": allRanges.max (axis = 1, skipna = True)})
            allRanges = allRanges.replace (np.nan, 0); allRanges.loc[defaultName] = dict (zip (["min", "max"], rangeGlobal.get ()))
            with ui.Progress () as p:
                p.set (message = "Download Running", detail = "This will take a while...")
                output = generateOutputFromDefault (featureList, zConcept, fit, allRanges, basicInfo, typeList, names, colors)
                outputStr = json.dumps (output, indent = 4)
                yield outputStr
            ui.notification_show ("Download Completed", type = "message", duration = 2)
        else:
            raise ValueError


    @reactive.effect
    def _ ():
        file = input.crispMatrix_visual ()
        if file is None:
            mtx = pd.DataFrame ()
        else:
            ui.update_selectize ("viewFeature_raw_visual", selected = "ALL")
            with ui.Progress () as p:
                p.set (message = "Importing Matrix")
                mtx = pd.read_csv (file[0]["datapath"], index_col = 0, sep = "\t").astype (float)
            ui.notification_show ("Import Successful", type = "message", duration = 1.5)
            xMin = np.floor (mtx.replace (-np.inf, np.nan).min (axis = None, skipna = True)) - 1
            xMax = np.ceil (mtx.replace (np.inf, np.nan).max (axis = None, skipna = True)) + 1
            plotRangeGlobal_visual.set ([xMin, xMax])
        matrix_visual.set (mtx)


    @reactive.effect
    def _ ():
        file = input.concept_visual ()
        if file is None:
            concepts = dict ()
        else:
            with open (file[0]["datapath"]) as f:
                tmp = json.load (f)
            const = {"-Infinity": -np.inf, "-infinity": -np.inf, "-Inf": -np.inf, "-inf": -np.inf,
                     "+Infinity": np.inf, "+infinity": np.inf, "+Inf": np.inf, "+inf": np.inf,
                     "Infinity": np.inf, "infinity": np.inf, "Inf": np.inf, "inf": np.inf,
                     "NaN": np.nan, "NAN": np.nan, "nan": np.nan, "NA": np.nan, "na": np.nan}
            concepts = dict ()
            for feature in tmp.keys ():
                c = tmp[feature]; c["label_values"] = [const.get (x, x) for x in c["label_values"]]
                concepts[feature] = c
            concepts_visual.set (concepts)


    @reactive.effect
    @reactive.event (input.checkInput_visual)
    def _ ():
        if input.useMatrix () == "original":
            mtx = matrix.get (); matrix_visual.set (mtx)
        elif input.useMatrix () == "upload" and matrix_visual.get ().empty:
            message = ui.modal ("Please upload one raw matrix (.TSV).", title = "No Crisp Matrix Available",
                                easy_close = True)
            ui.modal_show (message)
        mtx = matrix_visual.get ()
        xMin = np.floor (mtx.replace (-np.inf, np.nan).min (axis = None, skipna = True)) - 1
        xMax = np.ceil (mtx.replace (np.inf, np.nan).max (axis = None, skipna = True)) + 1
        step = estimateStep (xMin, xMax); plotRangeGlobal_visual.set ([xMin, xMax])
        ui.update_slider ("zoom_visual", min = xMin + 1, max = xMax - 1, value = (xMin + 1, xMax - 1), step = step)
        basicInfo = {"number_fuzzy_sets": 0, "label_values": list (set (labelValues.get ()) - set (plotRangeGlobal.get ()))}
        if np.isfinite (noiseCutoffLeft.get ()):
            basicInfo["MIN-NOISE"] = noiseCutoffLeft.get ()
        if np.isfinite (noiseCutoffRight.get ()):
            basicInfo["MAX-NOISE"] = noiseCutoffRight.get ()
        if input.useConcept () == "original":
            defined = input.definedBy ()
            if defined == "cons":
                num = numCards_cons.get (); basicInfo["number_fuzzy_sets"] = num
                names = list (); typeList = list (); colors = list (); tempConcept = list (); gaussIdx = list ()
                for idx in range (1, num + 1):
                    typeList.append (input[f"typeFS{idx}_cons"] ()); names.append (input[f"name{idx}_cons"] ())
                    colors.append (input[f"color{idx}_cons"] ())
                    if typeList[-1] == "trap":
                        tempConcept.append ([int (10 * input[f"coord{idx}_a_cons"] ()), int (10 * input[f"coord{idx}_b_cons"] ()),
                                             int (10 * input[f"coord{idx}_c_cons"] ()), int (10 * input[f"coord{idx}_d_cons"] ())])
                    else:
                        gaussIdx.append (idx)
                        tempConcept.append ([int (10 * input[f"center{idx}_cons"] ()), input[f"width{idx}_cons"] ()])
                concepts = generateOutputFromConstraint (["ALL"] + itemList.get ()["feature"], tempConcept, pctProp.get (), allStd.get (), input.fuzzyBy_cons (),
                                                         basicInfo, typeList, names, colors)
            else:
                num = numCards_default.get (); basicInfo["number_fuzzy_sets"] = num
                names = list (); typeList = list (); colors = list (); tempConcept = list ()
                for idx in range (1, num + 1):
                    typeList.append (input[f"typeFS{idx}_default"] ()); names.append (input[f"name{idx}_default"] ())
                    colors.append (input[f"color{idx}_default"] ())
                    if typeList[-1] == "trap":
                        tempConcept.append ([input[f"coord{idx}_a_default"] (), input[f"coord{idx}_b_default"] (),
                                             input[f"coord{idx}_c_default"] (), input[f"coord{idx}_d_default"] ()])
                    else:
                        tempConcept.append ([input[f"center{idx}_default"] (), input[f"width{idx}_default"] ()])
                allRanges = matrix.get ().replace (labelValues.get (), np.nan)
                allRanges = pd.DataFrame ({"min": allRanges.min (axis = 1, skipna = True), "max": allRanges.max (axis = 1, skipna = True)})
                allRanges = allRanges.replace (np.nan, 0); allRanges.loc["ALL"] = dict (zip (["min", "max"], rangeGlobal.get ()))
                concepts = generateOutputFromDefault (["ALL"] + itemList.get ()["feature"], tempConcept, curveFit.get (), allRanges,
                                                      basicInfo,typeList, names, colors)
            concepts_visual.set (concepts)
        else:
            concepts = concepts_visual.get ()
        ui.update_selectize ("viewFeature_raw_visual", choices = ["ALL"] + list (mtx.index), selected = "ALL")
        ui.update_selectize ("viewFeature_concept_visual", choices = list (concepts.keys ()))
        ui.notification_show ("Import Done", type = "message", duration = 1.5)



    @render.plot
    def globalDist_visual ():
        mtx = matrix_visual.get ().replace ([-np.inf, np.inf], np.nan)
        concepts = concepts_visual.get (); xRange = plotRangeGlobal_visual.get (); selectRange = input.zoom_visual ()
        if mtx.empty or (not concepts) or len (xRange) != 2:
            return
        concept = concepts.get (input.viewFeature_concept_visual (), dict ()); num = concept.get ("number_fuzzy_sets", 0)
        minLevel = concept.get ("MIN-NOISE", -np.inf); maxLevel = concept.get ("MAX-NOISE", np.inf)
        xMin = max (xRange[0], minLevel, selectRange[0]); xMax = min (xRange[1], maxLevel, selectRange[1])
        featureRaw = input.viewFeature_raw_visual ()
        if featureRaw == "ALL":
            values = mtx.melt ()["value"]
        else:
            try:
                values = mtx.loc[featureRaw]
            except KeyError:
                values = pd.Series (dtype = float)
        pctUnlabelled = len (values.mask ((values <= minLevel) | (values >= maxLevel)).dropna ()) / len (values)
        fig, ax = plt.subplots (figsize = (8, 5))
        if input.numBins_visual () != 0:
            ax.hist (values, bins = input.numBins_visual (), color = "lightgray")
        ax.set_xlim (xRange); ax.tick_params (axis = "both", which = "major", labelsize = 8)
        if minLevel > xRange[0]:
            ax.axvline (minLevel, color = "black", linestyle = "dashed")
        if maxLevel < xRange[1]:
            ax.axvline (maxLevel, color = "black", linestyle = "dashed")
        if selectRange[0] > xRange[0]:
            ax.axvline (selectRange[0], color = "darkgray", linestyle = "dashed")
        if selectRange[1] < xRange[1]:
            ax.axvline (selectRange[1], color = "darkgray", linestyle = "dashed")
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of finite values", size = 10)
        ax.set_title (f"percent of unlabelled values - {pctUnlabelled:.1%}", size = 15)
        ax2 = ax.twinx (); ax2.set_ylim ((0, 1.05)); ax2.set_ylabel ("fuzzy value", size = 10)
        names = list (); colors = list (); params = list (); handles = list ()
        for key in concept.keys ():
            if key in ["number_fuzzy_sets", "label_values", "MIN-NOISE", "MAX-NOISE"]:
                continue
            p, typeFS, color, _ = concept[key]; names.append (key); colors.append (color)
            handles.append (Line2D ([0], [0], color = color, linewidth = 2))
            if len (names) == 1 and typeFS == "trapezoidal":
                params.append ([xMin, xMin, p[2], p[3]])
            elif len (names) == num and typeFS == "trapezoidal":
                params.append ([p[0], p[1], xMax, xMax])
            else:
                params.append (p)
        lines, curves = getLines (params, [xMin, xMax], colors)
        ax2.plot (*lines, linewidth = 2)
        for curve in curves:
            ax2.plot (curve[0], curve[1], c = curve[2], linewidth = 2)
        ax2.legend (handles, names, facecolor = "white"); fig.tight_layout ()
        return fig


    @render.plot
    def deviation_visual ():
        mtx = matrix_visual.get ().replace ([-np.inf, np.inf], np.nan)
        concepts = concepts_visual.get (); xRange = plotRangeGlobal_visual.get (); selectRange = input.zoom_visual ()
        if mtx.empty or (not concepts) or len (xRange) != 2:
            return
        concept = concepts.get (input.viewFeature_concept_visual (), dict ()); num = concept.get ("number_fuzzy_sets", 0)
        minLevel = concept.get ("MIN-NOISE", -np.inf); maxLevel = concept.get ("MAX-NOISE", np.inf)
        xMin = max (xRange[0], minLevel, selectRange[0]); xMax = min (xRange[1], maxLevel, selectRange[1])
        expectation = dict (); params = list ()
        for key in concept.keys ():
            if key in ["number_fuzzy_sets", "label_values", "MIN-NOISE", "MAX-NOISE"]:
                continue
            p, typeFS, _, exp = concept[key]; expectation[key] = exp
            if len (concept) == 0 and typeFS == "trapezoidal":
                params.append ([xMin, xMin, p[2], p[3]])
            elif len (concept) == num - 1 and typeFS == "trapezoidal":
                params.append ([p[0], p[1], xMax, xMax])
            else:
                params.append (p)
        featureRaw = input.viewFeature_raw_visual ()
        if featureRaw == "ALL":
            values = mtx.melt ()["value"]
        else:
            try:
                values = mtx.loc[featureRaw]
            except KeyError:
                values = pd.Series (dtype = float)
        observation = dict (zip (expectation.keys (), getPercentage (values, params, labels = list (), minLevel = xMin, maxLevel = xMax)))
        annData = pd.DataFrame ({"expectation": expectation, "observation": observation,
                                 "deviation": {key: observation[key] - expectation[key] for key in expectation.keys ()}})
        pltData = pd.DataFrame ({"expectation": 0, "observation": 0, "deviation": annData["deviation"]}, index = annData.index).T
        cutoff = input.deviationCutoff (); plotCutoff = 0.01 if cutoff == 0 else cutoff
        cmap = sns.color_palette ("vlag", 3)
        fig, ax = plt.subplots (figsize = (8, 5))
        sns.heatmap (pltData, vmin = -3 * plotCutoff, vmax = 3 * plotCutoff, cmap = cmap, center = 0, annot = annData.T, fmt = ".1%",
                     linewidths = 0.5, linecolor = "black", ax = ax)
        ax.set_xticks (ax.get_xticks ()); ax.set_xticklabels (ax.get_xticklabels (), rotation = 0, ha = "center")
        ax.set_yticks (ax.get_yticks ()); ax.set_yticklabels (ax.get_yticklabels (), rotation = 0, ha = "right")
        ax.set_xlabel (""); ax.set_ylabel ("observed percentage - expected percentage", size = 10); ax.yaxis.set_label_position ("right")
        colorbar = ax.collections[0].colorbar; colorbar.set_ticks ([-2 * plotCutoff, -plotCutoff, 0, plotCutoff, 2 * plotCutoff])
        colorbar.set_ticklabels (["not\naccepted", f"{-cutoff:.1%}", "accepted", f"{cutoff:.1%}", "not\naccepted"])
        fig.tight_layout ()
        return fig



app = App (app_ui, server)


