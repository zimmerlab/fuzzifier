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

sns.set_theme (style = "white", rc = {"axes.facecolor": (0, 0, 0, 0)})


app_ui = ui.page_fluid (
    ui.panel_title (ui.h2 ("2-Aspect Fuzzifier - Interactive Tool", class_ = "pt-5")),
    ui.accordion (
        ui.accordion_panel (
            "Data Import",
            ui.navset_card_pill (
                ui.nav_panel (
                    "Crisp Value Matrix for x-Axis",
                    ui.row (
                        ui.column (
                            4,
                            ui.card (
                                ui.input_text ("xLabel", "Name:", value = "X"),
                                ui.input_file ("crispMatrix_X", "Select raw value matrix (.TSV):",
                                               accept = ".tsv", multiple = False, width = "80%"),
                                ui.input_checkbox_group ("specValue_X", "Select values to label:",
                                                         choices = {"-Inf": "-inf", "+Inf": "+inf", "0": "zero"},
                                                         selected = ("-Inf", "+Inf", "0"), inline = True),
                                ui.input_switch ("addNoise_X", "Add category for noise?", False),
                                ui.panel_conditional (
                                    "input.addNoise_X === true",
                                    ui.input_numeric ("minNoiseLevel_X", "Values no smaller than:",
                                                      min = 0, max = 10, value = 0, step = 0.01),
                                    ui.input_numeric ("maxNoiseLevel_X", "Values no larger than:",
                                                      min = -10, max = 0, value = 0, step = 0.01)
                                )
                            ),
                        ),
                        ui.column (
                            8,
                            ui.navset_card_pill (
                                ui.nav_panel (
                                    "Statistics",
                                    ui.output_data_frame ("summarizeCrispMtx_X")
                                ),
                                ui.nav_panel (
                                    "Crisp Value Distribution",
                                    ui.layout_columns (
                                        ui.input_slider ("numBins_X", "Number of bins:", min = 5, max = 100, step = 5,
                                                         value = 50, width = "200px"),
                                        ui.input_slider ("zoom_X", "Visualize range:", min = 0, max = 0, step = 1,
                                                         value = (0, 0), width = "250px")
                                    ),
                                    ui.div (
                                        ui.output_plot ("crispDistribution_X", width = "700px", height = "450px"),
                                        style = "display: flex; justify-content: center;"
                                    )
                                )
                            )
                        )
                    )
                ),
                ui.nav_panel (
                    "Crisp Value Matrix for y-Axis",
                    ui.row (
                        ui.column (
                            4,
                            ui.card (
                                ui.input_text ("yLabel", "Name:", value = "Y"),
                                ui.input_file ("crispMatrix_Y", "Select raw value matrix (.TSV):",
                                               accept = ".tsv", multiple = False, width = "80%"),
                                ui.input_checkbox_group ("specValue_Y", "Select values to label:",
                                                         choices = {"-Inf": "-inf", "+Inf": "+inf",  "0": "zero"},
                                                         selected = ("-Inf", "+Inf", "0"), inline = True),
                                ui.input_switch ("addNoise_Y", "Add category for noise?", False),
                                ui.panel_conditional (
                                    "input.addNoise_Y === true",
                                    ui.input_numeric ("minNoiseLevel_Y", "Values no smaller than:",
                                                      min = 0, max = 10, value = 0, step = 0.01),
                                    ui.input_numeric ("maxNoiseLevel_Y", "Values no larger than:",
                                                      min = -10, max = 0, value = 0, step = 0.01)
                                )
                            )
                        ),
                        ui.column (
                            8,
                            ui.navset_card_pill (
                                ui.nav_panel (
                                    "Statistics",
                                    ui.output_data_frame ("summarizeCrispMtx_Y")
                                ),
                                ui.nav_panel (
                                    "Crisp Value Distribution",
                                    ui.layout_columns (
                                        ui.input_slider ("numBins_Y", "Number of bins:", min = 5, max = 100, step = 5,
                                                         value = 50, width = "200px"),
                                        ui.input_slider ("zoom_Y", "Visualize range:", min = 0, max = 0, step = 1,
                                                         value = (0, 0), width = "250px")
                                    ),
                                    ui.div (
                                        ui.output_plot ("crispDistribution_Y", width = "700px", height = "450px"),
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
                                ui.input_select ("metadataCol", "Select column for clusters:", choices = {"--": "--"}, multiple = False)
                            )
                        ),
                        ui.column (
                            8,
                            ui.output_data_frame ("showMetadata")
                        )
                    )
                )
            ),
            ui.card (
                ui.input_action_button ("checkInput", "Confirm input and proceed", width = "300px")
            )
        ),
        ui.accordion_panel (
            "Derivation Strategy",
            ui.navset_card_tab (
                ui.nav_panel (
                    "x-Axis",
                    ui.navset_pill (
                        ui.nav_panel (
                            "Fixed Parameters",
                            ui.layout_sidebar (
                                ui.sidebar (
                                    ui.card (
                                        id = "FS0_fixed_X"
                                    ),
                                    width = "300px", position = "left", open = "open"
                                ),
                                ui.card (
                                    ui.layout_columns (
                                        "Number of fuzzy sets:",
                                        ui.input_numeric ("numFS_fixed_X", "", value = 3, min = 2, max = 10, step = 1),
                                        ui.input_action_button ("start_fixed_X", "Estimate", width = "200px")
                                    ),
                                    height = "100px"
                                ),
                                ui.card (
                                    ui.layout_column_wrap (
                                        ui.input_radio_buttons ("typeFS_fixed_X", "Function type:",
                                                                choices = {"trap": "trapezoidal", "gauss": "Gaussian"},
                                                                selected = "trap", inline = False, width = "80%"),
                                        ui.download_button ("download_fixed_X", "Download concept", width = "200px"),
                                        ui.input_action_button ("confirm_fixed_X", "Fuzzify", width = "200px")
                                    ),
                                    height = "150px"
                                ),
                                ui.layout_columns (
                                    "Select feature for visualization:",
                                    ui.input_select ("viewFeature_fixed_X", "", choices = {"ALL": "ALL"}, multiple = False, selectize = True)
                                ),
                                ui.div (
                                    ui.output_plot ("globalDist_fixed_X", width = "700px", height = "400px"),
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
                                        id = "FS0_width_X"
                                    ),
                                    width = "300px", position = "left", open = "open"
                                ),
                                ui.card (
                                    ui.layout_columns (
                                        "Number of fuzzy sets:",
                                        ui.input_numeric ("numFS_width_X", "", value = 3, min = 2, max = 10, step = 1),
                                        ui.input_action_button ("start_width_X", "Estimate", width = "200px")
                                    ),
                                    height = "100px"
                                ),
                                ui.card (
                                    ui.layout_column_wrap (
                                        ui.input_radio_buttons ("typeFS_width_X", "Function type:",
                                                                choices = {"trap": "trapezoidal", "gauss": "Gaussian"},
                                                                selected = "trap", inline = False, width = "80%"),
                                        ui.input_radio_buttons ("fuzzyBy_width_X", "Direction:",
                                                                choices = {"feature": "per feature", "dataset": "per data set"},
                                                                selected = "feature", inline = False, width = "80%"),
                                        ui.download_button ("download_width_X", "Download concept", width = "200px"),
                                        ui.div (),
                                        ui.div (),
                                        ui.input_action_button ("confirm_width_X", "Fuzzify", width = "200px"),
                                        width = 1 / 3
                                    ),
                                    height = "150px"
                                ),
                                ui.layout_columns (
                                    "Select feature for visualization:",
                                    ui.input_select ("viewFeature_width_X", "", choices = {"ALL": "ALL"}, multiple = False, selectize = True)
                                ),
                                ui.div (
                                    ui.output_plot ("globalDist_width_X", width = "700px", height = "400px"),
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
                                        id = "FS0_prop_X"
                                    ),
                                    width = "300px", position = "left", open = "open"
                                ),
                                ui.card (
                                    ui.layout_columns (
                                        "Number of fuzzy sets:",
                                        ui.input_numeric ("numFS_prop_X", "", value = 3, min = 2, max = 10, step = 1),
                                        ui.input_action_button ("start_prop_X", "Estimate", width = "200px")
                                    ),
                                    height = "100px"
                                ),
                                ui.card (
                                    ui.layout_column_wrap (
                                        ui.input_radio_buttons ("typeFS_prop_X", "Function type:",
                                                                choices = {"trap": "trapezoidal", "gauss": "Gaussian"},
                                                                selected = "trap", inline = False, width = "80%"),
                                        ui.input_radio_buttons ("fuzzyBy_prop_X", "Direction:",
                                                                choices = {"feature": "per feature", "dataset": "per data set"},
                                                                selected = "feature", inline = False, width = "80%"),
                                        ui.download_button ("download_prop_X", "Download concept", width = "200px"),
                                        ui.div (),
                                        ui.div (),
                                        ui.input_action_button ("confirm_prop_X", "Fuzzify", width = "200px"),
                                        width = 1 / 3
                                    ),
                                    height = "150px"
                                ),
                                ui.layout_columns (
                                    "Select feature for visualization:",
                                    ui.input_select ("viewFeature_prop_X", "", choices = {"ALL": "ALL"}, multiple = False, selectize = True)
                                ),
                                ui.div (
                                    ui.output_plot ("globalDist_prop_X", width = "700px", height = "400px"),
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
                                    ui.input_radio_buttons ("defMode_X", "", selected = "custom", inline = True,
                                                            choices = {"custom": "Customize modes",
                                                                       "fit": "Estimate by Boostrapping"})
                                ),
                                ui.panel_conditional (
                                    "input.defMode_X === 'custom'",
                                    ui.row (
                                        ui.column (
                                            4,
                                            ui.card (
                                                ui.layout_columns (
                                                    "Number of curves to fit:",
                                                    ui.input_numeric ("numModes_custom_X", "", value = 3, min = 1, max = 5, step = 1)
                                                ),
                                                ui.input_action_button ("getMode_X", "Get modes", width = "200px"),
                                                id = "custom0_X"
                                            )
                                        ),
                                        ui.column (
                                            6,
                                            ui.div (
                                                ui.output_plot ("globalModes_custom_X", width = "800px", height = "500px"),
                                                style = "display: flex; justify-content: center;"
                                            )
                                        )
                                    ),
                                    height = "650px"
                                ),
                                ui.panel_conditional (
                                    "input.defMode_X === 'fit'",
                                    ui.row (
                                        ui.column (
                                            4,
                                            ui.input_numeric ("seed_X", "Random seed for Bootstrapping:",
                                                              value = 1, min = 1, max = 100, step = 1),
                                            ui.br (),
                                            ui.input_action_button ("randomSeed_X", "Change seed", width = "250px"),
                                            ui.br (),
                                            ui.br (),
                                            ui.input_numeric ("numValues_X", "Number of raw values per iteration:",
                                                              value = 1000, min = 100, max = 10000, step = 100),
                                            ui.input_numeric ("numIteration_X", "Number of iterations:",
                                                              value = 100, min = 100, max = 1000, step = 100),
                                            ui.br (),
                                            ui.input_action_button ("estimate_X", "Estimate modes", width = "250px")
                                        ),
                                        ui.column (
                                            6,
                                            ui.div (
                                                ui.output_plot ("globalModes_fit_X", width = "800px", height = "500px"),
                                                style = "display: flex; justify-content: center;"
                                            )
                                        )
                                    ),
                                    height = "650px"
                                )
                            ),
                            ui.card (
                                ui.input_action_button ("proceedMode_X", "Proceed with selected modes", width = "500px")
                            ),
                            ui.card (
                                ui.card_header ("Final Fuzzy Concept Derivation"),
                                ui.layout_sidebar (
                                    ui.sidebar (
                                        ui.card (
                                            ui.input_slider ("pctOverlap_X", "Percent of slope region:",
                                                             min = 0, max = 1, value = 0.5, step = 0.05),
                                            id = "PFC0_X"
                                        ),
                                        width = "300px", position = "left", open = "open", heihgt = "1250px"
                                    ),
                                    ui.card (
                                        ui.layout_column_wrap (
                                            ui.download_button ("download_mode_X", "Download concept", width = "200px"),
                                            ui.input_action_button ("confirm_mode_X", "Fuzzify", width = "200px")
                                        ),
                                        height = "100px"
                                    ),
                                    ui.card (
                                        ui.card_header ("Partial Fuzzy Concepts"),
                                        ui.div (
                                            ui.output_plot ("partialConcepts_X", width = "600px", height = "600px"),
                                            style = "display: flex; justify-content: center;"
                                        )
                                    ),
                                    ui.layout_columns (
                                        "Select feature for visualization:",
                                        ui.input_select ("viewFeature_mode_X", "", choices = {"ALL": "ALL"}, multiple = False, selectize = True),
                                    ),
                                    ui.card (
                                        ui.card_header ("Merged Fuzzy Concept"),
                                        ui.div (
                                            ui.output_plot ("mergedConcept_X", width = "750px", height = "350px"),
                                            style = "display: flex; justify-content: center;"
                                        )
                                    )
                                ),
                                height = "1400px"
                            )
                        )
                    )
                ),
                ui.nav_panel (
                    "y-Axis",
                    ui.navset_pill (
                        ui.nav_panel (
                            "Fixed Parameters",
                            ui.layout_sidebar (
                                ui.sidebar (
                                    ui.card (
                                        id = "FS0_fixed_Y"
                                    ),
                                    width = "300px", position = "left", open = "open"
                                ),
                                ui.card (
                                    ui.layout_columns (
                                        "Number of fuzzy sets:",
                                        ui.input_numeric ("numFS_fixed_Y", "", value = 3, min = 2, max = 10, step = 1),
                                        ui.input_action_button ("start_fixed_Y", "Estimate", width = "200px")
                                    ),
                                    height = "100px"
                                ),
                                ui.card (
                                    ui.layout_column_wrap (
                                        ui.input_radio_buttons ("typeFS_fixed_Y", "Function type:",
                                                                choices = {"trap": "trapezoidal", "gauss": "Gaussian"},
                                                                selected = "trap", inline = False, width = "80%"),
                                        ui.download_button ("download_fixed_Y", "Download concept", width = "200px"),
                                        ui.input_action_button ("confirm_fixed_Y", "Fuzzify", width = "200px")
                                    ),
                                    height = "150px"
                                ),
                                ui.layout_columns (
                                    "Select feature for visualization:",
                                    ui.input_select ("viewFeature_fixed_Y", "", choices = {"ALL": "ALL"}, multiple = False, selectize = True)
                                ),
                                ui.div (
                                    ui.output_plot ("globalDist_fixed_Y", width = "700px", height = "400px"),
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
                                        id = "FS0_width_Y"
                                    ),
                                    width = "300px", position = "left", open = "open"
                                ),
                                ui.card (
                                    ui.layout_columns (
                                        "Number of fuzzy sets:",
                                        ui.input_numeric ("numFS_width_Y", "", value = 3, min = 2, max = 10, step = 1),
                                        ui.input_action_button ("start_width_Y", "Estimate", width = "200px")
                                    ),
                                    height = "100px"
                                ),
                                ui.card (
                                    ui.layout_column_wrap (
                                        ui.input_radio_buttons ("typeFS_width_Y", "Function type:",
                                                                choices = {"trap": "trapezoidal", "gauss": "Gaussian"},
                                                                selected = "trap", inline = False, width = "80%"),
                                        ui.input_radio_buttons ("fuzzyBy_width_Y", "Direction:",
                                                                choices = {"feature": "per feature", "dataset": "per data set"},
                                                                selected = "feature", inline = False, width = "80%"),
                                        ui.download_button ("download_width_Y", "Download concept", width = "200px"),
                                        ui.div (),
                                        ui.div (),
                                        ui.input_action_button ("confirm_width_Y", "Fuzzify", width = "200px"),
                                        width = 1 / 3
                                    ),
                                    height = "150px"
                                ),
                                ui.layout_columns (
                                    "Select feature for visualization:",
                                    ui.input_select ("viewFeature_width_Y", "", choices = {"ALL": "ALL"}, multiple = False, selectize = True)
                                ),
                                ui.div (
                                    ui.output_plot ("globalDist_width_Y", width = "700px", height = "400px"),
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
                                        id = "FS0_prop_Y"
                                    ),
                                    width = "300px", position = "left", open = "open"
                                ),
                                ui.card (
                                    ui.layout_columns (
                                        "Number of fuzzy sets:",
                                        ui.input_numeric ("numFS_prop_Y", "", value = 3, min = 2, max = 10, step = 1),
                                        ui.input_action_button ("start_prop_Y", "Estimate", width = "200px")
                                    ),
                                    height = "100px"
                                ),
                                ui.card (
                                    ui.layout_columns (
                                        ui.input_radio_buttons ("typeFS_prop_Y", "Function type:",
                                                                choices = {"trap": "trapezoidal", "gauss": "Gaussian"},
                                                                selected = "trap", inline = False, width = "80%"),
                                        ui.input_radio_buttons ("fuzzyBy_prop_Y", "Direction:",
                                                                choices = {"feature": "per feature", "dataset": "per data set"},
                                                                selected = "feature", inline = False, width = "80%"),
                                        ui.download_button ("download_prop_Y", "Download concept", width = "200px"),
                                        ui.input_action_button ("confirm_prop_Y", "Fuzzify", width = "200px")
                                    ),
                                    height = "150px"
                                ),
                                ui.layout_columns (
                                    "Select feature for visualization:",
                                    ui.input_select ("viewFeature_prop_Y", "", choices = {"ALL": "ALL"}, multiple = False, selectize = True)
                                ),
                                ui.div (
                                    ui.output_plot ("globalDist_prop_Y", width = "700px", height = "400px"),
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
                                    ui.input_radio_buttons ("defMode_Y", "", selected = "custom", inline = True,
                                                            choices = {"custom": "Customize modes",
                                                                       "fit": "Estimate by Boostrapping"})
                                ),
                                ui.panel_conditional (
                                    "input.defMode_Y === 'custom'",
                                    ui.row (
                                        ui.column (
                                            4,
                                            ui.card (
                                                ui.layout_columns (
                                                    "Number of curves to fit:",
                                                    ui.input_numeric ("numModes_custom_Y", "", value = 3, min = 1, max = 5, step = 1)
                                                ),
                                                ui.input_action_button ("getMode_Y", "Get modes", width = "200px"),
                                                id = "custom0_Y"
                                            )
                                        ),
                                        ui.column (
                                            6,
                                            ui.div (
                                                ui.output_plot ("globalModes_custom_Y", width = "800px", height = "500px"),
                                                style = "display: flex; justify-content: center;"
                                            )
                                        )
                                    ),
                                    height = "650px"
                                ),
                                ui.panel_conditional (
                                    "input.defMode_Y === 'fit'",
                                    ui.row (
                                        ui.column (
                                            4,
                                            ui.input_numeric ("seed_Y", "Random seed for Bootstrapping:",
                                                              value = 1, min = 1, max = 100, step = 1),
                                            ui.br (),
                                            ui.input_action_button ("randomSeed_Y", "Change seed", width = "250px"),
                                            ui.br (),
                                            ui.br (),
                                            ui.input_numeric ("numValues_Y", "Number of raw values per iteration:",
                                                              value = 1000, min = 100, max = 10000, step = 100),
                                            ui.input_numeric ("numIteration_Y", "Number of iterations:",
                                                              value = 100, min = 100, max = 1000, step = 100),
                                            ui.br (),
                                            ui.input_action_button ("estimate_Y", "Estimate modes", width = "250px")
                                        ),
                                        ui.column (
                                            6,
                                            ui.div (
                                                ui.output_plot ("globalModes_fit_Y", width = "800px", height = "500px"),
                                                style = "display: flex; justify-content: center;"
                                            )
                                        )
                                    ),
                                    height = "650px"
                                )
                            ),
                            ui.card (
                                ui.input_action_button ("proceedMode_Y", "Proceed with selected modes", width = "500px")
                            ),
                            ui.card (
                                ui.card_header ("Final Fuzzy Concept Derivation"),
                                ui.layout_sidebar (
                                    ui.sidebar (
                                        ui.card (
                                            ui.input_slider ("pctOverlap_Y", "Percent of slope region:",
                                                             min = 0, max = 1, value = 0.5, step = 0.05),
                                            id = "PFC0_Y"
                                        ),
                                        width = "300px", position = "left", open = "open", heihgt = "1250px"
                                    ),
                                    ui.card (
                                        ui.layout_column_wrap (
                                            ui.download_button ("download_mode_Y", "Download concept", width = "200px"),
                                            ui.input_action_button ("confirm_mode_Y", "Fuzzify", width = "200px")
                                        ),
                                        height = "100px"
                                    ),
                                    ui.card (
                                        ui.card_header ("Partial Fuzzy Concepts"),
                                        ui.div (
                                            ui.output_plot ("partialConcepts_Y", width = "600px", height = "600px"),
                                            style = "display: flex; justify-content: center;"
                                        )
                                    ),
                                    ui.layout_columns (
                                        "Select feature for visualization:",
                                        ui.input_select ("viewFeature_mode_Y", "", choices = {"ALL": "ALL"}, multiple = False, selectize = True)
                                    ),
                                    ui.card (
                                        ui.card_header ("Merged Fuzzy Concept"),
                                        ui.div (
                                            ui.output_plot ("mergedConcept_Y", width = "750px", height = "350px"),
                                            style = "display: flex; justify-content: center;"
                                        )
                                    )
                                ),
                                height = "1400px"
                            )
                        )
                    )
                ),
                ui.nav_panel (
                    "Theoretical Volcano Plot",
                    ui.row (
                        ui.column (
                            3,
                            ui.input_select ("viewFeature", "Select feature for visualization:",
                                             choices = {"ALL": "ALL"}, multiple = False,
                                             selectize = True),
                            ui.input_select ("viewConcept_X", "Select cutoff for x-axis:",
                                             choices = {"fixed": "Fixed parameters",
                                                        "width": "Derived from width",
                                                        "prop": "Derived from proportion",
                                                        "mode": "Derived from modes"},
                                             multiple = False, selectize = False),
                            ui.input_select ("viewConcept_Y", "Select cutoff for y-axis:",
                                             choices = {"fixed": "Fixed parameters",
                                                        "width": "Derived from width",
                                                        "prop": "Derived from proportion",
                                                        "mode": "Derived from modes"},
                                             multiple = False, selectize = False),
                            ui.input_action_button ("updateVolcano", "Update", width = "200px")
                        ),
                        ui.column (
                            8,
                            ui.div (
                                ui.output_plot ("volcano", width = "750px", height = "750px"),
                                style = "display: flex; justify-content: center;"
                            )
                        )
                    )
                )
            )
        ),
        ui.accordion_panel (
            "Customization and Download Section",
            ui.layout_columns (
                ui.card (
                    ui.layout_columns (
                        ui.input_select ("downloadDirection", "", selected = "set",
                                         choices = {"feature": "Per feature",
                                                    "sample": "Per sample",
                                                    "set": "Per fuzzy set"},
                                         multiple = False, width = "200px"),
                        ui.download_button ("saveFuzzy", "Download results",
                                            width = "200px", height = "200px")
                    )
                ),
                ui.card (
                    ui.download_button ("saveEvaluation", "Download report(s)", width = "200px")
                )
            ),
            ui.navset_card_tab (
                ui.nav_panel (
                    "Fuzzy Variable and Colouring",
                    ui.layout_columns (
                        ui.card (
                            ui.card_header ("x-Axis"),
                            ui.card (
                                id = "rename_PH_X"
                            )
                        ),
                        ui.card (
                            ui.card_header ("y-Axis"),
                            ui.card (
                                id = "rename_PH_Y"
                            )
                        )
                    )
                ),
                ui.nav_panel (
                    "Feature Selection Parameters",
                    ui.layout_columns (
                        ui.card (
                            ui.card_header ("x-Axis"),
                            ui.input_select ("base_X", "Select fuzzy set as base level for marker selection:",
                                             choices = {0: "FS0"}, multiple = False, selectize = False),
                            ui.input_slider ("maxSpecific_X",
                                             "Select maximal number of clusters for markers:",
                                             min = 1, max = 10, value = 1, step = 1),
                            ui.input_slider ("minPercent_X",
                                             "Select minimal percentage of samples per main fuzzy set:",
                                             min = 0, max = 1, value = 0.5, step = 0.01),
                            ui.input_select ("sizeCol_X",
                                             "Select parameter for significance of cluster-specificity:",
                                             choices = {"avgFV": "average fuzzy value",
                                                        "pctMain": "percentage of samples"},
                                             selected = "avgFV", multiple = False, selectize = False)
                        ),
                        ui.card (
                            ui.card_header ("y-Axis"),
                            ui.input_select ("base_Y", "Select fuzzy set as base level for marker selection:",
                                             choices = {0: "FS0"}, multiple = False, selectize = False),
                            ui.input_slider ("maxSpecific_Y",
                                             "Select maximal number of clusters for markers:",
                                             min = 1, max = 10, value = 1, step = 1),
                            ui.input_slider ("minPercent_Y",
                                             "Select minimal percentage of samples per main fuzzy set:",
                                             min = 0, max = 1, value = 0.5, step = 0.01),
                            ui.input_select ("sizeCol_Y",
                                             "Select parameter for significance of cluster-specificity:",
                                             choices = {"avgFV": "average fuzzy value",
                                                        "pctMain": "percentage of samples"},
                                             selected = "avgFV", multiple = False, selectize = False)
                        )
                    )
                ),
                ui.nav_panel (
                    "Annotated Volcano Plot",
                    ui.card (
                        ui.row (
                            ui.column (
                                3,
                                ui.input_slider ("cutoffMainFV_X",
                                                 "Select cutoff for main fuzzy value (X):",
                                                 min = 0, max = 1, value = 0.75, step = 0.01),
                                ui.input_slider ("cutoffMainFV_Y",
                                                 "Select cutoff for main fuzzy value (Y):",
                                                 min = 0, max = 1, value = 0.75, step = 0.01),
                                ui.input_action_button ("updateVolcanoAnnot", "Update annotation",
                                                        width = "200px")
                            ),
                            ui.column (
                                8,
                                ui.div (
                                    ui.output_plot ("volcanoAnnotated", width = "750px", height = "750px"),
                                    style = "display: flex; justify-content: center;"
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)



def server (input, output, session):
    itemList = reactive.value ({"feature": list (), "sample": list ()})
    clustering = reactive.value (pd.Series (dtype = str))

    matrix_X = reactive.value (pd.DataFrame (dtype = float))
    tempMatrix_X = reactive.value (pd.DataFrame (dtype = float))
    plotRangeGlobal_X = reactive.value (list ())
    rangeGlobal_X = reactive.value (list ())
    addNoiseLeft_X = reactive.value (False)
    addNoiseRight_X = reactive.value (False)
    labelValues_X = reactive.value (list ())
    pctWidth_X = reactive.value (pd.DataFrame (dtype = float))
    pctProp_X = reactive.value (pd.DataFrame (dtype = float))
    tempConcepts_fixed_X = reactive.value (dict ())
    tempConcepts_width_X = reactive.value (dict ())
    tempConcepts_prop_X = reactive.value (dict ())
    numCards_fixed_X = reactive.value (0)
    numCards_width_X = reactive.value (0)
    numCards_prop_X = reactive.value (0)
    numCards_custom_X = reactive.value (0)
    numCards_mode_X = reactive.value (0)
    centerGlobal_custom_X = reactive.value (list ())
    widthGlobal_custom_X = reactive.value (list ())
    centerGlobal_fit_X = reactive.value (list ())
    widthGlobal_fit_X = reactive.value (list ())
    useModeMethod_X = reactive.value ("custom")
    proceed_X = reactive.value (False)
    tempPartialConcepts_X = reactive.value (list ())
    tempMergedConcept_X = reactive.value (np.array (list ()))
    allConcepts_X = reactive.value (dict ())
    numFuzzySets_X = reactive.value (0)
    nameFuzzySets_X = reactive.value (list ())
    fuzzyValues_X = reactive.value (np.array (list ()))
    globalConcept_X = reactive.value (np.array (list ()))
    conceptInfo_X = reactive.value (dict ())
    idRenameCards_X = reactive.value (list ())
    markerStats_X = reactive.value (pd.DataFrame ())
    mainFuzzyValues_X = reactive.value (pd.DataFrame ())
    mainFuzzySets_X = reactive.value (pd.DataFrame ())
    diffMainFuzzyValues_X = reactive.value (pd.DataFrame ())

    matrix_Y = reactive.value (pd.DataFrame (dtype = float))
    tempMatrix_Y = reactive.value (pd.DataFrame (dtype = float))
    plotRangeGlobal_Y = reactive.value (list ())
    rangeGlobal_Y = reactive.value (list ())
    addNoiseLeft_Y = reactive.value (False)
    addNoiseRight_Y = reactive.value (False)
    labelValues_Y = reactive.value (list ())
    pctWidth_Y = reactive.value (pd.DataFrame (dtype = float))
    pctProp_Y = reactive.value (pd.DataFrame (dtype = float))
    tempConcepts_fixed_Y = reactive.value (dict ())
    tempConcepts_width_Y = reactive.value (dict ())
    tempConcepts_prop_Y = reactive.value (dict ())
    numCards_fixed_Y = reactive.value (0)
    numCards_width_Y = reactive.value (0)
    numCards_prop_Y = reactive.value (0)
    numCards_custom_Y = reactive.value (0)
    numCards_mode_Y = reactive.value (0)
    centerGlobal_custom_Y = reactive.value (list ())
    widthGlobal_custom_Y = reactive.value (list ())
    centerGlobal_fit_Y = reactive.value (list ())
    widthGlobal_fit_Y = reactive.value (list ())
    useModeMethod_Y = reactive.value ("custom")
    proceed_Y = reactive.value (False)
    tempPartialConcepts_Y = reactive.value (list ())
    tempMergedConcept_Y = reactive.value (np.array (list ()))
    allConcepts_Y = reactive.value (dict ())
    numFuzzySets_Y = reactive.value (0)
    nameFuzzySets_Y = reactive.value (list ())
    fuzzyValues_Y = reactive.value (np.array (list ()))
    globalConcept_Y = reactive.value (np.array (list ()))
    conceptInfo_Y = reactive.value (dict ())
    idRenameCards_Y = reactive.value (list ())
    markerStats_Y = reactive.value (pd.DataFrame ())
    mainFuzzyValues_Y = reactive.value (pd.DataFrame ())
    mainFuzzySets_Y = reactive.value (pd.DataFrame ())
    diffMainFuzzyValues_Y = reactive.value (pd.DataFrame ())


    @reactive.effect
    def _ ():
        file = input.crispMatrix_X ()
        if file is None:
            mtx = pd.DataFrame ()
        else:
            with ui.Progress () as p:
                p.set (message = "Importing Matrix")
                mtx = pd.read_csv (file[0]["datapath"], index_col = 0, sep = "\t").astype (float)
            ui.notification_show ("Import Successful", type = "message", duration = 1.5)
            xMin = np.floor (mtx.replace (-np.inf, np.nan).min (axis = None, skipna = True)) - 1
            xMax = np.ceil (mtx.replace (np.inf, np.nan).max (axis = None, skipna = True)) + 1
            step = estimateStep (xMin, xMax); plotRangeGlobal_X.set ([xMin, xMax])
            ui.update_numeric ("minNoiseLevel_X", min = xMin, max = xMax, value = xMin,
                               step = step)
            ui.update_numeric ("maxNoiseLevel_X", min = xMin, max = xMax, value = xMax,
                               step = step)
            ui.update_numeric ("zoom_X", min = xMin, max = xMax, value = (xMin, xMax), step = step)
        matrix_X.set (mtx)


    @reactive.effect
    def _ ():
        mtx = matrix_X.get ()
        if (mtx.empty) or (not input.addNoise_X ()) or len (plotRangeGlobal_X.get ()) != 2:
            tempMatrix_X.set (pd.DataFrame ())
            return
        if input.addNoise_X ():
            noiseRepLeft, noiseRepRight = plotRangeGlobal_X.get ()
            minLevel = input.minNoiseLevel_X (); minLevel = noiseRepLeft if minLevel is None else minLevel
            maxLevel = input.maxNoiseLevel_X (); maxLevel = noiseRepRight if maxLevel is None else maxLevel
            if minLevel > noiseRepLeft:
                tempMatrix_X.set (mtx.mask ((mtx <= minLevel) & np.isfinite (mtx) & (mtx != 0), noiseRepLeft))
                addNoiseLeft_X.set (True)
            else:
                addNoiseLeft_X.set (False)
            if maxLevel < noiseRepRight:
                tempMatrix_X.set (mtx.mask ((mtx >= maxLevel) & np.isfinite (mtx) & (mtx != 0), noiseRepRight))
                addNoiseRight_X.set (True)
            else:
                addNoiseRight_X.set (False)
        else:
            addNoiseLeft_X.set (False); addNoiseRight_X.set (False)


    @render.data_frame
    def summarizeCrispMtx_X ():
        if input.addNoise_X () and (not tempMatrix_X.get ().empty):
            mtx = tempMatrix_X.get ()
        else:
            mtx = matrix_X.get ()
        try:
            noiseRep = plotRangeGlobal_X.get ()
        except:
            noiseRep = None
        summary = getMtxSummary (mtx, labelValues_X.get (), noiseRep = noiseRep)
        return render.DataGrid (summary, width = "100%", styles = {"style": {"height": "50px"}})


    @reactive.effect
    def _ ():
        labels = [float (x) for x in input.specValue_X ()] + [np.nan]
        if matrix_X.get ().empty:
            return
        if input.addNoise_X () and (not tempMatrix_X.get ().empty):
            if addNoiseLeft_X.get ():
                labels.append (plotRangeGlobal_X.get ()[0])
            if addNoiseRight_X.get ():
                labels.append (plotRangeGlobal_X.get ()[1])
            mtx = tempMatrix_X.get ().replace (labels + [-np.inf, np.inf], np.nan)
        else:
            mtx = matrix_X.get ().replace (labels + [-np.inf, np.inf], np.nan)
        xMin = np.floor (mtx.min (axis = None, skipna = True)) - 1
        xMax = np.ceil (mtx.max (axis = None, skipna = True)) + 1
        labelValues_X.set (labels); rangeGlobal_X.set ([xMin, xMax])


    @render.plot
    def crispDistribution_X ():
        visualRange = input.zoom_X ()
        if matrix_X.get ().empty or visualRange[0] == visualRange[1]:
            return
        label = labelValues_X.get ()
        if input.addNoise_X () and (not tempMatrix_X.get ().empty):
            mtx = tempMatrix_X.get ().replace (label + [-np.inf, np.inf], np.nan)
        else:
            mtx = matrix_X.get ().replace (label + [-np.inf, np.inf], np.nan)
        mtx = mtx.melt ()["value"].dropna ()
        mtx = mtx.loc[(mtx >= visualRange[0]) & (mtx <= visualRange[1])]
        fig, ax = plt.subplots (1, figsize = (15, 6))
        ax.hist (mtx, bins = input.numBins_X ())
        ax.set_xlim (visualRange)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        return fig


    @reactive.effect
    def _ ():
        file = input.crispMatrix_Y ()
        if file is None:
            mtx = pd.DataFrame ()
        else:
            with ui.Progress () as p:
                p.set (message = "Importing Matrix")
                mtx = pd.read_csv (file[0]["datapath"], index_col = 0, sep = "\t").astype (float)
            ui.notification_show ("Import Successful", type = "message", duration = 1.5)
            xMin = np.floor (mtx.replace (-np.inf, np.nan).min (axis = None, skipna = True)) - 1
            xMax = np.ceil (mtx.replace (np.inf, np.nan).max (axis = None, skipna = True)) + 1
            step = estimateStep (xMin, xMax); plotRangeGlobal_Y.set ([xMin, xMax])
            ui.update_numeric ("minNoiseLevel_Y", min = xMin, max = xMax, value = xMin,
                               step = step)
            ui.update_numeric ("maxNoiseLevel_Y", min = xMin, max = xMax, value = xMax,
                               step = step)
            ui.update_numeric ("zoom_Y", min = xMin, max = xMax, value = (xMin, xMax), step = step)
        matrix_Y.set (mtx)


    @reactive.effect
    def _ ():
        mtx = matrix_Y.get ()
        if (mtx.empty) or (not input.addNoise_Y ()) or len (plotRangeGlobal_Y.get ()) != 2:
            tempMatrix_Y.set (pd.DataFrame ())
            return
        if input.addNoise_Y ():
            noiseRepLeft, noiseRepRight = plotRangeGlobal_Y.get ()
            minLevel = input.minNoiseLevel_Y (); minLevel = noiseRepLeft if minLevel is None else minLevel
            maxLevel = input.maxNoiseLevel_Y (); maxLevel = noiseRepRight if maxLevel is None else maxLevel
            if minLevel > noiseRepLeft:
                tempMatrix_Y.set (mtx.mask ((mtx <= minLevel) & np.isfinite (mtx) & (mtx != 0), noiseRepLeft))
                addNoiseLeft_Y.set (True)
            else:
                addNoiseLeft_Y.set (False)
            if maxLevel < noiseRepRight:
                tempMatrix_Y.set (mtx.mask ((mtx >= maxLevel) & np.isfinite (mtx) & (mtx != 0), noiseRepRight))
                addNoiseRight_Y.set (True)
            else:
                addNoiseRight_Y.set (False)
        else:
            addNoiseLeft_Y.set (False); addNoiseRight_Y.set (False)


    @render.data_frame
    def summarizeCrispMtx_Y ():
        if input.addNoise_Y () and (not tempMatrix_Y.get ().empty):
            mtx = tempMatrix_Y.get ()
        else:
            mtx = matrix_Y.get ()
        try:
            noiseRep = plotRangeGlobal_Y.get ()
        except:
            noiseRep = None
        summary = getMtxSummary (mtx, labelValues_Y.get (), noiseRep = noiseRep)
        return render.DataGrid (summary, width = "100%", styles = {"style": {"height": "50px"}})


    @reactive.effect
    def _ ():
        labels = [float (x) for x in input.specValue_Y ()] + [np.nan]
        if matrix_Y.get ().empty:
            return
        if input.addNoise_Y () and (not tempMatrix_Y.get ().empty):
            if addNoiseLeft_Y.get ():
                labels.append (plotRangeGlobal_Y.get ()[0])
            if addNoiseRight_Y.get ():
                labels.append (plotRangeGlobal_Y.get ()[1])
            mtx = tempMatrix_Y.get ().replace (labels + [-np.inf, np.inf], np.nan)
        else:
            mtx = matrix_Y.get ().replace (labels + [-np.inf, np.inf], np.nan)
        xMin = np.floor (mtx.min (axis = None, skipna = True)) - 1
        xMax = np.ceil (mtx.max (axis = None, skipna = True)) + 1
        labelValues_Y.set (labels); rangeGlobal_Y.set ([xMin, xMax])


    @render.plot
    def crispDistribution_Y ():
        visualRange = input.zoom_Y ()
        if matrix_Y.get ().empty or visualRange[0] == visualRange[1]:
            return
        label = labelValues_Y.get ()
        if input.addNoise_Y () and (not tempMatrix_Y.get ().empty):
            mtx = tempMatrix_Y.get ().replace (label + [-np.inf, np.inf], np.nan)
        else:
            mtx = matrix_Y.get ().replace (label + [-np.inf, np.inf], np.nan)
        mtx = mtx.melt ()["value"].dropna ()
        mtx = mtx.loc[(mtx >= visualRange[0]) & (mtx <= visualRange[1])]
        fig, ax = plt.subplots (1, figsize = (15, 6))
        ax.hist (mtx, bins = input.numBins_Y ())
        ax.set_xlim (visualRange)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        return fig


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
    @reactive.event (input.checkInput)
    def _ ():
        if matrix_X.get ().empty or matrix_Y.get ().empty:
            message = ui.modal ("Please upload one raw matrix (.TSV) per axis.",
                                title = "No Crisp Matrix Available", easy_close = True)
            ui.modal_show (message)
        else:
            if input.addNoise_X ():
                matrix_X.set (tempMatrix_X.get ())
            if input.addNoise_Y ():
                matrix_Y.set (tempMatrix_Y.get ())
            mtx_X = matrix_X.get (); tempMatrix_X.set (pd.DataFrame ())
            mtx_Y = matrix_Y.get (); tempMatrix_Y.set (pd.DataFrame ())
            items = {"feature": sorted (set (mtx_X.index) & set (mtx_Y.index)),
                     "sample": sorted (set (mtx_X.columns) & set (mtx_Y.columns))}
            mtx_X = mtx_X.loc[items["feature"], items["sample"]]; matrix_X.set (mtx_X)
            mtx_Y = mtx_Y.loc[items["feature"], items["sample"]]; matrix_Y.set (mtx_Y)
            xLabels = labelValues_X.get (); yLabels = labelValues_Y.get (); itemList.set (items)
            xRange = [np.floor (mtx_X.replace (xLabels + [-np.inf], np.nan).min (axis = None, skipna = True)) - 1,
                      np.ceil (mtx_X.replace (xLabels + [np.inf], np.nan).max (axis = None, skipna = True)) + 1]
            yRange = [np.floor (mtx_Y.replace (yLabels + [-np.inf], np.nan).min (axis = None, skipna = True)) - 1,
                      np.ceil (mtx_Y.replace (yLabels + [np.inf], np.nan).max (axis = None, skipna = True)) + 1]
            rangeGlobal_X.set (xRange); rangeGlobal_Y.set (yRange)
            widthTicks, propTicks = getSegments (mtx_X, xLabels, xRange)
            pctWidth_X.set (widthTicks); pctProp_X.set (propTicks)
            widthTicks, propTicks = getSegments (mtx_Y, yLabels, yRange)
            pctWidth_Y.set (widthTicks); pctProp_Y.set (propTicks)
            clusters = clustering.get ()
            if clusters.empty:
                clusters = pd.Series ("TOTAL", index = items["sample"])
            clusters = clusters[items["sample"]]; clustering.set (clusters)
            ui.update_slider ("maxSpecific_X", max = len (set (clustering.get ())))
            ui.update_slider ("maxSpecific_Y", max = len (set (clustering.get ())))
            ui.update_select ("viewFeature", choices = ["ALL"] + items["feature"])
            ui.update_select ("viewFeature_fixed_X", choices = ["ALL"] + items["feature"])
            ui.update_select ("viewFeature_width_X", choices = ["ALL"] + items["feature"])
            ui.update_select ("viewFeature_prop_X", choices = ["ALL"] + items["feature"])
            ui.update_select ("viewFeature_mode_X", choices = ["ALL"] + items["feature"])
            ui.update_select ("viewFeature_fixed_Y", choices = ["ALL"] + items["feature"])
            ui.update_select ("viewFeature_width_Y", choices = ["ALL"] + items["feature"])
            ui.update_select ("viewFeature_prop_Y", choices = ["ALL"] + items["feature"])
            ui.update_select ("viewFeature_mode_Y", choices = ["ALL"] + items["feature"])
            ui.notification_show ("Crisp Value Matrix Done", type = "message", duration = 1.5)


    @reactive.effect
    def _ ():
        if conceptInfo_X.get ():
            ui.update_select ("viewConcept_X", selected = conceptInfo_X.get ()["method"])
        if conceptInfo_Y.get ():
            ui.update_select ("viewConcept_Y", selected = conceptInfo_Y.get ()["method"])


    @reactive.effect
    @reactive.event (input.start_fixed_X)
    def _ ():
        mtx = matrix_X.get (); labels = labelValues_X.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_fixed_X (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame (mtx.melt ()["value"].replace (labels + [-np.inf, np.inf], np.nan).dropna ()).T
            cutoff = estimateCutoff (dummy, percents).loc["value"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)], 3).T,
                   "gauss": np.round (cutoff.tolist ()[1:-1], 3)}
            tempConcepts_fixed_X.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix_X.get (); concepts = tempConcepts_fixed_X.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]
        num = numCards_fixed_X.get (); currNum = trap.shape[0]; xMin, xMax = rangeGlobal_X.get ()
        width = xMax - xMin; step = estimateStep (xMin, xMax)
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_fixed_X", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"intersection{idx}_fixed_X", value = trap[i, 0])
            ui.update_numeric (f"slope{idx}_fixed_X", value = trap[i, 1])
            ui.update_numeric (f"cutoff{idx}_fixed_X", value = gauss[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.panel_conditional (
                        "input.typeFS_fixed_X === 'trap'",
                        ui.layout_columns (
                            "Cutoff:",
                            ui.input_numeric (f"intersection{idx}_fixed_X", "", step = step, min = xMin, max = xMax, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Slope:",
                            ui.input_numeric (f"slope{idx}_fixed_X", "", step = step, min = 0, max = width, value = trap[i, 1])
                        )
                    ),
                    ui.panel_conditional (
                        "input.typeFS_fixed_X === 'gauss'",
                        ui.layout_columns (
                            "Cutoff:",
                            ui.input_numeric (f"cutoff{idx}_fixed_X", "", step = step, min = xMin, max = xMax, value = gauss[i])
                        )
                    ),
                    id = f"FS{idx}_fixed_X"
                ),
                selector = f"#FS{i}_fixed_X", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_fixed_X.set (currNum)


    @render.plot
    def globalDist_fixed_X ():
        mtx = matrix_X.get (); feature = input.viewFeature_fixed_X ()
        if mtx.empty or len (plotRangeGlobal_X.get ()) != 2:
            return
        mtx = mtx.replace (labelValues_X.get () + [-np.inf, np.inf], np.nan)
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
            except:
                pctUnlabelled = "0.0%"
                pass
        ax.set_xlim (plotRangeGlobal_X.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_fixed_X.get (); valueRange = rangeGlobal_X.get ()
        if num > 0 and len (valueRange) == 2:
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal_X.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_ylabel ("fuzzy value", size = 10)
            if input.typeFS_fixed_X () == "trap":
                try:
                    concept = np.array ([[input[f"intersection{i}_fixed_X"] (), input[f"slope{i}_fixed_X"] ()]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "trap", valueRange)
                    ax2.plot (*getLines (concept, colours = list ()))
                except TypeError:
                    pass
            elif input.typeFS_fixed_X () == "gauss":
                try:
                    concept = np.array ([input[f"cutoff{i}_fixed_X"] () for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", valueRange)
                    xValues = np.linspace (*valueRange, 1000)
                    lines = getCurves (concept, valueRange, colours = list (), setPlateau = True)
                    for idx in range (num + 1):
                        ax2.plot (xValues, lines[idx][0], color = lines[idx][1])
                    ax2.plot ((valueRange[0], valueRange[0]), (0, 1), lines[0][1])
                    ax2.plot ((valueRange[1], valueRange[1]), (1, 0), lines[-1][1])
                except TypeError:
                    pass
            else:
                raise ValueError
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_fixed_parameters_X.json")
    def download_fixed_X ():
        num = numCards_fixed_X.get (); valueRange = pctWidth_X.get ()[[0, 1000]]
        globalRange = rangeGlobal_X.get ()
        if input.typeFS_fixed_X () == "trap":
            concept = np.array ([[input[f"intersection{i}_fixed_X"] (), input[f"slope{i}_fixed_X"] ()]
                                 for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "trap", globalRange)
            concepts = dict ()
            for feature in itemList.get ()["feature"]:
                tmp = concept.copy ()
                left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
                right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
                tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
                concepts[feature] = tmp
        elif input.typeFS_fixed_X () == "gauss":
            concept = np.array ([input[f"cutoff{i}_fixed_X"] () for i in range (1, num + 1)])
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
    @reactive.event (input.confirm_fixed_X)
    def _ ():
        mtx = matrix_X.get (); labels = labelValues_X.get (); globalRange = rangeGlobal_X.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        num = numCards_fixed_X.get (); valueRange = pctWidth_X.get ()[[0, 1000]]
        if input.typeFS_fixed_X () == "trap":
            concept = np.array ([[input[f"intersection{i}_fixed_X"] (), input[f"slope{i}_fixed_X"] ()]
                                 for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "trap", globalRange)
            concepts = dict ()
            for feature in itemList.get ()["feature"]:
                tmp = concept.copy ()
                left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
                right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
                tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
                concepts[feature] = tmp
        elif input.typeFS_fixed_X () == "gauss":
            concept = np.array ([input[f"cutoff{i}_fixed_X"] () for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", globalRange)
            concepts = {feature: concept for feature in itemList.get ()["feature"]}
        else:
            raise ValueError
        allFV = list (); featureList = itemList.get ()["feature"]
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts_X.set (concepts); globalConcept_X.set (concept)
        numFuzzySets_X.set (concept.shape[0]); fuzzyValues_X.set (np.array (allFV))
        markerStats_X.set (pd.DataFrame ()); mainFuzzyValues_X.set (pd.DataFrame ())
        mainFuzzySets_X.set (pd.DataFrame ()); diffMainFuzzyValues_X.set (pd.DataFrame ())
        noiseRep = plotRangeGlobal_X.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets_X.set (tmp); conceptInfo_X.set ({"method": "fixed", "direction": "dataset"})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.start_width_X)
    def _ ():
        mtx = matrix_X.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_width_X (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame ({"percents": range (101)}).T
            cutoff = estimateCutoff (dummy, percents).loc["percents"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T,
                   "gauss": np.round (cutoff.tolist ()[1:-1])}
            tempConcepts_width_X.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix_X.get (); concepts = tempConcepts_width_X.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]
        num = numCards_width_X.get (); currNum = trap.shape[0]
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_width_X", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"intersection{idx}_width_X", value = trap[i, 0])
            ui.update_numeric (f"slope{idx}_width_X", value = trap[i, 1])
            ui.update_numeric (f"cutoff{idx}_width_X", value = gauss[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.panel_conditional (
                        "input.typeFS_width_X === 'trap'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"intersection{idx}_width_X", "", step = 0.1, min = 0, max = 100, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Slope (%):",
                            ui.input_numeric (f"slope{idx}_width_X", "", step = 0.1, min = 0, max = 100, value = trap[i, 1])
                        )
                    ),
                    ui.panel_conditional (
                        "input.typeFS_width_X === 'gauss'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"cutoff{idx}_width_X", "", step = 0.1, min = 0, max = 100, value = gauss[i])
                        )
                    ),
                    id = f"FS{idx}_width_X"
                ),
                selector = f"#FS{i}_width_X", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_width_X.set (currNum)


    @render.plot
    def globalDist_width_X ():
        mtx = matrix_X.get (); feature = input.viewFeature_width_X ()
        if mtx.empty or len (plotRangeGlobal_X.get ()) != 2:
            return
        mtx = mtx.replace (labelValues_X.get () + [-np.inf, np.inf], np.nan)
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
                pass
        ax.set_xlim (plotRangeGlobal_X.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_width_X.get (); valueRange = rangeGlobal_X.get (); ticks = pctWidth_X.get ()
        if num > 0 and len (valueRange) == 2:
            feature = "ALL" if input.fuzzyBy_width_X () == "dataset" else feature
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal_X.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            if input.typeFS_width_X () == "trap":
                try:
                    concept = np.array ([[input[f"intersection{i}_width_X"] (), input[f"slope{i}_width_X"] ()]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "trap", [0, 100])
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in params] for params in concept])
                    concept[0, 0] = ticks.loc[feature, 0]
                    concept[0, 1] = ticks.loc[feature, 0]
                    concept[-1, 2] = ticks.loc[feature, 1000]
                    concept[-1, 3] = ticks.loc[feature, 1000]
                    ax2.plot (*getLines (concept, colours = list ()))
                except (KeyError, TypeError):
                    pass
            elif input.typeFS_width_X () == "gauss":
                try:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_width_X"] ())] for i in range (1, num + 1)])
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


    @render.download (filename = "concept_width_X.json")
    def download_width_X ():
        num = numCards_width_X.get (); ticks = pctWidth_X.get ()
        concepts = dict ()
        if input.typeFS_width_X () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_width_X"] (), input[f"slope{i}_width_X"] ()]
                                    for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_width_X () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), ticks.loc[feature, 0])
                    right = max (np.ceil (concept[-1, 1]), ticks.loc[feature, 1000])
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
            if input.fuzzyBy_width_X () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
        elif input.typeFS_width_X () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_width_X"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            if input.fuzzyBy_width_X () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_width_X"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_width_X () == "dataset":
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
    @reactive.event (input.confirm_width_X)
    def _ ():
        mtx = matrix_X.get (); labels = labelValues_X.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        num = numCards_width_X.get (); ticks = pctWidth_X.get ()
        concepts = dict ()
        if input.typeFS_width_X () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_width_X"] (), input[f"slope{i}_width_X"] ()] for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_width_X () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), ticks.loc[feature, 0])
                    right = max (np.ceil (concept[-1, 1]), ticks.loc[feature, 1000])
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
                globalConcept_X.set (pctConcept)
            if input.fuzzyBy_width_X () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
                globalConcept_X.set (concept)
        elif input.typeFS_width_X () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_width_X"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            globalConcept_X.set (concept)
            if input.fuzzyBy_width_X () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_width_X"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_width_X () == "dataset":
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
            else:
                raise ValueError
        else:
            raise ValueError
        allFV = list (); featureList = itemList.get ()["feature"]
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts_X.set (concepts); numFuzzySets_X.set (concept.shape[0]); fuzzyValues_X.set (np.array (allFV))
        markerStats_X.set (pd.DataFrame ()); mainFuzzyValues_X.set (pd.DataFrame ())
        mainFuzzySets_X.set (pd.DataFrame ()); diffMainFuzzyValues_X.set (pd.DataFrame ())
        noiseRep = plotRangeGlobal_X.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets_X.set (tmp); conceptInfo_X.set ({"method": "width", "direction": input.fuzzyBy_width_X ()})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.start_prop_X)
    def _ ():
        mtx = matrix_X.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_prop_X (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame ({"percents": range (101)}).T
            cutoff = estimateCutoff (dummy, percents).loc["percents"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T,
                   "gauss": np.round (cutoff.tolist ()[1:-1])}
            tempConcepts_prop_X.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix_X.get (); concepts = tempConcepts_prop_X.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]
        num = numCards_prop_X.get (); currNum = trap.shape[0]
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_prop_X", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"intersection{idx}_prop_X", value = trap[i, 0])
            ui.update_numeric (f"slope{idx}_prop_X", value = trap[i, 1])
            ui.update_numeric (f"cutoff{idx}_prop_X", value = gauss[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.panel_conditional (
                        "input.typeFS_prop_X === 'trap'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"intersection{idx}_prop_X", "", step = 0.1, min = 0, max = 100, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Slope (%):",
                            ui.input_numeric (f"slope{idx}_prop_X", "", step = 0.1, min = 0, max = 100, value = trap[i, 1])
                        )
                    ),
                    ui.panel_conditional (
                        "input.typeFS_prop_X === 'gauss'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"cutoff{idx}_prop_X", "", step = 0.1, min = 0, max = 100, value = gauss[i])
                        )
                    ),
                    id = f"FS{idx}_prop_X"
                ),
                selector = f"#FS{i}_prop_X", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_prop_X.set (currNum)


    @render.plot
    def globalDist_prop_X ():
        mtx = matrix_X.get (); feature = input.viewFeature_prop_X ()
        if mtx.empty or len (plotRangeGlobal_X.get ()) != 2:
            return
        mtx = mtx.replace (labelValues_X.get () + [-np.inf, np.inf], np.nan)
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
                pass
        ax.set_xlim (plotRangeGlobal_X.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_prop_X.get (); valueRange = rangeGlobal_X.get (); ticks = pctProp_X.get ()
        if num > 0 and len (valueRange) == 2:
            feature = "ALL" if input.fuzzyBy_prop_X () == "dataset" else feature
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal_X.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            if input.typeFS_prop_X () == "trap":
                try:
                    concept = np.array ([[input[f"intersection{i}_prop_X"] (), input[f"slope{i}_prop_X"] ()] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "trap", [0, 100])
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in params] for params in concept])
                    concept[0, 0] = np.floor (ticks.loc[feature, 0]) - 1
                    concept[0, 1] = np.floor (ticks.loc[feature, 0]) - 1
                    concept[-1, 2] = np.ceil (ticks.loc[feature, 1000]) + 1
                    concept[-1, 3] = np.ceil (ticks.loc[feature, 1000]) + 1
                    ax2.plot (*getLines (concept, colours = list ()))
                except (KeyError, TypeError):
                    pass
            elif input.typeFS_prop_X () == "gauss":
                try:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_prop_X"] ())] for i in range (1, num + 1)])
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


    @render.download (filename = "concept_proportion_X.json")
    def download_prop_X ():
        num = numCards_prop_X.get (); ticks = pctProp_X.get ()
        concepts = dict ()
        if input.typeFS_prop_X () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_prop_X"] (), input[f"slope{i}_prop_X"] ()] for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_prop_X () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), np.floor (ticks.loc[feature, 0]) - 1)
                    right = max (np.ceil (concept[-1, 1]), np.ceil (ticks.loc[feature, 1000]) + 1)
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
            if input.fuzzyBy_prop_X () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 100]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
        elif input.typeFS_prop_X () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_prop_X"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            if input.fuzzyBy_prop_X () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_prop_X"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_prop_X () == "dataset":
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
    @reactive.event (input.confirm_prop_X)
    def _ ():
        mtx = matrix_X.get (); labels = labelValues_X.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        num = numCards_prop_X.get (); ticks = pctProp_X.get ()
        concepts = dict ()
        if input.typeFS_prop_X () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_prop_X"] (), input[f"slope{i}_prop_X"] ()] for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_prop_X () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), np.floor (ticks.loc[feature, 0]) - 1)
                    right = max (np.ceil (concept[-1, 1]), np.ceil (ticks.loc[feature, 1000]) + 1)
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
                globalConcept_X.set (pctConcept)
            if input.fuzzyBy_prop_X () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
                globalConcept_X.set (pctConcept)
        elif input.typeFS_prop_X () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_prop_X"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            if input.fuzzyBy_prop_X () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_prop_X"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_prop_X () == "dataset":
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
            else:
                raise ValueError
        else:
            raise ValueError
        allFV = list (); featureList = itemList.get ()["feature"]
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts_X.set (concepts); numFuzzySets_X.set (concept.shape[0]); fuzzyValues_X.set (np.array (allFV))
        markerStats_X.set (pd.DataFrame ()); mainFuzzyValues_X.set (pd.DataFrame ())
        mainFuzzySets_X.set (pd.DataFrame ()); diffMainFuzzyValues_X.set (pd.DataFrame ())
        noiseRep = plotRangeGlobal_X.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets_X.set (tmp); conceptInfo_X.set ({"method": "prop", "direction": input.fuzzyBy_prop_X ()})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.getMode_X)
    def _ ():
        if len (rangeGlobal_X.get ()) != 2:
            return
        num = numCards_custom_X.get (); currNum = input.numModes_custom_X ()
        xMin, xMax = rangeGlobal_X.get (); step = estimateStep (xMin, xMax); decimal = int (max (0, -np.log10 (step)))
        modes = list (np.round (np.linspace (xMin, xMax, currNum + 2)[1:-1], decimal))
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (f"#custom{idx}_X", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"mode{idx}_custom_X", value = modes[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.layout_columns (
                        "Mode:",
                        ui.input_numeric (f"mode{idx}_custom_X", "", step = step, min = xMin, max = xMax, value = modes[i])
                    ),
                    id = f"custom{idx}_X"
                ),
                selector = f"#custom{i}_X", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_custom_X.set (currNum)


    @reactive.effect
    def _ ():
        if numCards_custom_X.get () != input.numModes_custom_X ():
            return
        elif numCards_custom_X.get () == 0 and input.numModes_custom_X () == 0:
            return
        num = numCards_custom_X.get (); valueRange = rangeGlobal_X.get ()
        modes = [input[f"mode{i + 1}_custom_X"] () for i in range (num)]
        width = np.ediff1d ([valueRange[0]] + modes + [valueRange[1]]) / (2 * np.sqrt (2 * np.log (2)))
        width = np.round (np.array ([[width[i], width[i + 1]] for i in range (len (modes))]), 3)
        centerGlobal_custom_X.set (modes); widthGlobal_custom_X.set (width)


    @render.plot
    def globalModes_custom_X ():
        mtx = matrix_X.get (); labels = labelValues_X.get (); valueRange = rangeGlobal_X.get ()
        modes = centerGlobal_custom_X.get (); width = widthGlobal_custom_X.get ()
        if mtx.empty or len (modes) == 0 or len (width) == 0 or len (modes) != len (width):
            return
        params = np.array ([[modes[i], width[i, 0], width[i, 1]] for i in range (len (modes))])
        xValues = np.linspace (*valueRange, 1000)
        leftFunc = lambda x, mean, std: (x <= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
        rightFunc = lambda x, mean, std: (x >= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
        fig, ax = plt.subplots (figsize = (8, 5)); ax2 = ax.twinx ()
        ax.hist (mtx.melt ()["value"].replace (labels, np.nan).dropna (), bins = 50)
        with np.errstate (divide = "ignore", invalid = "ignore"):
            for p in params:
                yValues = leftFunc (xValues, p[0], p[1]) + rightFunc (xValues, p[0], p[2])
                ax2.plot (xValues, yValues, color = "red")
                ax2.axvline (p[0], color = "black", linestyle = "dashed")
        ax.set_xlim (plotRangeGlobal_X.get ())
        ax2.set_xlim (plotRangeGlobal_X.get ()); ax2.set_ylim ((0, 1.05))
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of values", size = 10)
        ax2.set_ylabel ("fitted fuzzy value", size = 10)
        fig.tight_layout ()
        return fig


    @reactive.effect
    @reactive.event (input.randomSeed_X)
    def _ ():
        seed = np.random.choice (range (1, 101))
        ui.update_numeric ("seed_X", value = int (seed))


    @reactive.effect
    @reactive.event (input.estimate_X)
    def _ ():
        label = labelValues_X.get (); valueRange = rangeGlobal_X.get ()
        crisp = matrix_X.get ().melt ()["value"].replace (label, np.nan).dropna ()
        numValues = input.numValues_X (); numIteration = input.numIteration_X ()
        np.random.seed (input.seed_X ())
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
            centerGlobal_fit_X.set (modes); widthGlobal_fit_X.set (width)
            ui.notification_show ("Estimation Completed", type = "message", duration = 2)


    @render.plot
    @reactive.event (input.estimate_X)
    def globalModes_fit_X ():
        mtx = matrix_X.get (); labels = labelValues_X.get (); valueRange = rangeGlobal_X.get ()
        modes = centerGlobal_fit_X.get (); width = widthGlobal_fit_X.get ()
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
        ax.set_xlim (plotRangeGlobal_X.get ())
        ax2.set_xlim (plotRangeGlobal_X.get ()); ax2.set_ylim ((0, 1.05))
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of values", size = 10)
        ax2.set_ylabel ("scaled density", size = 10)
        fig.tight_layout ()
        return fig


    @reactive.effect
    @reactive.event (input.proceedMode_X)
    def _ ():
        useModeMethod_X.set (input.defMode_X ())
        proceed_X.set (True)


    @reactive.effect
    @reactive.event (input.proceedMode_X)
    def _ ():
        if useModeMethod_X.get () == "custom":
            center = centerGlobal_custom_X.get ()
        elif useModeMethod_X.get () == "fit":
            center = centerGlobal_fit_X.get ()
        else:
            raise ValueError
        num = numCards_mode_X.get ()
        if len (center) == 0:
            return
        for i in range (len (center), num):
            idx = i + 1
            ui.remove_ui (selector = f"#PFC{idx}_X", multiple = False, immediate = False)
        for i in range (num, len (center)):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Concept {idx}"),
                    ui.input_slider (f"numSet{idx}_X", "Number of fuzzy sets:", value = 3, min = 0, max = 5, step = 1),
                    ui.input_slider (f"fctWidth{idx}_X", "Scaling factor for concept width:", min = 0, max = 2, value = 1, step = 0.05),
                    id = f"PFC{idx}_X"
                ),
                selector = f"#PFC{i}_X", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_mode_X.set (len (center))


    @reactive.effect ()
    def _ ():
        if not proceed_X.get ():
            return
        if useModeMethod_X.get () == "custom":
            center = centerGlobal_custom_X.get (); width = widthGlobal_custom_X.get ()
        elif useModeMethod_X.get () == "fit":
            center = centerGlobal_fit_X.get (); width = widthGlobal_fit_X.get ()
        else:
            raise ValueError
        tempPC = list ()
        if len (center) == 0 or len (width) == 0 or len (center) != len (width):
            return
        for idx in range (len (center)):
            miu = center[idx]; sigma = width[idx]; pct = input.pctOverlap_X ()
            num = input[f"numSet{idx + 1}_X"] (); fct = input[f"fctWidth{idx + 1}_X"] ()
            if num == 0 or fct == 0:
                partialConcept = np.array (list ())
            else:
                if num % 2 == 0:
                    leftCoords = [miu + fct * (cutoff + overlap) * sigma[0] for cutoff in np.linspace (-num, -2, int (num / 2))
                                  for overlap in [-pct, pct]]
                    middleCoords = [miu - fct * pct * min (sigma), miu + fct * pct * min (sigma)]
                    rightCoords = [miu + fct * (cutoff + overlap) * sigma[1] for cutoff in np.linspace (2, num, int (num / 2))
                                   for overlap in [-pct, pct]]
                    coords = leftCoords + middleCoords + rightCoords
                else:
                    leftCoords = [miu + fct * (cutoff + overlap) * sigma[0] for cutoff in np.linspace (-num, -1, int (num / 2 + 1))
                                  for overlap in [-pct, pct]]
                    rightCoords = [miu + fct * (cutoff + overlap) * sigma[1] for cutoff in np.linspace (1, num, int (num / 2 + 1))
                                   for overlap in [-pct, pct]]
                    coords = leftCoords + rightCoords
                partialConcept = np.round ([coords[(2 * i - 2):(2 * i + 2)] for i in range (1, num + 1)], 3)
            tempPC.append (partialConcept.tolist ())
        tempPartialConcepts_X.set (tempPC)


    @render.plot
    def partialConcepts_X ():
        if not proceed_X.get ():
            return
        tempPC = tempPartialConcepts_X.get (); valueRange = rangeGlobal_X.get ()
        if len (tempPC) == 0 or len (tempPC) != numCards_mode_X.get ():
            return
        if useModeMethod_X.get () == "custom":
            center = centerGlobal_custom_X.get (); width = widthGlobal_custom_X.get ()
        elif useModeMethod_X.get () == "fit":
            center = centerGlobal_fit_X.get (); width = widthGlobal_fit_X.get ()
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
            axs[idx].set_xlim (plotRangeGlobal_X.get ()); axs[idx].set_ylim ((0, 1.05))
            axs[idx].tick_params (axis = "both", which = "major", labelsize = 8)
            if len (tempPC[idx]) != 0:
                axs[idx].plot (*getLines (np.array (tempPC[idx]), colours = list ()))
            else:
                continue
        fig.supxlabel ("raw value", size = 10); fig.supylabel ("fuzzy value", size = 10)
        fig.tight_layout ()
        return fig


    @render.plot
    def mergedConcept_X ():
        if not proceed_X.get ():
            return
        tempPC = [p for params in tempPartialConcepts_X.get () for p in params]
        if len (tempPC) == 0:
            return
        mtx = matrix_X.get ().replace (labelValues_X.get () + [-np.inf, np.inf], np.nan)
        if mtx.empty or len (rangeGlobal_X.get ()) != 2:
            return
        xMin, xMax = rangeGlobal_X.get (); xMin += 1; xMax -= 1
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
        valueRange = rangeGlobal_X.get ()
        concept[0][0] = valueRange[0]; concept[0][1] = valueRange[0]; concept[-1][2] = valueRange[1]; concept[-1][3] = valueRange[1]
        concept = np.array (concept); tempMergedConcept_X.set (concept)
        feature = input.viewFeature_mode_X ()
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
                pass
        ax2.plot (*getLines (concept, colours = list ()))
        ax.set_xlim (plotRangeGlobal_X.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax2.set_xlim (plotRangeGlobal_X.get ()); ax2.set_ylim ((0, 1.05))
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of values", size = 10)
        ax2.set_ylabel ("fuzzy value", size = 10)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax2.tick_params (axis = "y", which = "major", labelsize = 8)
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_gaussian_mode_X.json")
    def download_mode_X ():
        concept = tempMergedConcept_X.get (); valueRange = pctWidth_X.get ()[[0, 1000]]
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
    @reactive.event (input.confirm_mode_X)
    def _ ():
        mtx = matrix_X.get (); labels = labelValues_X.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        concept = tempMergedConcept_X.get (); valueRange = pctWidth_X.get ()[[0, 1000]]
        concepts = dict ()
        for feature in itemList.get ()["feature"]:
            tmp = concept.copy ()
            left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
            right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
            tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
            concepts[feature] = tmp
        allFV = list (); featureList = itemList.get ()["feature"]; fuzzyValues_X.set (np.array (list ()))
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts_X.set (concepts); globalConcept_X.set (concept)
        numFuzzySets_X.set (concept.shape[0]); fuzzyValues_X.set (np.array (allFV))
        noiseRep = plotRangeGlobal_X.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets_X.set (tmp); conceptInfo_X.set ({"method": "mode", "direction": "dataset"})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.start_fixed_Y)
    def _ ():
        mtx = matrix_Y.get (); labels = labelValues_Y.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_fixed_Y (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame (mtx.melt ()["value"].replace (labels + [-np.inf, np.inf], np.nan).dropna ()).T
            cutoff = estimateCutoff (dummy, percents).loc["value"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)], 3).T,
                   "gauss": np.round (cutoff.tolist ()[1:-1], 3)}
            tempConcepts_fixed_Y.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix_Y.get (); concepts = tempConcepts_fixed_Y.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]
        num = numCards_fixed_Y.get (); currNum = trap.shape[0]; xMin, xMax = rangeGlobal_Y.get ()
        width = xMax - xMin; step = estimateStep (xMin, xMax)
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_fixed_Y", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"intersection{idx}_fixed_Y", value = trap[i, 0])
            ui.update_numeric (f"slope{idx}_fixed_Y", value = trap[i, 1])
            ui.update_numeric (f"cutoff{idx}_fixed_Y", value = gauss[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.panel_conditional (
                        "input.typeFS_fixed_Y === 'trap'",
                        ui.layout_columns (
                            "Cutoff:",
                            ui.input_numeric (f"intersection{idx}_fixed_Y", "", step = step, min = xMin, max = xMax, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Slope:",
                            ui.input_numeric (f"slope{idx}_fixed_Y", "", step = step, min = 0, max = width, value = trap[i, 1])
                        )
                    ),
                    ui.panel_conditional (
                        "input.typeFS_fixed_Y === 'gauss'",
                        ui.layout_columns (
                            "Cutoff:",
                            ui.input_numeric (f"cutoff{idx}_fixed_Y", "", step = step, min = xMin, max = xMax, value = gauss[i])
                        )
                    ),
                    id = f"FS{idx}_fixed_Y"
                ),
                selector = f"#FS{i}_fixed_Y", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_fixed_Y.set (currNum)


    @render.plot
    def globalDist_fixed_Y ():
        mtx = matrix_Y.get (); feature = input.viewFeature_fixed_Y ()
        if mtx.empty or len (plotRangeGlobal_Y.get ()) != 2:
            return
        mtx = mtx.replace (labelValues_Y.get () + [-np.inf, np.inf], np.nan)
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
                pass
        ax.set_xlim (plotRangeGlobal_Y.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_fixed_Y.get (); valueRange = rangeGlobal_Y.get ()
        if num > 0 and len (valueRange) == 2:
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal_Y.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_ylabel ("fuzzy value", size = 10)
            if input.typeFS_fixed_Y () == "trap":
                try:
                    concept = np.array ([[input[f"intersection{i}_fixed_Y"] (), input[f"slope{i}_fixed_Y"] ()]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "trap", valueRange)
                    ax2.plot (*getLines (concept, colours = list ()))
                except TypeError:
                    pass
            elif input.typeFS_fixed_Y () == "gauss":
                try:
                    concept = np.array ([input[f"cutoff{i}_fixed_Y"] () for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", valueRange)
                    xValues = np.linspace (*valueRange, 1000)
                    lines = getCurves (concept, valueRange, colours = list (), setPlateau = True)
                    for idx in range (num + 1):
                        ax2.plot (xValues, lines[idx][0], color = lines[idx][1])
                    ax2.plot ((valueRange[0], valueRange[0]), (0, 1), lines[0][1])
                    ax2.plot ((valueRange[1], valueRange[1]), (1, 0), lines[-1][1])
                except TypeError:
                    pass
            else:
                raise ValueError
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_fixed_parameters_Y.json")
    def download_fixed_Y ():
        num = numCards_fixed_Y.get (); valueRange = pctWidth_Y.get ()[[0, 1000]]
        globalRange = rangeGlobal_Y.get ()
        if input.typeFS_fixed_Y () == "trap":
            concept = np.array ([[input[f"intersection{i}_fixed_Y"] (), input[f"slope{i}_fixed_Y"] ()]
                                 for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "trap", globalRange)
            concepts = dict ()
            for feature in itemList.get ()["feature"]:
                tmp = concept.copy ()
                left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
                right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
                tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
                concepts[feature] = tmp
        elif input.typeFS_fixed_Y () == "gauss":
            concept = np.array ([input[f"cutoff{i}_fixed_Y"] () for i in range (1, num + 1)])
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
    @reactive.event (input.confirm_fixed_Y)
    def _ ():
        mtx = matrix_Y.get (); labels = labelValues_Y.get (); globalRange = rangeGlobal_Y.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        num = numCards_fixed_Y.get (); valueRange = pctWidth_Y.get ()[[0, 1000]]
        if input.typeFS_fixed_Y () == "trap":
            concept = np.array ([[input[f"intersection{i}_fixed_Y"] (), input[f"slope{i}_fixed_Y"] ()]
                                 for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "trap", globalRange)
            concepts = dict ()
            for feature in itemList.get ()["feature"]:
                tmp = concept.copy ()
                left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
                right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
                tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
                concepts[feature] = tmp
        elif input.typeFS_fixed_Y () == "gauss":
            concept = np.array ([input[f"cutoff{i}_fixed_Y"] () for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", globalRange)
            concepts = {feature: concept for feature in itemList.get ()["feature"]}
        else:
            raise ValueError
        allFV = list (); featureList = itemList.get ()["feature"]
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts_Y.set (concepts); globalConcept_Y.set (concept)
        numFuzzySets_Y.set (concept.shape[0]); fuzzyValues_Y.set (np.array (allFV))
        markerStats_Y.set (pd.DataFrame ()); mainFuzzyValues_Y.set (pd.DataFrame ())
        mainFuzzySets_Y.set (pd.DataFrame ()); diffMainFuzzyValues_Y.set (pd.DataFrame ())
        noiseRep = plotRangeGlobal_Y.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets_Y.set (tmp); conceptInfo_Y.set ({"method": "fixed", "direction": "dataset"})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.start_width_Y)
    def _ ():
        mtx = matrix_Y.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_width_Y (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame ({"percents": range (101)}).T
            cutoff = estimateCutoff (dummy, percents).loc["percents"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T,
                   "gauss": np.round (cutoff.tolist ()[1:-1])}
            tempConcepts_width_Y.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix_Y.get (); concepts = tempConcepts_width_Y.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]
        num = numCards_width_Y.get (); currNum = trap.shape[0]
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_width_Y", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"intersection{idx}_width_Y", value = trap[i, 0])
            ui.update_numeric (f"slope{idx}_width_Y", value = trap[i, 1])
            ui.update_numeric (f"cutoff{idx}_width_Y", value = gauss[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.panel_conditional (
                        "input.typeFS_width_Y === 'trap'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"intersection{idx}_width_Y", "", step = 0.1, min = 0, max = 100, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Slope (%):",
                            ui.input_numeric (f"slope{idx}_width_Y", "", step = 0.1, min = 0, max = 100, value = trap[i, 1])
                        )
                    ),
                    ui.panel_conditional (
                        "input.typeFS_width_Y === 'gauss'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"cutoff{idx}_width_Y", "", step = 0.1, min = 0, max = 100, value = gauss[i])
                        )
                    ),
                    id = f"FS{idx}_width_Y"
                ),
                selector = f"#FS{i}_width_Y", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_width_Y.set (currNum)


    @render.plot
    def globalDist_width_Y ():
        mtx = matrix_Y.get (); feature = input.viewFeature_width_Y ()
        if mtx.empty or len (plotRangeGlobal_Y.get ()) != 2:
            return
        mtx = mtx.replace (labelValues_Y.get () + [-np.inf, np.inf], np.nan)
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
                pass
        ax.set_xlim (plotRangeGlobal_Y.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_width_Y.get (); valueRange = rangeGlobal_Y.get (); ticks = pctWidth_Y.get ()
        if num > 0 and len (valueRange) == 2:
            feature = "ALL" if input.fuzzyBy_width_Y () == "dataset" else feature
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal_Y.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            if input.typeFS_width_Y () == "trap":
                try:
                    concept = np.array ([[input[f"intersection{i}_width_Y"] (), input[f"slope{i}_width_Y"] ()]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "trap", [0, 100])
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in params] for params in concept])
                    concept[0, 0] = ticks.loc[feature, 0]; concept[0, 1] = ticks.loc[feature, 0]
                    concept[-1, 2] = ticks.loc[feature, 1000]; concept[-1, 3] = ticks.loc[feature, 1000]
                    ax2.plot (*getLines (concept, colours = list ()))
                except (KeyError, TypeError):
                    pass
            elif input.typeFS_width_Y () == "gauss":
                try:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_width_Y"] ())]
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


    @render.download (filename = "concept_width_Y.json")
    def download_width_Y ():
        num = numCards_width_Y.get (); ticks = pctWidth_Y.get ()
        concepts = dict ()
        if input.typeFS_width_Y () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_width_Y"] (), input[f"slope{i}_width_Y"] ()]
                                    for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_width_Y () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), ticks.loc[feature, 0])
                    right = max (np.ceil (concept[-1, 1]), ticks.loc[feature, 1000])
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
            if input.fuzzyBy_width_Y () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
        elif input.typeFS_width_Y () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_width_Y"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            if input.fuzzyBy_width_Y () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_width_Y"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_width_Y () == "dataset":
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
    @reactive.event (input.confirm_width_Y)
    def _ ():
        mtx = matrix_Y.get (); labels = labelValues_Y.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        num = numCards_width_Y.get (); ticks = pctWidth_Y.get ()
        concepts = dict ()
        if input.typeFS_width_Y () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_width_Y"] (), input[f"slope{i}_width_Y"] ()]
                                    for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_width_Y () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), ticks.loc[feature, 0])
                    right = max (np.ceil (concept[-1, 1]), ticks.loc[feature, 1000])
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
                globalConcept_Y.set (pctConcept)
            if input.fuzzyBy_width_Y () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
                globalConcept_Y.set (concept)
        elif input.typeFS_width_Y () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_width_Y"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].to_list ())
            globalConcept_Y.set (concept)
            if input.fuzzyBy_width_Y () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_width_Y"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].to_list ())
                    concepts[feature] = concept
            elif input.fuzzyBy_width_Y () == "dataset":
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
            else:
                raise ValueError
        else:
            raise ValueError
        allFV = list (); featureList = itemList.get ()["feature"]
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts_Y.set (concepts); numFuzzySets_Y.set (concept.shape[0]); fuzzyValues_Y.set (np.array (allFV))
        markerStats_Y.set (pd.DataFrame ()); mainFuzzyValues_Y.set (pd.DataFrame ())
        mainFuzzySets_Y.set (pd.DataFrame ()); diffMainFuzzyValues_Y.set (pd.DataFrame ())
        noiseRep = plotRangeGlobal_Y.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets_Y.set (tmp); conceptInfo_Y.set ({"method": "width", "direction": input.fuzzyBy_width_Y ()})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.start_prop_Y)
    def _ ():
        mtx = matrix_Y.get ()
        if mtx.empty:
            return
        with ui.Progress () as p:
            p.set (message = "Deriving Fuzzy Concepts", detail = "This will take a while...")
            numFS = input.numFS_prop_Y (); percents = [1 / numFS] * numFS
            dummy = pd.DataFrame ({"percents": range (101)}).T
            cutoff = estimateCutoff (dummy, percents).loc["percents"]
            slope = cutoff.diff ().iloc[1:].min () / 4
            tmp = {"trap": np.round ([cutoff.tolist ()[1:-1], [slope] * (numFS - 1)]).T,
                   "gauss": np.round (cutoff.tolist ()[1:-1])}
            tempConcepts_prop_Y.set (tmp)
        ui.notification_show ("Derivation Completed", type = "message", duration = 2)


    @reactive.effect
    def _ ():
        mtx = matrix_Y.get (); concepts = tempConcepts_prop_Y.get ()
        if mtx.empty or not concepts:
            return
        trap = concepts["trap"]; gauss = concepts["gauss"]
        num = numCards_prop_Y.get (); currNum = trap.shape[0]
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (selector = f"#FS{idx}_prop_Y", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"intersection{idx}_prop_Y", value = trap[i, 0])
            ui.update_numeric (f"slope{idx}_prop_Y", value = trap[i, 1])
            ui.update_numeric (f"cutoff{idx}_prop_Y", value = gauss[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Fuzzy Set {idx}"),
                    ui.panel_conditional (
                        "input.typeFS_prop_Y === 'trap'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"intersection{idx}_prop_Y", "", step = 0.1, min = 0, max = 100, value = trap[i, 0])
                        ),
                        ui.layout_columns (
                            "Slope (%):",
                            ui.input_numeric (f"slope{idx}_prop_Y", "", step = 0.1, min = 0, max = 100, value = trap[i, 1])
                        )
                    ),
                    ui.panel_conditional (
                        "input.typeFS_prop_Y === 'gauss'",
                        ui.layout_columns (
                            "Cutoff (%):",
                            ui.input_numeric (f"cutoff{idx}_prop_Y", "", step = 0.1, min = 0, max = 100, value = gauss[i])
                        )
                    ),
                    id = f"FS{idx}_prop_Y"
                ),
                selector = f"#FS{i}_prop_Y", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_prop_Y.set (currNum)


    @render.plot
    def globalDist_prop_Y ():
        mtx = matrix_Y.get (); feature = input.viewFeature_prop_Y ()
        if mtx.empty or len (plotRangeGlobal_Y.get ()) != 2:
            return
        mtx = mtx.replace (labelValues_Y.get () + [-np.inf, np.inf], np.nan)
        fig, ax = plt.subplots (1, figsize = (8, 5))
        if feature == "ALL":
            pltData = mtx.melt ()["value"]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
            ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
            del pltData
        else:
            try:
                pltData = mtx.loc[feature]; pctUnlabelled = "{:.1%}".format (len (pltData.dropna ()) / len (pltData))
                ax.hist (pltData.dropna (), bins = 50, color = "lightgray")
            except KeyError:
                pctUnlabelled = "0.0%"
                pass
        ax.set_xlim (plotRangeGlobal_Y.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax.set_xlabel ("raw value", size = 10)
        ax.set_ylabel ("number of unlabelled values", size = 10)
        num = numCards_prop_Y.get (); valueRange = rangeGlobal_Y.get (); ticks = pctProp_Y.get ()
        if num > 0 and len (valueRange) == 2:
            feature = "ALL" if input.fuzzyBy_prop_Y () == "dataset" else feature
            ax2 = ax.twinx ()
            ax2.set_xlim (plotRangeGlobal_Y.get ()); ax2.set_ylim ((0, 1.05))
            ax2.tick_params (axis = "y", which = "major", labelsize = 8)
            ax2.set_xlabel ("raw value", size = 10); ax2.set_ylabel ("fuzzy value", size = 10)
            if input.typeFS_prop_Y () == "trap":
                try:
                    concept = np.array ([[input[f"intersection{i}_prop_Y"] (), input[f"slope{i}_prop_Y"] ()]
                                         for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "trap", [0, 100])
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in params] for params in concept])
                    concept[0, 0] = np.floor (ticks.loc[feature, 0]) - 1; concept[0, 1] = np.floor (ticks.loc[feature, 0]) - 1
                    concept[-1, 2] = np.ceil (ticks.loc[feature, 1000]) + 1; concept[-1, 3] = np.ceil (ticks.loc[feature, 1000]) + 1
                    ax2.plot (*getLines (concept, colours = list ()))
                except (KeyError, TypeError):
                    pass
            elif input.typeFS_prop_Y () == "gauss":
                try:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_prop_Y"] ())] for i in range (1, num + 1)])
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


    @render.download (filename = "concept_proportion_Y.json")
    def download_prop_Y ():
        num = numCards_prop_Y.get (); ticks = pctProp_Y.get ()
        concepts = dict ()
        if input.typeFS_prop_Y () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_prop_Y"] (), input[f"slope{i}_prop_Y"] ()] for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_prop_Y () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), np.floor (ticks.loc[feature, 0]) - 1)
                    right = max (np.ceil (concept[-1, 1]), np.ceil (ticks.loc[feature, 1000]) + 1)
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
            if input.fuzzyBy_prop_Y () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 100]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
        elif input.typeFS_prop_Y () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_prop_Y"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            if input.fuzzyBy_prop_Y () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_prop_Y"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_prop_Y () == "dataset":
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
    @reactive.event (input.confirm_prop_Y)
    def _ ():
        mtx = matrix_Y.get (); labels = labelValues_Y.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        num = numCards_prop_Y.get (); ticks = pctProp_Y.get ()
        concepts = dict ()
        if input.typeFS_prop_Y () == "trap":
            pctConcept = np.array ([[input[f"intersection{i}_prop_Y"] (), input[f"slope{i}_prop_Y"] ()] for i in range (1, num + 1)])
            pctConcept = getFinalConcept (pctConcept, "trap", [0, 100])
            if input.fuzzyBy_prop_Y () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([[ticks.loc[feature, int (10 * i)] for i in p] for p in pctConcept])
                    left = min (np.floor (concept[0, 2]), np.floor (ticks.loc[feature, 0]) - 1)
                    right = max (np.ceil (concept[-1, 1]), np.ceil (ticks.loc[feature, 1000]) + 1)
                    concept[0, 0] = left; concept[0, 1] = left; concept[-1, 2] = right; concept[-1, 3] = right
                    concepts[feature] = concept
                globalConcept_Y.set (pctConcept)
            if input.fuzzyBy_prop_Y () == "dataset":
                concept = np.array ([[ticks.loc["ALL", int (10 * i)] for i in p] for p in pctConcept])
                concept[0, 0] = ticks.loc["ALL", 0]; concept[0, 1] = ticks.loc["ALL", 0]
                concept[-1, 2] = ticks.loc["ALL", 1000]; concept[-1, 3] = ticks.loc["ALL", 1000]
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
                globalConcept_Y.set (concept)
        elif input.typeFS_prop_Y () == "gauss":
            concept = np.array ([ticks.loc["ALL", int (10 * input[f"cutoff{i}_prop_Y"] ())] for i in range (1, num + 1)])
            concept = getFinalConcept (concept, "gauss", ticks.loc["ALL", [0, 1000]].tolist ())
            globalConcept_Y.set (concept)
            if input.fuzzyBy_prop_Y () == "feature":
                for feature in itemList.get ()["feature"]:
                    concept = np.array ([ticks.loc[feature, int (10 * input[f"cutoff{i}_prop_Y"] ())] for i in range (1, num + 1)])
                    concept = getFinalConcept (concept, "gauss", ticks.loc[feature, [0, 1000]].tolist ())
                    concepts[feature] = concept
            elif input.fuzzyBy_prop_Y () == "dataset":
                concepts = {feature: concept for feature in itemList.get ()["feature"]}
            else:
                raise ValueError
        else:
            raise ValueError
        allFV = list (); featureList = itemList.get ()["feature"]
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts_Y.set (concepts); numFuzzySets_Y.set (concept.shape[0]); fuzzyValues_Y.set (np.array (allFV))
        markerStats_Y.set (pd.DataFrame ()); mainFuzzyValues_Y.set (pd.DataFrame ())
        mainFuzzySets_Y.set (pd.DataFrame ()); diffMainFuzzyValues_Y.set (pd.DataFrame ())
        noiseRep = plotRangeGlobal_Y.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets_Y.set (tmp); conceptInfo_Y.set ({"method": "prop", "direction": input.fuzzyBy_prop_Y ()})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @reactive.effect
    @reactive.event (input.getMode_Y)
    def _ ():
        if len (rangeGlobal_Y.get ()) != 2:
            return
        num = numCards_custom_Y.get (); currNum = input.numModes_custom_Y ()
        xMin, xMax = rangeGlobal_Y.get (); step = estimateStep (xMin, xMax); decimal = int (max (0, -np.log10 (step)))
        modes = list (np.round (np.linspace (xMin, xMax, currNum + 2)[1:-1], decimal))
        for i in range (currNum, num):
            idx = i + 1
            ui.remove_ui (f"#custom{idx}_Y", multiple = False, immediate = False)
        for i in range (min (currNum, num)):
            idx = i + 1
            ui.update_numeric (f"mode{idx}_custom_Y", value = modes[i])
        for i in range (num, currNum):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.layout_columns (
                        "Mode:",
                        ui.input_numeric (f"mode{idx}_custom_Y", "", step = step, min = xMin, max = xMax, value = modes[i])
                    ),
                    id = f"custom{idx}_Y"
                ),
                selector = f"#custom{i}_Y", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_custom_Y.set (currNum)


    @reactive.effect
    def _ ():
        if numCards_custom_Y.get () != input.numModes_custom_Y ():
            return
        elif numCards_custom_Y.get () == 0 and input.numModes_custom_Y () == 0:
            return
        num = numCards_custom_Y.get (); valueRange = rangeGlobal_Y.get ()
        modes = [input[f"mode{i + 1}_custom_Y"] () for i in range (num)]
        width = np.ediff1d ([valueRange[0]] + modes + [valueRange[1]]) / (2 * np.sqrt (2 * np.log (2)))
        width = np.round (np.array ([[width[i], width[i + 1]] for i in range (len (modes))]), 3)
        centerGlobal_custom_Y.set (modes); widthGlobal_custom_Y.set (width)


    @render.plot
    def globalModes_custom_Y ():
        mtx = matrix_Y.get (); labels = labelValues_Y.get (); valueRange = rangeGlobal_Y.get ()
        modes = centerGlobal_custom_Y.get (); width = widthGlobal_custom_Y.get ()
        if mtx.empty or len (modes) == 0 or len (width) == 0 or len (modes) != len (width):
            return
        params = np.array ([[modes[i], width[i, 0], width[i, 1]] for i in range (len (modes))])
        xValues = np.linspace (*valueRange, 1000)
        leftFunc = lambda x, mean, std: (x <= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
        rightFunc = lambda x, mean, std: (x >= mean) * np.exp (-(x - mean) ** 2 / (2 * std ** 2))
        fig, ax = plt.subplots (figsize = (8, 5)); ax2 = ax.twinx ()
        ax.hist (mtx.melt ()["value"].replace (labels, np.nan).dropna (), bins = 50)
        with np.errstate (divide = "ignore", invalid = "ignore"):
            for p in params:
                yValues = leftFunc (xValues, p[0], p[1]) + rightFunc (xValues, p[0], p[2])
                ax2.plot (xValues, yValues, color = "red")
                ax2.axvline (p[0], color = "black", linestyle = "dashed")
        ax.set_xlim (plotRangeGlobal_Y.get ())
        ax2.set_xlim (plotRangeGlobal_Y.get ()); ax2.set_ylim ((0, 1.05))
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of values", size = 10)
        ax2.set_ylabel ("fitted fuzzy value", size = 10)
        fig.tight_layout ()
        return fig


    @reactive.effect
    @reactive.event (input.randomSeed_Y)
    def _ ():
        seed = np.random.choice (range (1, 101))
        ui.update_numeric ("seed_Y", value = int (seed))


    @reactive.effect
    @reactive.event (input.estimate_Y)
    def _ ():
        label = labelValues_Y.get (); valueRange = rangeGlobal_Y.get ()
        crisp = matrix_Y.get ().melt ()["value"].replace (label, np.nan).dropna ()
        numValues = input.numValues_Y (); numIteration = input.numIteration_Y ()
        np.random.seed (input.seed_Y ())
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
            centerGlobal_fit_Y.set (modes); widthGlobal_fit_Y.set (width)
            ui.notification_show ("Estimation Completed", type = "message", duration = 2)


    @render.plot
    @reactive.event (input.estimate_Y)
    def globalModes_fit_Y ():
        mtx = matrix_Y.get (); labels = labelValues_Y.get (); valueRange = rangeGlobal_Y.get ()
        modes = centerGlobal_fit_Y.get (); width = widthGlobal_fit_Y.get ()
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
        ax.set_xlim (plotRangeGlobal_Y.get ())
        ax2.set_xlim (plotRangeGlobal_Y.get ()); ax2.set_ylim ((0, 1.05))
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of values", size = 10)
        ax2.set_ylabel ("scaled density", size = 10)
        fig.tight_layout ()
        return fig


    @reactive.effect
    @reactive.event (input.proceedMode_Y)
    def _ ():
        useModeMethod_Y.set (input.defMode_Y ())
        proceed_Y.set (True)


    @reactive.effect
    @reactive.event (input.proceedMode_Y)
    def _ ():
        if useModeMethod_Y.get () == "custom":
            center = centerGlobal_custom_Y.get ()
        elif useModeMethod_Y.get () == "fit":
            center = centerGlobal_fit_Y.get ()
        else:
            raise ValueError
        num = numCards_mode_Y.get ()
        if len (center) == 0:
            return
        for i in range (len (center), num):
            idx = i + 1
            ui.remove_ui (selector = f"#PFC{idx}_Y", multiple = False, immediate = False)
        for i in range (num, len (center)):
            idx = i + 1
            ui.insert_ui (
                ui.card (
                    ui.card_header (f"Concept {idx}"),
                    ui.input_slider (f"numSet{idx}_Y", "Number of fuzzy sets:", value = 3, min = 0, max = 5, step = 1),
                    ui.input_slider (f"fctWidth{idx}_Y", "Scaling factor for concept width:", min = 0, max = 2, value = 1, step = 0.05),
                    id = f"PFC{idx}_Y"
                ),
                selector = f"#PFC{i}_Y", where = "afterEnd", multiple = False, immediate = False
            )
        numCards_mode_Y.set (len (center))


    @reactive.effect ()
    def _ ():
        if not proceed_Y.get ():
            return
        if useModeMethod_Y.get () == "custom":
            center = centerGlobal_custom_Y.get (); width = widthGlobal_custom_Y.get ()
        elif useModeMethod_Y.get () == "fit":
            center = centerGlobal_fit_Y.get (); width = widthGlobal_fit_Y.get ()
        else:
            raise ValueError
        tempPC = list ()
        if len (center) == 0 or len (width) == 0 or len (center) != len (width):
            return
        for idx in range (len (center)):
            miu = center[idx]; sigma = width[idx]; pct = input.pctOverlap_Y ()
            num = input[f"numSet{idx + 1}_Y"] (); fct = input[f"fctWidth{idx + 1}_Y"] ()
            if num == 0 or fct == 0:
                partialConcept = np.array (list ())
            else:
                if num % 2 == 0:
                    leftCoords = [miu + fct * (cutoff + overlap) * sigma[0] for cutoff in np.linspace (-num, -2, int (num / 2))
                                  for overlap in [-pct, pct]]
                    middleCoords = [miu - fct * pct * min (sigma), miu + fct * pct * min (sigma)]
                    rightCoords = [miu + fct * (cutoff + overlap) * sigma[1] for cutoff in np.linspace (2, num, int (num / 2))
                                   for overlap in [-pct, pct]]
                    coords = leftCoords + middleCoords + rightCoords
                else:
                    leftCoords = [miu + fct * (cutoff + overlap) * sigma[0] for cutoff in np.linspace (-num, -1, int (num / 2 + 1))
                                  for overlap in [-pct, pct]]
                    rightCoords = [miu + fct * (cutoff + overlap) * sigma[1] for cutoff in np.linspace (1, num, int (num / 2 + 1))
                                   for overlap in [-pct, pct]]
                    coords = leftCoords + rightCoords
                partialConcept = np.round ([coords[(2 * i - 2):(2 * i + 2)] for i in range (1, num + 1)], 3)
            tempPC.append (partialConcept.tolist ())
        tempPartialConcepts_Y.set (tempPC)


    @render.plot
    def partialConcepts_Y ():
        if not proceed_Y.get ():
            return
        tempPC = tempPartialConcepts_Y.get (); valueRange = rangeGlobal_Y.get ()
        if len (tempPC) == 0 or len (tempPC) != numCards_mode_Y.get ():
            return
        if useModeMethod_Y.get () == "custom":
            center = centerGlobal_custom_Y.get (); width = widthGlobal_custom_Y.get ()
        elif useModeMethod_Y.get () == "fit":
            center = centerGlobal_fit_Y.get (); width = widthGlobal_fit_Y.get ()
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
            axs[idx].set_xlim (plotRangeGlobal_Y.get ()); axs[idx].set_ylim ((0, 1.05))
            axs[idx].tick_params (axis = "both", which = "major", labelsize = 8)
            if len (tempPC[idx]) != 0:
                axs[idx].plot (*getLines (np.array (tempPC[idx]), colours = list ()))
            else:
                continue
        fig.supxlabel ("raw value", size = 10); fig.supylabel ("fuzzy value", size = 10)
        fig.tight_layout ()
        return fig


    @render.plot
    def mergedConcept_Y ():
        if not proceed_Y.get ():
            return
        tempPC = [p for params in tempPartialConcepts_Y.get () for p in params]
        if len (tempPC) == 0:
            return
        mtx = matrix_Y.get ().replace (labelValues_Y.get () + [-np.inf, np.inf], np.nan)
        if mtx.empty or len (rangeGlobal_Y.get ()) != 2:
            return
        xMin, xMax = rangeGlobal_Y.get (); xMin += 1; xMax -= 1
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
        valueRange = rangeGlobal_Y.get ()
        concept[0][0] = valueRange[0]; concept[0][1] = valueRange[0]; concept[-1][2] = valueRange[1]; concept[-1][3] = valueRange[1]
        concept = np.array (concept); tempMergedConcept_Y.set (concept)
        feature = input.viewFeature_mode_Y ()
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
                pass
        ax2.plot (*getLines (concept, colours = list ()))
        ax.set_xlim (plotRangeGlobal_Y.get ()); ax.set_title (f"unlabelled values - {pctUnlabelled}", size = 15)
        ax2.set_xlim (plotRangeGlobal_Y.get ()); ax2.set_ylim ((0, 1.05))
        ax.set_xlabel ("raw value", size = 10); ax.set_ylabel ("number of values", size = 10)
        ax2.set_ylabel ("fuzzy value", size = 10)
        ax.tick_params (axis = "both", which = "major", labelsize = 8)
        ax2.tick_params (axis = "y", which = "major", labelsize = 8)
        fig.tight_layout ()
        return fig


    @render.download (filename = "concept_gaussian_mode_Y.json")
    def download_mode_Y ():
        concept = tempMergedConcept_Y.get (); valueRange = pctWidth_Y.get ()[[0, 1000]]
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
    @reactive.event (input.confirm_mode_Y)
    def _ ():
        mtx = matrix_Y.get (); labels = labelValues_Y.get ()
        if mtx.empty:
            return
        fuzzyParams = {"addIndicator": len (labels) != 0, "indicateValue": labels}
        concept = tempMergedConcept_Y.get (); valueRange = pctWidth_Y.get ()[[0, 1000]]
        concepts = dict ()
        for feature in itemList.get ()["feature"]:
            tmp = concept.copy ()
            left = min (np.floor (concept[0, 2]), valueRange.loc[feature, 0])
            right = max (np.ceil (concept[-1, 1]), valueRange.loc[feature, 1000])
            tmp[0, 0] = left; tmp[0, 1] = left; tmp[-1, 2] = right; tmp[-1, 3] = right
            concepts[feature] = tmp
        allFV = list (); featureList = itemList.get ()["feature"]; fuzzyValues_Y.set (np.array (list ()))
        with ui.Progress (min = 0, max = mtx.shape[0] - 1) as p:
            p.set (message = "Fuzzification Running", detail = "This will take a while...")
            for idx in range (mtx.shape[0]):
                p.set (idx, message = "Fuzzifying")
                feature = featureList[idx]
                memberships = fuzzify (mtx.loc[feature], concepts[feature], furtherParams = fuzzyParams)
                allFV.append (memberships.round (3).to_numpy ())
        allConcepts_Y.set (concepts); globalConcept_Y.set (concept)
        numFuzzySets_Y.set (concept.shape[0]); fuzzyValues_Y.set (np.array (allFV))
        noiseRep = plotRangeGlobal_Y.get (); noiseName = [f"FS0_{noiseRep[0]}", f"FS0_{noiseRep[1]}"]
        tmp = list ()
        for col in memberships.columns:
            if col == noiseName[0]:
                tmp.append ("FS0_noiseLeft")
            elif col == noiseName[1]:
                tmp.append ("FS0_noiseRight")
            else:
                tmp.append (col)
        nameFuzzySets_Y.set (tmp); conceptInfo_Y.set ({"method": "mode", "direction": "dataset"})
        ui.notification_show ("Fuzzification Completed", type = "message", duration = 2)


    @render.plot
    @reactive.event (input.updateVolcano)
    def volcano ():
        mtx_X = matrix_X.get (); mtx_Y = matrix_Y.get (); items = itemList.get ()
        if mtx_X.empty or mtx_Y.empty or len (items["feature"]) == 0 or len (items["sample"]) == 0:
            return
        xRange = rangeGlobal_X.get (); yRange = rangeGlobal_Y.get (); feature = input.viewFeature ()
        if len (xRange) != 2 or len (yRange) != 2:
            return
        mtx_X = mtx_X.replace (labelValues_X.get () + [-np.inf, np.inf], np.nan)
        mtx_Y = mtx_Y.replace (labelValues_Y.get () + [-np.inf, np.inf], np.nan)
        if feature == "ALL":
            pltData = pd.DataFrame ({"X": mtx_X.loc[items["feature"], items["sample"]].melt ()["value"],
                                     "Y": mtx_Y.loc[items["feature"], items["sample"]].melt ()["value"]})
        else:
            pltData = pd.DataFrame ({"X": mtx_X.loc[feature, items["sample"]],
                                     "Y": mtx_Y.loc[feature, items["sample"]]})
        del mtx_X, mtx_Y
        intersection_X = list ()
        if input.viewConcept_X () == "fixed":
            num = numCards_fixed_X.get ()
            if num > 0:
                if input.typeFS_fixed_X () == "trap":
                    intersection_X = [input[f"intersection{i}_fixed_X"] () for i in range (1, num + 1)]
                elif input.typeFS_fixed_X () == "gauss":
                    intersection_X = [input[f"cutoff{i}_fixed_X"] () for i in range (1, num + 1)]
                else:
                    raise ValueError
            elif tempConcepts_fixed_X.get ():
                if input.typeFS_fixed_X () == "trap":
                    intersection_X = tempConcepts_fixed_X.get ()["trap"][:, 0].tolist ()
                elif input.typeFS_fixed_X () == "gauss":
                    intersection_X = tempConcepts_fixed_X.get ()["gauss"].tolist ()
                else:
                    raise ValueError
        elif input.viewConcept_X () == "width":
            num = numCards_width_X.get (); ticks = pctWidth_X.get ()
            if num > 0:
                if input.typeFS_width_X () == "trap":
                    intersection_X = [ticks.loc[feature, int (10 * input[f"intersection{i}_width_X"] ())] for i in range (1, num + 1)]
                elif input.typeFS_width_X () == "gauss":
                    intersection_X = [ticks.loc[feature, int (10 * input[f"cutoff{i}_width_X"] ())] for i in range (1, num + 1)]
                else:
                    raise ValueError
            elif tempConcepts_width_X.get ():
                if input.typeFS_width_X () == "trap":
                    intersection_X = [ticks.loc[feature, int (10 * x)] for x in tempConcepts_width_X.get ()["trap"][:, 0]]
                elif input.typeFS_width_X () == "gauss":
                    intersection_X = [ticks.loc[feature, int (10 * x)] for x in tempConcepts_width_X.get ()["gauss"]]
                else:
                    raise ValueError
        elif input.viewConcept_X () == "prop":
            num = numCards_prop_X.get (); ticks = pctProp_X.get ()
            if num > 0:
                if input.typeFS_prop_X () == "trap":
                    intersection_X = [ticks.loc[feature, int (10 * input[f"intersection{i}_prop_X"] ())] for i in range (1, num + 1)]
                elif input.typeFS_prop_X () == "gauss":
                    intersection_X = [ticks.loc[feature, int (10 * input[f"cutoff{i}_prop_X"] ())] for i in range (1, num + 1)]
                else:
                    raise ValueError
            elif tempConcepts_prop_X.get ():
                if input.typeFS_prop_X () == "trap":
                    intersection_X = [ticks.loc[feature, int (10 * x)] for x in tempConcepts_prop_X.get ()["trap"][:, 0]]
                elif input.typeFS_prop_X () == "gauss":
                    intersection_X = [ticks.loc[feature, int (10 * x)] for x in tempConcepts_prop_X.get ()["gauss"]]
                else:
                    raise ValueError
        elif input.viewConcept_X () == "mode":
            intersection_X = getIntersection (tempMergedConcept_X.get (), "trap", xRange)[1:-1]
        else:
            raise ValueError
        intersection_X = [xRange[0]] + intersection_X + [xRange[1]]
        intersection_Y = list ()
        if input.viewConcept_Y () == "fixed":
            num = numCards_fixed_Y.get ()
            if num > 0:
                if input.typeFS_fixed_Y () == "trap":
                    intersection_Y = [input[f"intersection{i}_fixed_Y"] () for i in range (1, num + 1)]
                elif input.typeFS_fixed_Y () == "gauss":
                    intersection_Y = [input[f"cutoff{i}_fixed_Y"] () for i in range (1, num + 1)]
                else:
                    raise ValueError
            elif tempConcepts_fixed_Y.get ():
                if input.typeFS_fixed_Y () == "trap":
                    intersection_Y = tempConcepts_fixed_Y.get ()["trap"][:, 0].tolist ()
                elif input.typeFS_fixed_Y () == "gauss":
                    intersection_Y = tempConcepts_fixed_Y.get ()["gauss"].tolist ()
                else:
                    raise ValueError
        elif input.viewConcept_Y () == "width":
            num = numCards_width_Y.get (); ticks = pctWidth_Y.get ()
            if num > 0:
                if input.typeFS_width_Y () == "trap":
                    intersection_Y = [ticks.loc[feature, int (10 * input[f"intersection{i}_width_Y"] ())] for i in range (1, num + 1)]
                elif input.typeFS_width_Y () == "gauss":
                    intersection_Y = [ticks.loc[feature, int (10 * input[f"cutoff{i}_width_Y"] ())] for i in range (1, num + 1)]
                else:
                    raise ValueError
            elif tempConcepts_width_Y.get ():
                if input.typeFS_width_Y () == "trap":
                    intersection_Y = [ticks.loc[feature, int (10 * x)] for x in tempConcepts_width_Y.get ()["trap"][:, 0]]
                elif input.typeFS_width_Y () == "gauss":
                    intersection_Y = [ticks.loc[feature, int (10 * x)] for x in tempConcepts_width_Y.get ()["gauss"]]
                else:
                    raise ValueError
        elif input.viewConcept_Y () == "prop":
            num = numCards_prop_Y.get (); ticks = pctProp_Y.get ()
            if num > 0:
                if input.typeFS_prop_Y () == "trap":
                    intersection_Y = [ticks.loc[feature, int (10 * input[f"intersection{i}_prop_Y"] ())] for i in range (1, num + 1)]
                elif input.typeFS_prop_Y () == "gauss":
                    intersection_Y = [ticks.loc[feature, int (10 * input[f"cutoff{i}_prop_Y"] ())] for i in range (1, num + 1)]
                else:
                    raise ValueError
            elif tempConcepts_prop_Y.get ():
                if input.typeFS_prop_Y () == "trap":
                    intersection_Y = [ticks.loc[feature, int (10 * x)] for x in tempConcepts_prop_Y.get ()["trap"][:, 0]]
                elif input.typeFS_prop_Y () == "gauss":
                    intersection_Y = [ticks.loc[feature, int (10 * x)] for x in tempConcepts_prop_Y.get ()["gauss"]]
                else:
                    raise ValueError
        elif input.viewConcept_X () == "mode":
            intersection_Y = getIntersection (tempMergedConcept_Y.get (), "trap", yRange)[1:-1]
        else:
            raise ValueError
        intersection_Y = [yRange[0]] + intersection_Y + [yRange [1]]
        pltData = pltData.dropna (); num = pltData.shape[0]
        coords = [[(intersection_X[i] + intersection_X[i + 1]) / 2, (intersection_Y[j] + intersection_Y[j + 1]) / 2]
                  for j in range (len (intersection_Y) - 1) for i in range (len (intersection_X) - 1)]
        labels = [pltData.loc[(pltData["X"] >= intersection_X[i]) & (pltData["X"] < intersection_X[i + 1]) &
                              (pltData["Y"] >= intersection_Y[j]) & (pltData["Y"] < intersection_Y[j + 1])].shape[0]
                  for j in range (len (intersection_Y) - 1) for i in range (len (intersection_X) - 1)]
        labels = [x / num for x in labels]
        fig, ax = plt.subplots (1, figsize = (10, 10))
        sns.scatterplot (pltData, x = "X", y = "Y", color = "lightgray", size = 3, legend = None, ax = ax)
        ax.set_xlim (xRange); ax.set_ylim (yRange)
        for val in intersection_X[1:-1]:
            ax.axvline (val, color = "black", linestyle = "dashed")
        for val in intersection_Y[1:-1]:
            ax.axhline (val, color = "black", linestyle = "dashed")
        if len (intersection_X) > 2 or len (intersection_Y) > 2:
            for idx in range (len (coords)):
                ax.text (x = coords[idx][0], y = coords[idx][1], s = "{:.2%}".format (labels[idx]),
                         size = 10, weight = "bold", ha = "center")
        ax.set_xlabel (input.xLabel (), size = 10); ax.set_ylabel (input.yLabel (), size = 10); fig.tight_layout ()
        return fig


    @reactive.effect
    def _ ():
        if len (nameFuzzySets_X.get ()) == len (idRenameCards_X.get ()):
            return
        allSets = ["PH"] + nameFuzzySets_X.get (); labels = labelValues_X.get ()
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
        currCards = idRenameCards_X.get (); newCards = list ()
        for idx in range (1, len (allSets)):
            prevName = allSets[idx - 1]; prevID = prevName.replace ("-", "_").replace (".", "_")
            currName = allSets[idx]; currID = currName.replace ("-", "_").replace (".", "_")
            if f"rename_{currID}_X" in currCards:
                ui.update_text (f"new_{currID}_X", value = currName)
                ui.update_select (f"colour_{currID}_X", selected = allColours[idx])
            else:
                ui.insert_ui (
                    ui.card (
                        ui.layout_columns (
                            f"{currName}:",
                            ui.input_text (f"new_{currID}_X", "", value = currName, spellcheck = False, width = "200px"),
                            ui.input_select (f"colour_{currID}_X", "", choices = colourDict, selected = allColours[idx], multiple = False)
                        ),
                        id = f"rename_{currID}_X"
                    ),
                    selector = f"#rename_{prevID}_X", where = "afterEnd",
                    multiple = False, immediate = False
                )
            newCards.append (f"rename_{currID}_X")
        for ID in set (currCards) - set (newCards):
            ui.remove_ui (selector = f"#{ID}", multiple = False, immediate = False)
        if currCards != newCards:
            idRenameCards_X.set (newCards)


    @reactive.effect
    def _ ():
        names = ["FS0"] + [input[f"new_{N.replace ("-", "_").replace (".", "_")}_X"] ()
                           for N in nameFuzzySets_X.get () if not N.startswith ("FS0_")]
        ui.update_select ("base_X", choices = dict (zip (range (len (names)), names)))


    @reactive.effect
    def _ ():
        if len (nameFuzzySets_Y.get ()) == len (idRenameCards_Y.get ()):
            return
        allSets = ["PH"] + nameFuzzySets_Y.get (); labels = labelValues_Y.get ()
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
        currCards = idRenameCards_Y.get (); newCards = list ()
        for idx in range (1, len (allSets)):
            prevName = allSets[idx - 1]; prevID = prevName.replace ("-", "_").replace (".", "_")
            currName = allSets[idx]; currID = currName.replace ("-", "_").replace (".", "_")
            if f"rename_{currID}_Y" in currCards:
                ui.update_text (f"new_{currID}_Y", value = currName)
                ui.update_select (f"colour_{currID}_Y", selected = allColours[idx])
            else:
                ui.insert_ui (
                    ui.card (
                        ui.layout_columns (
                            f"{currName}:",
                            ui.input_text (f"new_{currID}_Y", "", value = currName, spellcheck = False, width = "200px"),
                            ui.input_select (f"colour_{currID}_Y", "", choices = colourDict, selected = allColours[idx], multiple = False)
                        ),
                        id = f"rename_{currID}_Y"
                    ),
                    selector = f"#rename_{prevID}_Y", where = "afterEnd",
                    multiple = False, immediate = False
                )
            newCards.append (f"rename_{currID}_Y")
        for ID in set (currCards) - set (newCards):
            ui.remove_ui (selector = f"#{ID}", multiple = False, immediate = False)
        if currCards != newCards:
            idRenameCards_Y.set (newCards)


    @reactive.effect
    def _ ():
        names = ["FS0"] + [input[f"new_{N.replace ("-", "_").replace (".", "_")}_Y"] ()
                           for N in nameFuzzySets_Y.get () if not N.startswith ("FS0_")]
        ui.update_select ("base_Y", choices = dict (zip (range (len (names)), names)))


    @render.download (filename = "results_2aspect.zip")
    def saveFuzzy ():
        items = itemList.get ()
        allFV_X = fuzzyValues_X.get (); allFV_Y = fuzzyValues_Y.get ()
        labels_X = labelValues_X.get (); labels_Y = labelValues_Y.get ()
        concepts_X = allConcepts_X.get (); concepts_Y = allConcepts_Y.get ()
        if addNoiseLeft_X.get ():
            labels_X = ["noiseLeft" if x == plotRangeGlobal_X.get ()[0] else x for x in labels_X]
        if addNoiseRight_X.get ():
            labels_X = ["noiseRight" if x == plotRangeGlobal_X.get ()[1] else x for x in labels_X]
        if addNoiseLeft_Y.get ():
            labels_Y = ["noiseLeft" if x == plotRangeGlobal_Y.get ()[0] else x for x in labels_Y]
        if addNoiseRight_Y.get ():
            labels_Y = ["noiseRight" if x == plotRangeGlobal_Y.get ()[1] else x for x in labels_Y]
        if input.downloadDirection () == "feature":
            maxNum = allFV_X.shape[0] + allFV_Y.shape[0] + 2
        elif input.downloadDirection () == "sample":
            maxNum = allFV_X.shape[1] + allFV_Y.shape[0] + 2
        else:
            maxNum = allFV_X.shape[2] + allFV_Y.shape[2] + 2
        currentDir = os.getcwd (); dirX = input.xLabel (); dirY = input.yLabel ()
        with ui.Progress (min = 0, max = maxNum) as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            with io.BytesIO () as buf:
                tmpDir = f"{date.today().isoformat()}-{np.random.randint(100, 999)}"; os.mkdir (tmpDir)
                os.chdir (tmpDir); os.mkdir (dirX); os.mkdir (dirY)
                with zipfile.ZipFile (buf, "w") as zf:
                    p.set (0, message = "Downloading")
                    constRev = {-np.inf: "-Infinity", np.inf: "Infinity"}
                    with open (f"./{dirX}/fuzzyConcepts.json", "w", encoding = "utf-8") as f:
                        tmp = {feature: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t]
                                         for t in concepts_X[feature]] for feature in concepts_X.keys ()}
                        json.dump (tmp, f, indent = 4)
                    zf.write (f"./{dirX}/fuzzyConcepts.json")
                    os.remove (f"./{dirX}/fuzzyConcepts.json")
                    with open (f"./{dirY}/fuzzyConcepts.json", "w", encoding = "utf-8") as f:
                        tmp = {feature: [[constRev.get (x, x) if not np.isnan (x) else "NaN" for x in t]
                                         for t in concepts_Y[feature]] for feature in concepts_Y.keys ()}
                        json.dump (tmp, f, indent = 4)
                    zf.write (f"./{dirY}/fuzzyConcepts.json")
                    os.remove (f"./{dirY}/fuzzyConcepts.json")
                    p.set (1, "Downloading")
                    defaultNames = nameFuzzySets_X.get ()
                    newNames_X = [input[f"new_{N.replace ("-", "_").replace (".", "_")}_X"] () for N in defaultNames]
                    colours = [input[f"colour_{N.replace ("-", "_").replace (".", "_")}_X"] () for N in defaultNames]
                    summaryDF = pd.DataFrame ({"default name": defaultNames, "new name": newNames_X, "colour": colours})
                    summaryDF.to_csv (f"./{dirX}/fuzzy_set_summary.tsv", index = None, sep = "\t")
                    zf.write (f"./{dirX}/fuzzy_set_summary.tsv")
                    os.remove (f"./{dirX}/fuzzy_set_summary.tsv")
                    defaultNames = nameFuzzySets_Y.get ()
                    newNames_Y = [input[f"new_{N.replace ("-", "_").replace (".", "_")}_Y"] ()
                                  for N in defaultNames]
                    colours = [input[f"colour_{N.replace ("-", "_").replace (".", "_")}_Y"] ()
                               for N in defaultNames]
                    summaryDF = pd.DataFrame ({"default name": defaultNames, "new name": newNames_Y, "colour": colours})
                    summaryDF.to_csv (f"./{dirY}/fuzzy_set_summary.tsv", index = None, sep = "\t")
                    zf.write (f"./{dirY}/fuzzy_set_summary.tsv")
                    os.remove (f"./{dirY}/fuzzy_set_summary.tsv")
                    if input.downloadDirection () == "feature":
                        nameList = items["feature"]
                        for idx in range (allFV_X.shape[0]):
                            p.set (idx + 2, message = "Downloading"); name = nameList[idx]
                            output = pd.DataFrame (allFV_X[idx, :, :], index = items["sample"], columns = newNames_X)
                            output.to_csv (f"./{dirX}/fuzzyValues_{name}.tsv", sep = "\t")
                            zf.write (f"./{dirX}/fuzzyValues_{name}.tsv"); os.remove (f"./{dirX}/fuzzyValues_{name}.tsv")
                        for idx in range (allFV_Y.shape[0]):
                            p.set (allFV_X.shape[0] + idx + 2, message = "Downloading"); name = nameList[idx]
                            output = pd.DataFrame (allFV_Y[idx, :, :], index = items["sample"], columns = newNames_Y)
                            output.to_csv (f"./{dirY}/fuzzyValues_{name}.tsv", sep = "\t")
                            zf.write (f"./{dirY}/fuzzyValues_{name}.tsv"); os.remove (f"./{dirY}/fuzzyValues_{name}.tsv")
                    elif input.downloadDirection () == "sample":
                        nameList = items["sample"]
                        for idx in range (allFV_X.shape[1]):
                            p.set (idx + 2, message = "Downloading"); name = nameList[idx]
                            output = pd.DataFrame (allFV_X[:, idx, :], index = items["feature"], columns = newNames_X)
                            output.to_csv (f"./{dirX}/fuzzyValues_{name}.tsv", sep = "\t")
                            zf.write (f"./{dirX}/fuzzyValues_{name}.tsv"); os.remove (f"./{dirX}/fuzzyValues_{name}.tsv")
                        for idx in range (allFV_Y.shape[1]):
                            p.set (allFV_X.shape[1] + idx + 2, message = "Downloading"); name = nameList[idx]
                            output = pd.DataFrame (allFV_Y[:, idx, :], index = items["feature"], columns = newNames_Y)
                            output.to_csv (f"./{dirY}/fuzzyValues_{name}.tsv", sep = "\t")
                            zf.write (f"./{dirY}/fuzzyValues_{name}.tsv"); os.remove (f"./{dirY}/fuzzyValues_{name}.tsv")
                    else:
                        for idx in range (allFV_X.shape[2]):
                            p.set (idx + 2, message = "Downloading"); name = newNames_X[idx]
                            output = pd.DataFrame (allFV_X[:, :, idx], index = items["feature"], columns = items["sample"])
                            output.to_csv (f"./{dirX}/fuzzyValues_{name}.tsv", sep = "\t")
                            zf.write (f"./{dirX}/fuzzyValues_{name}.tsv"); os.remove (f"./{dirX}/fuzzyValues_{name}.tsv")
                        for idx in range (allFV_Y.shape[2]):
                            p.set (allFV_X.shape[2] + idx + 2); name = newNames_Y[idx]
                            output = pd.DataFrame (allFV_Y[:, :, idx], index = items["feature"], columns = items["sample"])
                            output.to_csv (f"./{dirY}/fuzzyValues_{name}.tsv", sep = "\t")
                            zf.write (f"./{dirY}/fuzzyValues_{name}.tsv"); os.remove (f"./{dirY}/fuzzyValues_{name}.tsv")
                os.chdir (currentDir)
                os.rmdir (f"./{tmpDir}/{dirX}/"); os.rmdir (f"./{tmpDir}/{dirY}/"); os.rmdir (tmpDir)
                yield buf.getvalue ()
        ui.notification_show ("Download Completed", type = "message", duration = 2)


    @render.download (filename = "joint_report.zip")
    def saveEvaluation ():
        print (input.xLabel (), globalConcept_X.get ())
        print (input.yLabel (), globalConcept_Y.get ())
        items = itemList.get (); clusters = clustering.get ()
        if clusters.empty:
            clusters = pd.Series ("TOTAL", index = items["sample"])
            clustering.set (clusters)
        clusters = pd.DataFrame ({"cluster": clusters.values}, index = clusters.index).loc[items["sample"]]
        ui.update_select ("viewConcept_X", selected = conceptInfo_X.get ()["method"])
        ui.update_select ("viewConcept_Y", selected = conceptInfo_Y.get ()["method"])
        labels_X = ["noise" if x == plotRangeGlobal_X.get ()[0] else x for x in labelValues_X.get ()]
        concept_X = globalConcept_X.get (); numFS_X = concept_X.shape[0]; info_X = conceptInfo_X.get ()
        method_X = info_X["method"]; fuzzyBy_X = info_X["direction"]
        typeFS_X = "trap" if method_X == "mode" else input[f"typeFS_{method_X}_X"] ()
        range_X = [0, 100] if fuzzyBy_X == "feature" and typeFS_X == "trap" else rangeGlobal_X.get ()
        baseLevel_X = int (input.base_X ()); maxNumCluster_X = input.maxSpecific_X (); minPctMainFS_X = input.minPercent_X ()
        defaultNames_X = nameFuzzySets_X.get ()
        newNames_X = [input[f"new_{N.replace ("-", "_").replace (".", "_")}_X"] () for N in defaultNames_X]
        renameDict_X = dict (zip (defaultNames_X, newNames_X))
        nameSets_X = [renameDict_X[key] for key in defaultNames_X if not key.startswith ("FS0_")]
        colours_X = [input[f"colour_{N.replace ("-", "_").replace (".", "_")}_X"] ()
                     for N in defaultNames_X if not N.startswith ("FS0_")]
        labels_Y = ["noise" if x == plotRangeGlobal_Y.get ()[0] else x for x in labelValues_Y.get ()]
        concept_Y = globalConcept_Y.get (); numFS_Y = concept_Y.shape[0]; info_Y = conceptInfo_Y.get ()
        method_Y = info_Y["method"]; fuzzyBy_Y = info_Y["direction"]
        typeFS_Y = "trap" if method_Y == "mode" else input[f"typeFS_{method_Y}_Y"] ()
        range_Y = [0, 100] if fuzzyBy_Y == "feature" and typeFS_Y == "trap" else rangeGlobal_Y.get ()
        baseLevel_Y = int (input.base_Y ()); maxNumCluster_Y = input.maxSpecific_Y (); minPctMainFS_Y = input.minPercent_Y ()
        defaultNames_Y = nameFuzzySets_Y.get ()
        newNames_Y = [input[f"new_{N.replace ("-", "_").replace (".", "_")}_Y"] () for N in defaultNames_Y]
        renameDict_Y = dict (zip (defaultNames_Y, newNames_Y))
        nameSets_Y = [renameDict_Y[key] for key in defaultNames_Y if not key.startswith ("FS0_")]
        colours_Y = [input[f"colour_{N.replace ("-", "_").replace (".", "_")}_Y"] ()
                     for N in defaultNames_Y if not N.startswith ("FS0_")]
        currentDir = os.getcwd (); dirX = input.xLabel (); dirY = input.yLabel ()
        with ui.Progress () as p:
            p.set (message = "Download Running", detail = "This will take a while...")
            with io.BytesIO () as buf:
                tmpDir = f"{date.today().isoformat()}-{np.random.randint(100, 999)}"
                os.mkdir (tmpDir); os.chdir (tmpDir)
                os.mkdir (dirX); os.mkdir (f"./{dirX}/markers/")
                os.mkdir (dirY); os.mkdir (f"./{dirY}/markers/")
                with zipfile.ZipFile (buf, "w") as zf:
                    downloadFiles (fuzzyValues_X.get (), items, clusters, labels_X, concept_X, numFS_X,
                                   info_X, typeFS_X, range_X, renameDict_X, colours_X, input.sizeCol_X (),
                                   baseLevel_X, maxNumCluster_X, minPctMainFS_X, dirX)
                    downloadFiles (fuzzyValues_Y.get (), items, clusters, labels_Y, concept_Y, numFS_Y,
                                   info_Y, typeFS_Y, range_Y, renameDict_Y, colours_Y, input.sizeCol_Y (),
                                   baseLevel_Y, maxNumCluster_Y, minPctMainFS_Y, dirY)
                    valueRange_X = rangeGlobal_X.get (); valueRange_Y = rangeGlobal_Y.get ()
                    markers_X = pd.read_csv (f"./{dirX}/marker_statistics.tsv", index_col = None, sep = "\t")
                    markers_Y = pd.read_csv (f"./{dirY}/marker_statistics.tsv", index_col = None, sep = "\t")
                    featureList_X = sorted (set (markers_X["feature"]))
                    featureList_Y = sorted (set (markers_Y["feature"]))
                    impurity_X = pd.read_csv (f"./{dirX}/gini_impurity.tsv", index_col = 0, sep = "\t")
                    impurity_Y = pd.read_csv (f"./{dirY}/gini_impurity.tsv", index_col = 0, sep = "\t")
                    commonMarkers = markers_X.merge (markers_Y, on = ["feature", "cluster", "isMarker"], how = "inner")
                    commonMarkers = list (set (commonMarkers.loc[commonMarkers["isMarker"], "feature"]))
                    intersection_X = getIntersection (concept_X, typeFS_X, valueRange_X)
                    intersection_Y = getIntersection (concept_Y, typeFS_Y, valueRange_Y)
                    pltData = matrix_X.get ().reset_index ().melt (id_vars = "index", value_name = dirX)
                    pltData[dirY] = matrix_Y.get ().melt ()["value"]
                    pltData["label"] = pltData["index"] + "__" + pltData["variable"]
                    pltData["is marker"] = "none"; pltData = pltData.dropna ()
                    tmp = markers_X.loc[markers_X["isMarker"]].copy ()
                    tmp["label"] = tmp["feature"] + "__" + tmp["cluster"]
                    pltData.loc[pltData["label"].isin (tmp["label"]), "is marker"] = dirX
                    tmp = markers_Y.loc[markers_Y["isMarker"]].copy ()
                    tmp["label"] = tmp["feature"] + "__" + tmp["cluster"]
                    pltData.loc[pltData["label"].isin (tmp["label"]), "is marker"] = dirY
                    tmp = markers_X.loc[markers_X["feature"].isin (commonMarkers) & markers_X["isMarker"]].copy ()
                    tmp["label"] = tmp["feature"] + "__" + tmp["cluster"]
                    pltData.loc[pltData["label"].isin (tmp["label"]), "is marker"] = "both"
                    palette = {dirX: "steelblue", dirY: "crimson", "both": "darkmagenta"}
                    fig, ax = plt.subplots (figsize = (8, 8))
                    sns.scatterplot (pltData.loc[pltData["is marker"] == "none"], x = dirX, y = dirY, c = "lightgray",
                                     size = 3, legend = None , ax = ax)
                    sns.scatterplot (pltData.loc[pltData["is marker"] != "none"], x = dirX, y = dirY,
                                     hue = "is marker", hue_order = [dirX, dirY, "both"],
                                     palette = palette, ax = ax)
                    ax.set_xlim (valueRange_X); ax.set_ylim (valueRange_Y); ax.legend (facecolor = "white")
                    for val in intersection_X[1:-1]:
                        ax.axvline (val, color = "black", linestyle = "dashed")
                    for val in intersection_Y[1:-1]:
                        ax.axhline (val, color = "black", linestyle = "dashed")
                    fig.tight_layout (); plt.savefig ("volcano_highlight.png"); del pltData
                    derivation = {"fixed": "by fixed parameters", "width": "by percent of width in raw value range",
                                  "prop": "by porportion of raw values per fuzzy set", "mode": "by estimated density maxima"}
                    direction = {"feature": "per feature", "dataset": "per data set"}
                    data_X = {"xLabel": dirX,
                              "addNoise_X": input.addNoise_X (),
                              "cutoffLeft_X": input.addNoise_X () and (input.minNoiseLevel_X () >= valueRange_X[0]),
                              "cutoffRight_X": input.addNoise_X () and (input.maxNoiseLevel_X () <= valueRange_X[1]),
                              "minNoise_X": input.minNoiseLevel_X (),
                              "maxNoise_X": input.maxNoiseLevel_X (),
                              "labelStr_X": ", ".join ([str (x) for x in labels_X]),
                              "crispStats_X": list ((summarizeCrispMtx_X.data_view ().iloc[4:, [0, 2]].to_dict (orient = "index").values ())),
                              "pctCompleted_X": (np.abs (fuzzyValues_X.get ().sum (axis = 2) - 1) <= 1e-3 + 1e-10).mean (axis = None),
                              "conceptStats_X": [{"statement": "additional fuzzy sets", "value": len (labels_X)},
                                                 {"statement": "fuzzy sets", "value": numFS_X},
                                                 {"statement": "derivation method", "value": derivation[method_X]},
                                                 {"statement": "fuzzificaion direction", "value": direction[fuzzyBy_X]}],
                              "conceptPlot_X": f"./{dirX}/globalConcept.png",
                              "summaryFV_X": f"./{dirX}/summaryFV.png",
                              "pctClear_X": (impurity_X[nameSets_X] >= 0.5).mean (axis = None),
                              "impurityPlot_X": f"./{dirX}/gini_impurity.png",
                              "numSpecific_X": len (featureList_X),
                              "baseLevel_X": nameSets_X[baseLevel_X - 1],
                              "maxNum_X": int (input.maxSpecific_X ()),
                              "minPct_X": input.minPercent_X (),
                              "firstDot_X": f"./{dirX}/markers/marker_scatter_1.png"}
                    data_Y = {"yLabel": dirY,
                              "addNoise_Y": input.addNoise_Y (),
                              "cutoffLeft_Y": input.addNoise_Y () and (input.minNoiseLevel_Y () >= valueRange_Y[0]),
                              "cutoffRight_Y": input.addNoise_Y () and (input.maxNoiseLevel_Y () <= valueRange_Y[1]),
                              "minNoise_Y": input.minNoiseLevel_Y (),
                              "maxNoise_Y": input.maxNoiseLevel_Y (),
                              "labelStr_Y": ", ".join ([str (x) for x in labels_Y]),
                              "crispStats_Y": list ((summarizeCrispMtx_Y.data_view ().iloc[4:, [0, 2]].to_dict (orient = "index").values ())),
                              "pctCompleted_Y": (np.abs (fuzzyValues_Y.get ().sum (axis = 2) - 1) <= 1e-3 + 1e-10).mean (axis = None),
                              "conceptStats_Y": [{"statement": "additional fuzzy sets", "value": len (labels_Y)},
                                                 {"statement": "fuzzy sets", "value": numFS_Y},
                                                 {"statement": "derivation method", "value": derivation[method_Y]},
                                                 {"statement": "fuzzificaion direction", "value": direction[fuzzyBy_Y]}],
                              "conceptPlot_Y": f"./{dirY}/globalConcept.png",
                              "summaryFV_Y": f"./{dirY}/summaryFV.png",
                              "pctClear_Y": (impurity_Y[nameSets_Y] >= 0.5).mean (axis = None),
                              "impurityPlot_Y": f"./{dirY}/gini_impurity.png",
                              "numSpecific_Y": len (featureList_Y),
                              "baseLevel_Y": nameSets_Y[baseLevel_Y - 1],
                              "maxNum_Y": int (input.maxSpecific_Y ()),
                              "minPct_Y": input.minPercent_Y (),
                              "firstDot_Y": f"./{dirY}/markers/marker_scatter_1.png"}
                    data = {"numRows": len (items["feature"]),
                            "numCols": len (items["sample"]),
                            "numCommonSpecific": len (commonMarkers),
                            "annotVolcano": "volcano_highlight.png"}
                    data.update (data_X); data.update (data_Y)
                    with open (os.path.join (os.path.dirname (os.path.realpath (__file__)), "template_2aspect.html"), "r") as f:
                        template = "".join (f.readlines ()); f.close ()
                    content = Template (template).render (**data)
                    with open ("report_2aspect.html", "w") as f:
                        f.write (content); f.close ()
                    zf.write (f"./{dirX}/marker_statistics.tsv"); os.remove (f"./{dirX}/marker_statistics.tsv")
                    zf.write (f"./{dirY}/marker_statistics.tsv"); os.remove (f"./{dirY}/marker_statistics.tsv")
                    zf.write (f"./{dirX}/main_fuzzy_values.tsv"); os.remove (f"./{dirX}/main_fuzzy_values.tsv")
                    zf.write (f"./{dirX}/main_fuzzy_sets.tsv"); os.remove (f"./{dirX}/main_fuzzy_sets.tsv")
                    zf.write (f"./{dirX}/diff_main_fuzzy_values.tsv"); os.remove (f"./{dirX}/diff_main_fuzzy_values.tsv")
                    zf.write (f"./{dirY}/main_fuzzy_values.tsv"); os.remove (f"./{dirY}/main_fuzzy_values.tsv")
                    zf.write (f"./{dirY}/main_fuzzy_sets.tsv"); os.remove (f"./{dirY}/main_fuzzy_sets.tsv")
                    zf.write (f"./{dirY}/diff_main_fuzzy_values.tsv"); os.remove (f"./{dirY}/diff_main_fuzzy_values.tsv")
                    zf.write (f"./{dirX}/gini_impurity.tsv"); os.remove (f"./{dirX}/gini_impurity.tsv")
                    zf.write (f"./{dirY}/gini_impurity.tsv"); os.remove (f"./{dirY}/gini_impurity.tsv")
                    zf.write ("report_2aspect.html"); os.remove ("report_2aspect.html")
                    zf.write (f"./{dirX}/globalConcept.png"); os.remove (f"./{dirX}/globalConcept.png")
                    zf.write (f"./{dirX}/summaryFV.png"); os.remove (f"./{dirX}/summaryFV.png")
                    zf.write (f"./{dirX}/gini_impurity.png"); os.remove (f"./{dirX}/gini_impurity.png")
                    zf.write (f"./{dirY}/globalConcept.png"); os.remove (f"./{dirY}/globalConcept.png")
                    zf.write (f"./{dirY}/summaryFV.png"); os.remove (f"./{dirY}/summaryFV.png")
                    zf.write (f"./{dirY}/gini_impurity.png"); os.remove (f"./{dirY}/gini_impurity.png")
                    for i in range (int (np.ceil (len (featureList_X) / 50))):
                        zf.write (f"./{dirX}/markers/marker_scatter_{50 * i + 1}.png")
                        os.remove (f"./{dirX}/markers/marker_scatter_{50 * i + 1}.png")
                    for i in range (int (np.ceil (len (featureList_Y) / 50))):
                        zf.write (f"./{dirY}/markers/marker_scatter_{50 * i + 1}.png")
                        os.remove (f"./{dirY}/markers/marker_scatter_{50 * i + 1}.png")
                    zf.write ("volcano_highlight.png"); os.remove ("volcano_highlight.png")
                os.rmdir (f"./{dirX}/markers/"); os.rmdir (f"{dirX}")
                os.rmdir (f"./{dirY}/markers/"); os.rmdir (f"{dirY}")
                os.chdir (currentDir); os.rmdir (tmpDir)
                yield buf.getvalue ()
        ui.notification_show ("Download Completed", type = "message", duration = 2, close_button = False)


    @render.plot
    @reactive.event (input.updateVolcanoAnnot)
    def volcanoAnnotated ():
        mtx_X = matrix_X.get (); mtx_Y = matrix_Y.get (); items = itemList.get ()
        if (not conceptInfo_X.get ()) or (not conceptInfo_Y.get ()):
            return
        xRange = rangeGlobal_X.get (); yRange = rangeGlobal_Y.get ()
        if len (xRange) != 2 or len (yRange) != 2:
            return
        xName = input.xLabel (); yName = input.yLabel ()
        mtx_X = mtx_X.replace (labelValues_X.get () + [-np.inf, np.inf], np.nan)
        mtx_Y = mtx_Y.replace (labelValues_Y.get () + [-np.inf, np.inf], np.nan)
        pltData = pd.DataFrame ({xName: mtx_X.loc[items["feature"], items["sample"]].melt ()["value"],
                                 yName: mtx_Y.loc[items["feature"], items["sample"]].melt ()["value"]})
        del mtx_X, mtx_Y
        method = conceptInfo_X.get ()["method"]
        intersection_X = getIntersection (globalConcept_X.get (), input[f"typeFS_{method}_X"] (), xRange)
        method = conceptInfo_Y.get ()["method"]
        intersection_Y = getIntersection (globalConcept_Y.get (), input[f"typeFS_{method}_Y"] (), yRange)
        allSets_X = [f"FS{i}" for i in range (1, numFuzzySets_X.get () + 1)]; numFS_X = numFuzzySets_X.get ()
        allSets_Y = [f"FS{i}" for i in range (1, numFuzzySets_Y.get () + 1)]; numFS_Y = numFuzzySets_Y.get ()
        mainFV_X = mainFuzzyValues_X.get (); mainFS_X = mainFuzzySets_X.get ()
        if mainFV_X.empty or mainFS_X.empty:
            mainFV_X, mainFS_X, _ = getCertaintyStats (fuzzyValues_X.get (), items, numFS_X, labelValues_X.get ())
        mainFV_Y = mainFuzzyValues_Y.get (); mainFS_Y = mainFuzzySets_Y.get ()
        if mainFV_Y.empty or mainFS_Y.empty:
            mainFV_Y, mainFS_Y, _ = getCertaintyStats (fuzzyValues_Y.get (), items, numFS_Y, labelValues_Y.get ())
        classified = (mainFV_X >= input.cutoffMainFV_X ())
        colouring_X = pd.DataFrame ({FS: ((mainFS_X == FS) & classified).melt ()["value"] for FS in allSets_X})
        classified = (mainFV_Y >= input.cutoffMainFV_Y ())
        colouring_Y = pd.DataFrame ({FS: ((mainFS_Y == FS) & classified).melt ()["value"] for FS in allSets_Y})
        colouring = pd.Series (False, index = range (len (items["feature"]) * len (items["sample"])))
        for col1 in allSets_X:
            for col2 in allSets_Y:
                colouring |= (colouring_X[col1] & colouring_Y[col2])
        pltData["classified"] = colouring
        pltData["colour"] = "lightgray"; pltData.loc[colouring == True, "colour"] = "red"
        del mainFS_X, mainFV_X, mainFS_Y, mainFV_Y, classified, colouring, colouring_X, colouring_Y
        category = pd.Series (index = range (len (items["feature"]) * len (items["sample"])), dtype = str)
        for idx in range (numFS_X):
            vMin = intersection_X[idx]; vMax = intersection_X[idx + 1]
            category[(pltData[xName] >= vMin) & (pltData[xName] < vMax)] = allSets_X[idx]
        pltData[f"FS_{xName}"] = category
        category = pd.Series (index = range (len (items["feature"]) * len (items["sample"])), dtype = str)
        for idx in range (numFS_Y):
            vMin = intersection_Y[idx]; vMax = intersection_Y[idx + 1]
            category[(pltData[yName] >= vMin) & (pltData[yName] < vMax)] = allSets_Y[idx]
        pltData[f"FS_{yName}"] = category
        del category
        pltData = pltData.dropna (); num = pltData.shape[0]
        coords = [[(intersection_X[i] + intersection_X[i + 1]) / 2, (intersection_Y[j] + intersection_Y[j + 1]) / 2]
                  for j in range (numFS_Y) for i in range (numFS_X)]
        labels = [pltData.loc[(pltData[f"FS_{xName}"] == C1) & (pltData[f"FS_{yName}"] == C2), "classified"].sum () / num
                  for C2 in allSets_Y for C1 in allSets_X]
        fig, ax = plt.subplots (1, figsize = (10, 10))
        sns.scatterplot (pltData, x = xName, y = yName, c = pltData["colour"], size = 3, legend = None, ax = ax)
        ax.set_xlim (xRange); ax.set_ylim (yRange)
        for val in intersection_X:
            ax.axvline (val, color = "black", linestyle = "dashed")
        for val in intersection_Y:
            ax.axhline (val, color = "black", linestyle = "dashed")
        for idx in range (len (allSets_X) * len (allSets_Y)):
            ax.text (x = coords[idx][0], y = coords[idx][1], s = "{:.2%}".format (labels[idx]),
                     size = 10, weight = "bold", ha = "center")
        fig.tight_layout ()
        return fig



app = App (app_ui, server)


