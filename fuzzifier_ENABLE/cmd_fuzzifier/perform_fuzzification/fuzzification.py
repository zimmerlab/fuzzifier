import numpy as np
import pandas as pd


### inputs:
# rawValues: pandas series of crisp values
# functionParams: function parameters as list of lists
# furtherParams: dictionary of further parameters, including "addIndicator", "indicateValue", "namePrefix", etc.
def fuzzify (rawValues, functionParams, furtherParams = dict ()):
    # Get additional parameters.
    addIndicator = furtherParams.get ("addIndicator", False); namePrefix = furtherParams.get ("namePrefix", "FS")
    # Prepare pandas dataframe for memberships.
    memberships = pd.DataFrame (index = rawValues.index, columns = [f"{namePrefix}{i}" for i in range (1, len (functionParams) + 1)], dtype = float)
    if addIndicator:
        indicateValue = furtherParams.get ("indicateValue", [np.nan])
        crispValues = rawValues.replace (indicateValue, np.nan).to_numpy ()
    else:
        crispValues = rawValues.to_numpy ()
    index = rawValues.index
    for idx in range (len (functionParams)):
        name = f"{namePrefix}{idx + 1}"; params = functionParams[idx]
        if len (params) == 2: # Gaussi membership function
            if idx == 0: # platform on the left
                platform = (crispValues <= params[0]).astype (int)
                memberships[name] = pd.Series (platform + (1 - platform) * np.exp (-((crispValues - params[0]) ** 2) / (2 * (params[1] ** 2))),
                                               index = index)
            elif idx == len (functionParams) - 1: # platform on the right
                platform = (crispValues >= params[0]).astype (int)
                memberships[name] = pd.Series (platform + (1 - platform) * np.exp (-((crispValues - params[0]) ** 2) / (2 * (params[1] ** 2))),
                                               index = index)
            else:
                memberships[name] = pd.Series (np.exp (-((crispValues - params[0]) ** 2) / (2 * (params[1] ** 2))), index = index)
            memberships.loc[memberships[name] < 1e-5, name] = 0
        elif len (params) == 4: # trapezoidal/triangular/crisp membership function
            if params[0] == params[1] and params[1] == params[2] and params[2] == params[3]:
                memberships[name] = pd.Series (0, index = index)
            else:
                if idx == 0: # platform on the left
                    if params[2] == params[3]:
                        memberships[name] = pd.Series (crispValues < params[2], index = index, dtype = float)
                    else:
                        memberships[name] = pd.Series (((params[3] - crispValues) / (params[3] - params[2])).clip (min = 0).clip (max = 1), index = index)
                elif idx == len (functionParams) - 1: # platform on the right
                    if params[0] == params[1]:
                        if functionParams[0][2] == params[0]:
                            memberships[name] = pd.Series (0, index = index)
                        else:
                            memberships[name] = pd.Series (crispValues > params[1], index = index, dtype = float)
                    else:
                        memberships[name] = pd.Series (((params[0] - crispValues) / (params[0] - params[1])).clip (min = 0).clip (max = 1), index = index)
                else:
                    if params[0] == params[1]:
                        leftSlope = np.zeros (len (index))
                    else:
                        leftSlope = (crispValues < params[1]).astype (int) * ((params[0] - crispValues) / (params[0] - params[1])).clip (min = 0)
                    middle = ((crispValues >= params[1]).astype (int) * (crispValues <= params[2]).astype (int))
                    if params[2] == params[3]:
                        rightSlope = np.zeros (len (index))
                    else:
                        rightSlope = (crispValues > params[2]).astype (int) * ((params[3] - crispValues) / (params[3] - params[2])).clip (min = 0)
                    memberships[name] = pd.Series (leftSlope + middle + rightSlope, index = index)
        else:
            raise ValueError ("There should be either 4 parameters (trapezoidal functions) or 2 parameters (Gaussian functions).")
    # Handle indicator value. There would be no indicator fuzzy set by default, otherwise zeros would be excluded and indicated if given no other values.
    if addIndicator:
        memberships.loc[np.isnan (crispValues)] = 0
        for val in indicateValue[::-1]:
            if val is None:
                memberships.insert (0, f"{namePrefix}0_{val}", np.array ([x is None for x in rawValues]).astype (float))
            elif np.isnan (val):
                memberships.insert (0, f"{namePrefix}0_{val}", np.isnan (rawValues).astype (float))
            elif not np.isfinite (val): # -np.inf or np.inf
                if val < 0: # -np.inf
                    memberships.insert (0, f"{namePrefix}0_{val}", (~np.isfinite (rawValues.replace ([None, np.nan, np.inf], 0))).astype (float))
                else: # np.inf
                    memberships.insert (0, f"{namePrefix}0_{val}", (~np.isfinite (rawValues.replace ([None, np.nan, -np.inf], 0))).astype (float))
            else:
                memberships.insert (0, f"{namePrefix}0_{val}", np.array ([x == val for x in rawValues]).astype (float))
    # Check if any feature lacks memberships due to fuzzy concept design (e.g. all cutoffs are the same).
    memberships.loc[memberships.sum (axis = 1) == 0, f"{namePrefix}{len (functionParams) - 1}"] = 1
    return memberships


