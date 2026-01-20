import numpy as np
import pandas as pd



def fuzzify (rawValues, functionParams, furtherParams = dict ()):
    addIndicator = furtherParams.get ("addIndicator", False)
    memberships = pd.DataFrame (index = rawValues.index, columns = [f"FS{i}" for i in range (1, len (functionParams) + 1)], dtype = float)
    if addIndicator:
        indicateValue = furtherParams.get ("indicateValue", [np.nan])
        raw = rawValues.replace (indicateValue, np.nan).to_numpy ()
    else:
        raw = rawValues.to_numpy ()
    index = rawValues.index
    for idx in range (len (functionParams)):
        name = f"FS{idx + 1}"; params = functionParams[idx]
        if len (params) == 2:
            if idx == 0:
                platform = (raw <= params[0]).astype (int)
                memberships[name] = pd.Series (platform + (1 - platform) * np.exp (-((raw - params[0]) ** 2) / (2 * (params[1] ** 2))),
                                               index = index)
            elif idx == len (functionParams) - 1:
                platform = (raw >= params[0]).astype (int)
                memberships[name] = pd.Series (platform + (1 - platform) * np.exp (-((raw - params[0]) ** 2) / (2 * (params[1] ** 2))),
                                               index = index)
            else:
                memberships[name] = pd.Series (np.exp (-((raw - params[0]) ** 2) / (2 * (params[1] ** 2))), index = index)
            memberships.loc[memberships[name] < 1e-5, name] = 0
        elif len (params) == 4:
            if params[0] == params[1] and params[1] == params[2] and params[2] == params[3]:
                memberships[name] = pd.Series (0, index = index)
            else:
                if idx == 0:
                    if params[2] == params[3]:
                        memberships[name] = pd.Series (raw < params[2], index = index, dtype = float)
                    else:
                        memberships[name] = pd.Series (((params[3] - raw) / (params[3] - params[2])).clip (min = 0).clip (max = 1), index = index)
                elif idx == len (functionParams) - 1:
                    if params[0] == params[1]:
                        if functionParams[0][2] == params[0]:
                            memberships[name] = pd.Series (0, index = index)
                        else:
                            memberships[name] = pd.Series (raw > params[1], index = index, dtype = float)
                    else:
                        memberships[name] = pd.Series (((params[0] - raw) / (params[0] - params[1])).clip (min = 0).clip (max = 1), index = index)
                else:
                    if params[0] == params[1]:
                        leftSlope = np.zeros (len (index))
                    else:
                        leftSlope = (raw < params[1]).astype (int) * ((params[0] - raw) / (params[0] - params[1])).clip (min = 0)
                    middle = ((raw >= params[1]).astype (int) * (raw <= params[2]).astype (int))
                    if params[2] == params[3]:
                        rightSlope = np.zeros (len (index))
                    else:
                        rightSlope = (raw > params[2]).astype (int) * ((params[3] - raw) / (params[3] - params[2])).clip (min = 0)
                    memberships[name] = pd.Series (leftSlope + middle + rightSlope, index = index)
        else:
            raise ValueError
    if addIndicator:
        memberships.loc[np.isnan (raw)] = 0
        for val in indicateValue[::-1]:
            if val is None:
                memberships.insert (0, f"FS0_{val}", np.array ([x is None for x in rawValues]).astype (float))
            elif np.isnan (val):
                memberships.insert (0, f"FS0_{val}", np.isnan (rawValues).astype (float))
            elif not np.isfinite (val): # -np.inf or np.inf
                if val < 0: # -np.inf
                    memberships.insert (0, f"FS0_{val}", (~np.isfinite (rawValues.replace ([None, np.nan, np.inf], 0))).astype (float))
                else: # np.inf
                    memberships.insert (0, f"FS0_{val}", (~np.isfinite (rawValues.replace ([None, np.nan, -np.inf], 0))).astype (float))
            else:
                memberships.insert (0, f"FS0_{val}", np.array ([x == val for x in rawValues]).astype (float))
    memberships.loc[memberships.sum (axis = 1) == 0, f"FS{len (functionParams)}"] = 1
    return memberships


