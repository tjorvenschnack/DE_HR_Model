#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created on 2024/05/24 09:22:40
 
@author: tjorvenschnack
"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

def comparison_method(orig:Union[np.ndarray[float], pd.Series], pred:Union[np.ndarray[float], pd.Series], method:str="rmse", ignore:int=None) -> float:
        """Comparison Method

        Compare two arrays of the same size. Supports different comparison methods.
        Omits NaNs
        """

        if isinstance(orig, pd.Series):
            orig = orig.to_numpy()
            pred = pred.to_numpy()

        # Change the slice to compare only a part of the data
        comparison_slice = slice(ignore, None)
        orig = orig[comparison_slice]
        pred = pred[comparison_slice]

        mask = np.isnan(orig) | np.isnan(pred)
        orig = orig[~mask]
        pred = pred[~mask]

        if len(orig) == 0 or len(pred) == 0:
             return np.NaN
        
        if method == "rmse":
            return np.sqrt(mean_squared_error(orig, pred))
        elif method == "mape":
            return mean_absolute_percentage_error(orig, pred) * 100
        elif method == "r2":
            return - r2_score(orig, pred) # minus sign for minimization
        elif method == 'mae':
            return mean_absolute_error(orig, pred)
        elif method == 'me':
            return np.sum(pred - orig) / len(orig)

