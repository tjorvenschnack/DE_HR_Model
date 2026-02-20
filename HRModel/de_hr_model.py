#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created on 2024/12/16 11:30:38
 
@author: tjorvenschnack
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Union
import pandas as pd
from HRModel.Calibration.calibration_algorithm import calibrate_model

class DeHRModel:
    '''DE HR Model
    
    Set all parameters via the constructor. 
    Then call predict to run the DE HR Model for a certain load.
    Based on Literature (Mongin et al., 2020a; Mongin et al., 2020b).
    '''

    # parameter
    init_hr: float
    K: float 
    tau: float
    init_value: float

    def __init__(self, parameter_list:list[float]=None, 
                 init_hr:float=None, K:float=None, tau:float=None, init_value:float=None):
        '''Instantiate class and set model parameter
        
        If parameter_list is given, it is used to set all parameters (individual parameters are overridden).
        The order for parameter_list is the same as for the individual parameter
        '''

        if parameter_list is not None:
            self.init_hr = parameter_list[0]
            self.K = parameter_list[1]
            self.tau = parameter_list[2]
            if len(parameter_list) > 3:
                self.init_value = parameter_list[3]
            else:
                self.init_value = init_value # none

        else:
            self.init_hr = init_hr
            self.K = K
            self.tau = tau
            self.init_value = init_value

    def __str__(self) -> str:
        output = 'DeHRModel'
        return output
    
    def get_param_dict(self) -> dict:
        '''Get Param Dict
        
        Returns all Model Parameters as dict
        '''
        return {'init_hr': self.init_hr, 'K': self.K, 'tau': self.tau}

    def predict_with_init_value(self, load:np.ndarray[float], temp_init_value:float) -> np.ndarray[float]:
        '''Predict with Initial Value

        Predicts the performance based on the load and a given initial value (initial value attribute is temporarily set)
        '''
        original_init_value = self.init_value
        self.init_value = temp_init_value
        result = self.predict(load)
        self.init_value = original_init_value
        return result

    def predict(self, load:Union[np.ndarray[float], pd.Series]) -> np.ndarray[float]:
        '''Predict
        
        Predicts the performance based on the load
        '''
        
        def ode_system(t, y):
            '''ODE System
            
            ODE System for the DE HR Model
            '''
            x1 = y
            if y > 10000:
                raise ValueError('Heart rate exceeds 10000 bpm, which is not realistic for DE HR Model.')
            dx1dt = (self.K / self.tau) * self._t_index_interp(t, load, load_index_timestamp) - (x1 - self.init_hr) / self.tau
            return [dx1dt]
        
        load_index_timestamp = None
        if isinstance(load, pd.Series):
            # Converts the Datetimes to Unix timestamps, which is the same unit that solve_ivp uses for t
            load_index_timestamp = load.index.map(lambda x: x.timestamp())

        x0 = [self.init_hr] if self.init_value is None else [self.init_value]
        t_span = (0, len(load)-1)
        t_eval = np.linspace(t_span[0], t_span[-1], len(load))
        if isinstance(load, pd.Series):
            t_span = (load.index[0].timestamp(), load.index[-1].timestamp()) 
            t_eval = load.index.map(lambda x: x.timestamp())
        sol = solve_ivp(ode_system, t_span, y0=x0, t_eval=t_eval, method='RK45')

        y = sol.y[0,:]

        if isinstance(load, pd.Series):
            y = pd.Series(y, index=load.index)

        return y
    
    def _t_index_interp(self, t:float, arr:np.ndarray[float], arr_index_timestamp:np.ndarray[float]) -> float:
            '''Time Index Interpolation

            Linear interpolation for float index.
            '''
            if isinstance(arr, pd.Series):
                t_lower = arr.index[arr_index_timestamp <= t].max()
                t_upper = arr.index[arr_index_timestamp >= t].min()
                if pd.isna(t_lower) or pd.isna(t_upper):
                    # Fallback, should not happen
                    print("Fallback should not happen {}, {}".format(t_lower, t_lower))
                    t_lower = int(np.floor(t))
                    t_upper = int(np.ceil(t))
                if t_lower == t_upper: return arr.loc[t_lower]
                return arr.loc[t_lower] + (arr.loc[t_upper] - arr.loc[t_lower]) * (t - t_lower.timestamp()) / (t_upper.timestamp() - t_lower.timestamp())
            t_floor = int(np.floor(t))
            t_ceil = int(np.ceil(t))
            if t_floor == t_ceil: return arr[t_floor]
            return arr[t_floor] + (arr[t_ceil] - arr[t_floor]) * (t - t_floor) / (t_ceil - t_floor) #  / (t_ceil - t_floor) could be ommited because always 1


def calibrate_de_hr_model(load:Union[np.ndarray[float],list[np.ndarray[float]], pd.Series], performance:Union[np.ndarray[float],list[np.ndarray[float]], pd.Series],
                         init_value=None) -> DeHRModel:
    '''Calibrate DE HR Model
    
    Returns a DE HR Model with parameters that provide the "best" fit for the given load and performance.
    Parameters are contrained by upper and lower bounds.
    Load and Performance can be a list of arrays (multiple recordings) or a single array (single recording).
    init_value can be given as an optional parameter, which is used as initial value for solve_ivp. If not given, the init_hr parameter is used as initial value.
    '''

    lb = 0.0001  # lower bound to avoid division by 0
    # parameter bounds
    init_hr_b = (20, 120)
    K_b = (0, 10)
    tau_b = (lb, 180)

    bounds = [init_hr_b, K_b, tau_b]

    # optional initial value for solve_ivp
    if init_value is not None:
        init_value_b = (init_value, init_value) 
        bounds.append(init_value_b)

    print('- DE HR Model -')
    
    return calibrate_model(load, performance, DeHRModel, bounds)
