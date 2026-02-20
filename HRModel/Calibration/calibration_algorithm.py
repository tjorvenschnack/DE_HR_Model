#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Created on 2024/05/24 09:58:29
 
@author: tjorvenschnack
"""
import functools
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from typing import Union

from HRModel.Calibration.comparison_method import comparison_method


def calibrate_model(load:Union[np.ndarray[float],list[np.ndarray[float]], pd.Series], performance:Union[np.ndarray[float],list[np.ndarray[float]], pd.Series], model_type:type, bounds: list[tuple[float, float]],
                    ignore:int=None) -> Union[object,float]:
    """Calibrate Model Parameters

    Constructs objective function and runs the parameter calibration.
    Returns the fitted model and the fitting error.
    """
    objective_function = functools.partial(_model_objective_function, load, performance, model_type, ignore)
    optimal_parameters, fitting_error = _calibration_algorithm(objective_function=objective_function, bounds=bounds)
    return model_type(optimal_parameters), fitting_error


def _model_objective_function(load:Union[np.ndarray[float],list[np.ndarray[float]], pd.Series], performance:Union[np.ndarray[float],list[np.ndarray[float]], pd.Series], model_type, ignore:int=None, parameter_list:list[float]=None) -> float:
    """Model Objective Function 

    Used as objective function for calibration.
    Creates a Model with the given parameter and predicts based on the load.
    With the comparison_method the accuracy of the predicted_performance is evaluated against the actual performance.
    The accuracy metric is returned.
    In case of an exception, a penalty value (max float) is returned.
    """
    try:
        if isinstance(load, list) and isinstance(performance, list):
            if len(load) != len(performance):
                raise ValueError('load and performance must have the same length')
            accuracy = 0
            for i in range(len(load)):
                model = model_type(parameter_list=parameter_list)
                predicted_performance = model.predict(load[i])
                accuracy += comparison_method(performance[i], predicted_performance, ignore=ignore)
            accuracy = accuracy/len(load)
            return accuracy

        model = model_type(parameter_list=parameter_list)
        predicted_performance = model.predict(load)
        accuracy = comparison_method(performance, predicted_performance, ignore=ignore)
        return accuracy
    except Exception as e:
            print(f"Exception for input {parameter_list}: {e}")
            penalty = np.finfo(np.float64).max
            return penalty
    
def _calibration_algorithm(objective_function, bounds:list, max_iter:int=10, 
              pop_size:int=1, initial_parameters:list=None, method:str='differential_evolution') -> tuple:
    """Calibration
    
    Using an optimization algorithm to find/calibrate optimal parameter for the objective_function.
    The bounds must correpond to the parameter of the objective_function.
    Returns the optimal parameter and the function value (fitting error).
    """

    max_iter_factor = 20
    max_iter = int(max_iter * max_iter_factor)
    pop_size = int(pop_size * max_iter_factor)

    print("Start Optimization")
    start_time = time.perf_counter() 

    method = 'differential_evolution'
    if method == 'differential_evolution':
        res = differential_evolution(objective_function, bounds=bounds, 
                                 maxiter=max_iter, popsize=pop_size, 
                                 x0=initial_parameters, polish=False, seed=42,
                                 workers=8, updating='deferred', tol=0, disp=True)

    finish_time = time.perf_counter()
    optimal_parameter = res.x
    function_value = res.fun
    print(res.message, "Number of Iterations {}".format(res.nit))
    print("Optimization completed in {:.2f} s".format(finish_time-start_time))
    print("Function Value: ", function_value)
    
    return optimal_parameter, function_value