#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HRModel.de_hr_model import calibrate_de_hr_model
from HRModel.de_hr_model import DeHRModel


if __name__ == "__main__":
    # Example Usage

    noise = False # optionally add noise to the load

    # Example Input
    load = np.ones(400)
    load[:100] = load[:100] * 50
    load[100:] = load[100:] * 70
    load[200:] = load[200:] * 5/7
    load[300:] = load[300:] * 0
    if noise:
        load = load + np.random.normal(0, 2, size=load.shape) # Add some noise
    load = pd.Series(load)
    load.index = pd.date_range(start=0, periods=len(load), freq='1s')

    # Example HR (simulated)
    mdl = DeHRModel(init_hr=60, K=2, tau=30)
    hr = mdl.predict(load)
    hr = pd.Series(hr, index=load.index)

    # Fit Model
    fitted_model, fitting_error = calibrate_de_hr_model(load, hr)
    print(f"Fitted Model Parameters: {fitted_model.get_param_dict()}")
    print(f"Fitting Error: {fitting_error}")

    # Simulate HR (load used for fitting)
    predicted_hr = fitted_model.predict(load)

    plt.plot(load, label='Load', color='tab:blue')
    plt.plot(hr, label='Heart Rate', color='tab:orange')
    plt.plot(predicted_hr, label='Predicted Heart Rate', linestyle='dashed')
    plt.legend()
    plt.title('HR Model Fitting')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (BPM)')
    plt.show()

    # Predict HR for new load (not used for fitting)
    # Example Input
    new_load = np.ones(400)
    new_load[:100] = new_load[:100] * 70
    new_load[100:] = new_load[100:] * 50
    new_load[200:] = new_load[200:] * 7/5
    new_load[300:] = new_load[300:] * 0
    if noise:
        new_load = new_load + np.random.normal(0, 2, size=new_load.shape) # Add some noise
    new_load = pd.Series(new_load)
    new_load.index = pd.date_range(start=0, periods=len(new_load), freq='1s')

    # Predict HR for new load
    predicted_hr_new_load = fitted_model.predict(new_load)

    plt.plot(new_load, label='New Load', color='tab:blue')
    plt.plot(predicted_hr_new_load, label='Predicted Heart Rate', linestyle='dashed', color='tab:green')
    plt.legend()
    plt.title('HR Model Prediction for New Load')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (BPM)')
    plt.show()