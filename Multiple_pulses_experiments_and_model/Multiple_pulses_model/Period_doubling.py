#%matplotlib widget
import numpy as np                             
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
from scipy.integrate import odeint, solve_ivp
from matplotlib.gridspec import GridSpec
from datetime import datetime
import scipy.stats as stats
import pandas as pd
import pickle
import multiprocessing
from tqdm import tqdm
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Functions_common_for_model_and_experiments import *
from Period_doubling_pars import *
def run(new_params):
    try:
        function_args = simulation_parameters.copy()
        function_args.update(Pars)
        function_args.update(new_params)
        print(new_params["Aosc"])
        x,kc2 = odeRungeKutta4_p53(f_p53,**function_args)
        #result = { f"Aosc={new_params['Aosc']}" : [x,kc2, function_args]}
        return new_params['Aosc'],x
    except Exception as ex:
        print(f"Error while computing: {new_params} {str(ex)}")
        return new_params['Aosc'],None


if __name__ == '__main__':
    multiprocessing.freeze_support()
    Aosc_vec = simulation_parameters["Aosc_vec"]
    tasks = []

    for Aosc in Aosc_vec:
        tasks.append(dict(Aosc=Aosc))

    print(len(tasks))
    tot = len(tasks)

    n = int(tot/partitions)

    for i in range(partitions):
        start = i * n
        end = start + n - 1
        print(start,end)
        with multiprocessing.Pool(processes=8) as pool:
            # Parallelize the function and collect the results
            t_start = time.time()
            result = pool.map(run, tasks[start:end])
            t_end = time.time()
            diff = t_end - t_start
            print(diff)
            result.sort(key=lambda x: x[0])
            pickle.dump(result, open(f"./Multiple_pulses_experiments/RESUBMISSION/Output/Period_doublin/Aosc_500_dt_{i}.pkl", 'wb'))

