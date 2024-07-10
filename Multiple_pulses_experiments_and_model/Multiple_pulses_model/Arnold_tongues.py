import numpy as np                             
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
import pickle
import multiprocessing
import time
import os
import sys
#sys.path.append('../../')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Functions_common_for_model_and_experiments import *
from Arnold_tongues_pars import *

def run(new_params):
    try:
        simN = new_params["simN"]
        simjk = new_params["simjk"]
        del new_params["simN"]
        del new_params["simjk"]
        function_args = simulation_parameters.copy()
        function_args.update(Pars)
        function_args.update(new_params)
        del function_args["Aosc_vec"]
        del function_args["Tosc_vec"]
        x,kc2 = odeRungeKutta4_p53(f_p53,**function_args)
        res = (x[:,2])[::save_every_n_timesteps]
        return  simN, simjk, res,function_args
    except Exception as ex:
        print(f"Error while computing: {new_params} {str(ex)}")
        return simN,simjk,None,function_args


kc1, kc20,kc3,kc4,kc5,kc6,kc7 = unpack_Pars_list(Pars)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    Aosc_vec = simulation_parameters["Aosc_vec"]
    Tosc_vec = simulation_parameters["Tosc_vec"]
    tasks = []
    os.makedirs(DIRECTORY_NAME, exist_ok=True)

    i=0
    for k,Tosc in enumerate(Tosc_vec):
        for j,Aosc in enumerate(Aosc_vec):
            tasks.append(dict(
                            simN=i,
                            simjk = [j,k],
                            Aosc=Aosc,
                            Tosc=Tosc
                                ))
            i+=1

    tot = len(tasks)
    
    n = int(tot/partitions)
    cores = multiprocessing.cpu_count()
    print("Running with: ", cores, "cores")
    for i in range(partitions):
        start = i * n
        end = start + n
        with multiprocessing.Pool(processes=cores) as pool:
            # Parallelize the function and collect the results
            t_start = time.time()
            result = pool.map(run, tasks[start:end])
            t_end = time.time() 
            result.sort(key=lambda x: x[0])
            pickle.dump(result, open(DIRECTORY_NAME+"partition_"+str(i)+".pkl", 'wb'))
            print("Finished partition", i, "time: ", t_end - t_start)


