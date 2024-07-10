
import numpy as np

DIRECTORY_NAME = "./Multiple_pulses_experiments/RESUBMISSION/Output/arnold_tongue_200x200/"
partitions = 200 #625
save_every_n_timesteps = 20


time_scale = 1.103 #1.103 #103
mol_scale = 1
Pars = {
    'kc1': mol_scale*time_scale*10,             # production of p53
    'kc20': time_scale*10,             # degradation of p53 by mdm2
    'kc3': mol_scale*0.12,           # degradation of p53 by mdm2
    'kc4': time_scale*0.01/mol_scale,           # production of mdm2 - mRNA
    'kc5': time_scale*0.25,           # degradation of mdm2 - mRNA
    'kc6':
time_scale*10,             # production of mdm2
    'kc7': time_scale*1,              # degradation of mdm2
}


simulation_parameters = dict(
x0 = np.array([1,1,0]) ,
dt = 0.002,
simT = 600,
Aosc=1,
Tosc = 2*5.50,
tON = 0,
tOFF = 600,
Aosc_vec = np.linspace(0.01,0.9,200),
Tosc_vec = np.linspace(1.6,13.6,200)
)
