import numpy as np

time_scale = 1.103
mol_scale = 1
Pars_exp = {
    'kc1': mol_scale*time_scale*10,             # production of p53
    'kc20': time_scale*10,             # degradation of p53 by mdm2
    'kc3': mol_scale*0.12,           # degradation of p53 by mdm2
    'kc4': time_scale*0.01/mol_scale,           # production of mdm2 - mRNA
    'kc5': time_scale*0.25,           # degradation of mdm2 - mRNA
    'kc6': time_scale*10,             # production of mdm2
    'kc7': time_scale*1,              # degradation of mdm2
}
nut_min = 0.5 # Minimum nutlin given (125 nM)
nut_max = 0.9

simulation_parameters = dict(
    x0 = np.array([0,0,0]), #dmdt,dMdt,dpdt
    dt = 0.005,
    simT = 60,
    kc8 = time_scale/mol_scale*1, #1, #0.01, #0.0036*60,               # binding of Mdm2 and nutlin
    nut_const_vec_model = nut_min+(nut_max-nut_min)*np.array([2**(-i) for i in range(5)])
)
