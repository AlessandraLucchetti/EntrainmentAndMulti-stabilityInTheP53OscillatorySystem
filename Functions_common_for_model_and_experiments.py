
import numpy as np
import matplotlib.pyplot as plt 
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

color_background = "#fffdeeff"
color_background_exp = "#f2f2f2ff"
color_3_to_2 = "#ca181eff"
color_2_to_1 = "#389ac1ff"
color_1_to_2 = "#ffcc0eff" #daa400ff"
color_1_to_1 = "#e68bc8ff" #"#ba7ea7ff"
color_red_chaos_cell = "#ff8080ff"

def unpack_Pars_list(chosen_parameters):
    kc1 = chosen_parameters['kc1']
    kc20 = chosen_parameters['kc20']
    kc3 = chosen_parameters['kc3']
    kc4 = chosen_parameters['kc4']
    kc5 = chosen_parameters['kc5']
    kc6 = chosen_parameters['kc6']
    kc7 = chosen_parameters['kc7']
    return kc1, kc20,kc3,kc4,kc5,kc6,kc7

def unpack_sim_pars_list(simulation_parameters):
    x0 = simulation_parameters['x0']
    dt = simulation_parameters['dt']
    simT = simulation_parameters['simT']
    kc8 = simulation_parameters['kc8']
    nut_const_vec_model = simulation_parameters['nut_const_vec_model']
    return x0, dt, simT, kc8, nut_const_vec_model

######## Functions to simulate ##########

def f_p53(t,x,kc1,kc2,kc3,kc4,kc5,kc6,kc7):
    m,M,p = x 
    dmdt = kc4*p**2-kc5*m # mdm2 mRNA 3*p**2/(p**2+100**2)
    dMdt = kc6*m-kc7*M  # free Mdm2
    dpdt = kc1-kc2*M*p/(kc3+p) #p53
    return np.array([dmdt,dMdt,dpdt])


def odeRungeKutta4_p53(f,x0,dt,simT, Aosc, Tosc, tON, tOFF, kc1, kc20,kc3,kc4,kc5,kc6,kc7):

    '''tstart_nut : time of start of nutlin stimulation
    tend_nut : time of end of nutlin stimulation
    t_ON : time interval nutlin is left in the system
    t_OFF : time interval nutlin is washed off
    if t_ON = t_OFF = Text / 2 the system has equal times on and off of nutlin'''

    t = np.arange(0, simT, dt)
    x = np.zeros((len(t),len(x0)))
    x[0,:] = x0
    dt2 = dt/2.0   
    kc2 = np.zeros(len(t))
    kc2[0:int(tON/dt)] = kc20
    kc2[int(tON/dt):int(tOFF/dt)] = kc20/2*(Aosc)*(np.cos(2*np.pi*(t[int(tON/dt):int(tOFF/dt)]-tON)/Tosc)-1)+kc20
    kc2[int(tOFF/dt):] = kc20
    for n in range(0,len(t)-1):            
        K1 = dt*f(t[n],x[n,:],  kc1,kc2[n],kc3,kc4,kc5,kc6,kc7)
        K2 = dt*f(t[n] + dt2, x[n,:] + 0.5*K1, kc1,kc2[n],kc3,kc4,kc5,kc6,kc7)
        K3 = dt*f(t[n] + dt2, x[n,:] + 0.5*K2, kc1,kc2[n],kc3,kc4,kc5,kc6,kc7)
        K4 = dt*f(t[n] + dt, x[n,:] + K3, kc1,kc2[n],kc3,kc4,kc5,kc6,kc7)
        x[n+1,:] = x[n,:] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        for i in range(len(x0)):
            x[n+1,i] = max(0,x[n+1,i])    # set min value to 0
    return x,kc2

def odeEuler_p53_with_noise(f,x0,dt,simT, Aosc, Tosc, tON, tOFF, D, kc1, kc20,kc3,kc4,kc5,kc6,kc7):
    '''tstart_nut : time of start of nutlin stimulation
    tend_nut : time of end of nutlin stimulation
    t_ON : time interval nutlin is left in the system
    t_OFF : time interval nutlin is washed off
    if t_ON = t_OFF = Text / 2 the system has equal times on and off of nutlin'''

    t = np.arange(0, simT, dt)
    x = np.zeros((len(t),len(x0)))
    x[0,:] = x0
    
    kc2 = np.zeros(len(t))
    kc2[0:int(tON/dt)] = kc20
    kc2[int(tON/dt):int(tOFF/dt)] = kc20/2*(Aosc)*(np.cos(2*np.pi*(t[int(tON/dt):int(tOFF/dt)]-tON)/Tosc)-1)+kc20
    kc2[int(tOFF/dt):] = kc20
    lenx = len(x0)
    for n in range(0,len(t)-1):            
        x[n+1,:] = x[n,:] + dt*f(t[n], x[n,:],kc1,kc2[n],kc3,kc4,kc5,kc6,kc7)+np.sqrt(2*D*dt)*np.random.randn(lenx)
        for i in range(lenx):
            x[n+1,i] = max(0,x[n+1,i])    # set min value to 0
    return x,kc2


def odeRungeKutta4_p53_single_pulse(f,x0,dt,simT, f_k2, kc8, tstart_nut, tend_nut, kc1, kc20,kc3,kc4,kc5,kc6,kc7):

    '''tstart_nut : time of start of nutlin stimulation
    tend_nut : time of end of nutlin stimulation
    t_ON : time interval nutlin is left in the system
    t_OFF : time interval nutlin is washed off
    if t_ON = t_OFF = Text / 2 the system has equal times on and off of nutlin'''

    t = np.arange(0, simT, dt)
    x = np.zeros((len(t),len(x0)))
    x[0,:] = x0
    dt2 = dt/2.0   
    kc2 = np.zeros(len(t))
    kc2[0:int(tstart_nut/dt)] = kc20
    kc2[int(tstart_nut/dt):int(tend_nut/dt)] = kc20*f_k2+(kc20-kc20*f_k2)*(1-np.exp(-kc8*(t[int(tstart_nut/dt):int(tend_nut/dt)]-t[int(tstart_nut/dt)])))
    kc2[int(tend_nut/dt):] = kc20
    for n in range(0,len(t)-1):            
        K1 = dt*f(t[n],x[n,:],  kc1,kc2[n],kc3,kc4,kc5,kc6,kc7)
        K2 = dt*f(t[n] + dt2, x[n,:] + 0.5*K1, kc1,kc2[n],kc3,kc4,kc5,kc6,kc7)
        K3 = dt*f(t[n] + dt2, x[n,:] + 0.5*K2, kc1,kc2[n],kc3,kc4,kc5,kc6,kc7)
        K4 = dt*f(t[n] + dt, x[n,:] + K3, kc1,kc2[n],kc3,kc4,kc5,kc6,kc7)
        x[n+1,:] = x[n,:] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        for i in range(len(x0)):
            x[n+1,i] = max(0,x[n+1,i])    # set min value to 0
    return x,kc2


############### Functions for single pulses experiments #################

def compute_PTC(cell_info, ith_peak = 1, trace_to_use = "p53_trace", distance = None, prom = 0):
    trace = cell_info[trace_to_use]
    tstart_nut = cell_info["time_nut_on"]
    tend_nut = cell_info["time_nut_off"]
    t = cell_info["time"]
    peaks,_ = find_peaks(trace, distance=distance, prominence = prom) # find peaks of p53
    tpeaks = t[peaks] # Time of peaks [min]
    tpeaks_before_nutlin = tpeaks[tpeaks<tstart_nut] #Time of peaks befor nutlin pulse to compute mean period
    last_peak_before_nutlin = tpeaks_before_nutlin[-1] # Time of last peak before nutlin
    ith_peak_after_nutlin = tpeaks[tpeaks>tstart_nut][ith_peak-1] # Time of first peak after nutlin
    meanT = np.mean(np.diff(tpeaks_before_nutlin[1:]))
    # compute the initial and final phases
    cell_info["phase0"] = np.mod((tstart_nut-last_peak_before_nutlin)/meanT*2*np.pi,2*np.pi)
    cell_info["phase1"] = np.mod((tstart_nut-(ith_peak_after_nutlin-meanT))/meanT*2*np.pi,2*np.pi)
    cell_info["peaks"] = peaks
    cell_info["meanT"] = meanT
    return cell_info

def compute_PRC_from_PTC(cell_info):
    phase1 = cell_info["phase1"]
    phase0 = cell_info["phase0"]
    deltaphi=phase1-phase0
    PRC = (deltaphi + np.pi) % (2*np.pi) - np.pi # normalized in [-pi,pi]
    cell_info["PRC"] = PRC
    return cell_info

def stability_criterium_PRC(PRC):
    PRC_prime = np.gradient(PRC)
    stability_condition = np.where(np.logical_and(PRC_prime<0, PRC_prime>-2))
    return stability_condition



############### Functions for data analysis #############
    

def smooth_filter(cell_info, nwindow=18, nsmooth=5):
    trace = cell_info["p53_trace"]
    p53_trace_smooth = savgol_filter(trace,nwindow,nsmooth)
    cell_info["p53_trace_smooth"] = p53_trace_smooth
    return cell_info


############### Functions for chaos #####################

def compute_lyap_div(trace_1, trace_2, timesteps):
    divergence = np.zeros(len(timesteps))
    deltaZ0 = np.linalg.norm(trace_1[0] - trace_2[0]) # initial separation
    for it, t in enumerate(timesteps):
        deltaZt = np.linalg.norm(trace_1[it] - trace_2[it]) # compute norm of difference of the 2 vectors
        #deltaZt = np.abs(trace2[it]-trace1[it]) 
        divergence[it] = np.log(deltaZt/deltaZ0)
    return divergence

def compute_lyap_exp(timesteps,divergence,startpoint = 0, endpoint = -1):
    model = stats.linregress(timesteps[startpoint:endpoint], divergence[startpoint:endpoint])
    lyapunov_exp = model.slope
    lyapunov_exp_intercept = model.intercept
    lyapunov_exp_error = model.stderr
    return lyapunov_exp, lyapunov_exp_error, lyapunov_exp_intercept


################## Functions for period doubling #######################

def find_peaks_for_period_doubling(t, my_list, unique_peaks_vec, transient = 10):
    peaks_list = []
    time_peaks_list = []
    for x in my_list:
        traces = x[1]
        tracep53 = traces[:,2]
        peaks2, _ = find_peaks(tracep53)
        peaks_list.append(tracep53[peaks2])
        time_peaks_list.append(t[peaks2])

    # Find the unique peaks values
    for i in range(len(peaks_list)):
        unique_peaks_vec.append(np.unique(np.round(peaks_list[i][time_peaks_list[i]>transient],2)))
    return unique_peaks_vec

################## Functions for plots ############################

def plot_divergence_lyap_exp(divergence_vec, t, mean_divergence, startpoint, endpoint, lyapunov_exp, 
                             lyapunov_exp_intercept, lyapunov_exp_error,alpha, color_background,filename):
    fig, ax = plt.subplots(figsize=(3,2))
    if len(divergence_vec)>0:
        for divergence in divergence_vec:    
            plt.plot(t, divergence, color = "gray", alpha = alpha)
    plt.plot(t,mean_divergence,color="black")
    plt.ylabel(r"Log($\delta$Z(t)/$\delta$Z(0))")
    plt.xlabel("Time (h)")
    plt.plot(t[startpoint:endpoint], lyapunov_exp*t[startpoint:endpoint]+lyapunov_exp_intercept, color = "yellow")
    plt.plot(t,np.zeros(len(t)), color= "tab:grey", ls = "--")
    print("Lyapunov exponent "+str(np.round(lyapunov_exp,4))+"+-"+str(np.round(lyapunov_exp_error,8)))
    ax.set_facecolor(color_background)
    fig.savefig(filename)


def plot_period_doubling(Aosc_vec,unique_peaks_vec):
    # PLOT
    plt.figure(figsize=(10,4))
    for i in range(len(Aosc_vec)):
        plt.scatter(Aosc_vec[i]*np.ones(len(unique_peaks_vec[i])), unique_peaks_vec[i], edgecolors = "black", facecolors = "white", linewidths=0.2, s = 0.1)
    #plt.title("Period doubling route to chaos")
    #plt.xlabel("A osc")
    #plt.ylim([0,11])
    #plt.xlim([0.07,0.8])
    #plt.ylabel("p53 amplitude of peaks")

def plot_PTC(ax,ith, phase0vec,phase1vec,nut_conc,markershape,markercolor):
    ax.plot(phase0vec,phase1vec, markershape,c=markercolor, label = "model", zorder=10);
    #ax.set_xlabel(r"Initial phase $\theta_i$"); ax.set_ylabel(r"Final phase $\theta_f$");
    ax.set_xlim([0,2*np.pi]); ax.set_ylim([0,2*np.pi])
    ax.set_xticks([0,np.pi, 2*np.pi]); ax.set_xticklabels(["0", r"$T_{int}/2$", r"$T_{int}$"])
    ax.set_yticks([0,np.pi, 2*np.pi]); 
    if ith==1:
        ax.set_yticklabels(["0", r"$T_{int}/2$", r"$T_{int}$"])
    else:
        ax.set_yticklabels(["", "", ""])
    ax.title.set_text(str(nut_conc)+r" $\mu$M")
    
def plot_PRC(ax,ith, phase0vec,PRC,nut_conc,markershape,markercolor, hor_lines = False):
    ax.plot(phase0vec,PRC, markershape,c=markercolor, label = 'model', zorder=10);
    if hor_lines:
        ax.hlines(min(PRC), min(phase0vec),max(phase0vec), color = markercolor)
        ax.hlines(max(PRC), min(phase0vec),max(phase0vec), color = markercolor)
        ax.annotate('', xy=(2*np.pi-0.5, min(PRC)), xytext=(2*np.pi-0.5, max(PRC)),
             arrowprops=dict(arrowstyle='<->', color='black'), zorder = 100)
    #ax.set_xlabel(r"Initial phase $\theta_i$"); ax.set_ylabel(r"Phase shift $\Delta\theta$");
    ax.set_xticks([0, np.pi, 2*np.pi]);         ax.set_yticks([-np.pi, 0, np.pi])
    if ith ==1:
        ax.set_yticklabels([r"$-T_{int}/2$", "0", r"$T_{int}/2$"])
    else:
        ax.set_yticklabels(["", "", ""])

    ax.set_xticklabels(["0", r"$T_{int}/2$", r"$T_{int}$"]); 
    ax.title.set_text(str(nut_conc)+r" $\mu$M")
