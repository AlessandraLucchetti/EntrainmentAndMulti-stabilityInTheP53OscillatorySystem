import numpy as np                                     
from scipy.signal import medfilt


def split_trajectories_in_3_zones(cell_info):
    trace = cell_info["p53_trace"]
    ton = cell_info["ton"]
    toff = cell_info["toff"]
    num_nutlin_pulses = cell_info["nutlin_pulses"]
    tend_zone2 = int(toff+(toff-ton)/(2*num_nutlin_pulses)) # add the final off time to have full periods of nutlin in t2
    len_trace = len(trace)
    time = np.linspace(1,len_trace,len_trace) # Unit: timesteps
    cell_info["time"] = time
    cell_info["trace_zone1"] = trace[:ton];  cell_info["t1"] = time[:ton]         
    cell_info["trace_zone2"] = trace[ton:tend_zone2];  cell_info["t2"] = time[ton:tend_zone2]
    cell_info["trace_zone3"] = trace[tend_zone2:];  cell_info["t3"] = time[tend_zone2:]
    return cell_info

def compute_position_nutlin_pulses(cell_info):
    time_nutlin_pulses = [cell_info["ton"]] 
    while time_nutlin_pulses[-1]+cell_info["nutlin_period [hour]"]*6<(cell_info["toff"]):
        time_nutlin_pulses = np.append(time_nutlin_pulses,time_nutlin_pulses[-1]+cell_info["nutlin_period [hour]"]*6)
        
    cell_info["time_nutlin_pulses"] = time_nutlin_pulses # in timesteps

    return cell_info

def compute_nutlin_square_signal(cell_info):
    square_signal = [[0,0]]
    time_nutlin_pulses = cell_info["time_nutlin_pulses"]
    T = cell_info["nutlin_period [hour]"]
    time = cell_info["time"]
    for pulses in time_nutlin_pulses:
        square_signal.append([pulses,0])
        square_signal.append([pulses,1])
        square_signal.append([pulses+T*3,1])
        square_signal.append([pulses+T*3,0])
    square_signal.append([time[-1],0])
    cell_info["square_signal"] = square_signal # in timesteps
    return cell_info


def subtract_poly_fit(cell_info, npol=3): # fit to polynomial of degree npol
    trace = cell_info["trace_zone2"]
    time = cell_info["t2"]-cell_info["t2"][0]
    fy = np.polyfit(time,trace,npol) # fy are the coefficients in descending order
    nf = 0
    for i in range(npol+1):
        nf += fy[i]*time**(npol-i)
    cell_info["trace_zone2_smooth"] = trace-(nf) # subtract from the original signal the polynomial fit
    return  cell_info

def smooth_filter_median(cell_info, kernel_size=5): #median filter to remove the outliers
    trace = cell_info["trace_zone2_smooth"]
    smoothed_trace = medfilt(trace, kernel_size=kernel_size)
    cell_info["trace_zone2_smooth"] = smoothed_trace
    return cell_info


def smooth_filter_polyn(cell_info, nwindow = 5,nsmooth = 3): # smooth the function with gaussian filter
    trace = cell_info["trace_zone2_smooth"]
    ynew = []

    # Smooth the first segment
    ytmp = trace[:nwindow+1]
    xtmp = np.linspace(1, nwindow+1, nwindow+1)
    fyx = np.polyfit(xtmp, ytmp, nsmooth)
    f = np.polyval(fyx, xtmp)
    ynew.extend(f[:-1])

    # Smooth the middle segments
    for i in range(nwindow, len(trace)-nwindow):
        ytmp = trace[i-nwindow:i+nwindow+1]
        xtmp = np.linspace(1, 2*nwindow+1, 2*nwindow+1)
        fyx = np.polyfit(xtmp, ytmp, nsmooth)
        f = np.polyval(fyx, xtmp)
        ynew.append(f[nwindow])

    # Smooth the last segment
    ytmp = trace[len(trace)-nwindow:]
    xtmp = np.linspace(1,nwindow, nwindow)
    fyx = np.polyfit(xtmp, ytmp, nsmooth)
    f = np.polyval(fyx, xtmp)
    ynew.extend(f)
    cell_info["trace_zone2_smooth"] = np.array(ynew)
    return cell_info