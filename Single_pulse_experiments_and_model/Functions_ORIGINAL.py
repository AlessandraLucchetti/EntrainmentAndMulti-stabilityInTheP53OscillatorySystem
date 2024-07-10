import numpy as np      
import matplotlib.colors                                
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks
import scipy
import matplotlib as mpl
import scipy.io
from matplotlib.gridspec import GridSpec

def smooth_filter(cell_info, nwindow = 18,nsmooth = 5): # smooth the function with gaussian filter
    trace = cell_info["p53_trace"]
    time = cell_info["time"]
#     filtered = lowess(trace, time, )
#     cell_info["p53_trace"] = filtered[:,1]
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
    cell_info["p53_trace"] = np.array(ynew)
    return cell_info

def f_p53(t,x,kc1,kc2,kc3,kc4,kc5,kc6,kc7,kc8, kc9):
    n,m,Mfree,p,Mbound = x 
    dndt=-kc8*n*Mfree +kc9*Mbound #nutlin
    dmdt = kc4*p**2-kc5*m # mdm2 mRNA
    dMfreedt = kc6*m-kc8*n*Mfree-kc7*Mfree + kc9*Mbound # free Mdm2
    dMbounddt = -kc7*Mbound+kc8*n*Mfree-kc9*Mbound # bound Mdm2 (complex with nutlin)
    dpdt = kc1-kc2*Mfree*p/(kc3+p) #p53
    return np.array([dndt,dmdt,dMfreedt,dpdt,dMbounddt])

def odeRungeKutta4_p53(f,x0,dt,simT, tstart_nut,t_ON, t_OFF, nut_const, args = ()):
    t = np.arange(0, simT, dt)
    dose=np.zeros(len(t)) 
    ntot=np.zeros(len(t)) 
    x = np.zeros((len(t),len(x0)))
    x[0,:] = x0
    dt2 = dt/2.0   
    tend_nut = tstart_nut+t_ON
    while tend_nut<=simT: #Define nutlin pulses
        dose[int(tstart_nut/dt):int(tstart_nut/dt)+1]=nut_const# give pulse of nutlin for 1 dt
        ntot[int(tstart_nut/dt):int(tend_nut/dt)] = 1 # set nutlin to 0 outside each interval tstart-tend
        tstart_nut += t_ON+t_OFF
        tend_nut += t_ON+t_OFF          
    
    for n in range(0,len(t)-1):            
        K1 = dt*f(t[n],x[n,:],  *args)
        K2 = dt*f(t[n] + dt2, x[n,:] + 0.5*K1, *args)
        K3 = dt*f(t[n] + dt2, x[n,:] + 0.5*K2, *args)
        K4 = dt*f(t[n] + dt, x[n,:] + K3, *args)
        x[n+1,:] = x[n,:] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        x[n+1,0] = ntot[n]*(x[n+1,0]+dose[n])
        for i in range(4):
            x[n+1,i] = max(0,x[n+1,i])    # set min value to 0
    return x

def compute_PTC(cell_info, ith_peak = 3, distance = None):
    trace = cell_info["p53_trace"]
    tstart_nut = cell_info["time_nut_on"]
    t = cell_info["time"]
    peaks,_ = find_peaks(trace, distance=distance) # find peaks of p53
    tpeaks = t[peaks] # Time of peaks [min]
    tpeaks_before_nutlin = tpeaks[tpeaks<tstart_nut] #Time of peaks befor nutlin pulse to compute mean period
    last_peak_before_nutlin = tpeaks_before_nutlin[-1] # Time of last peak before nutlin
    ith_peak_after_nutlin = tpeaks[tpeaks>tstart_nut][ith_peak-1] # Time of fourth peak after nutlin
    meanT = np.mean(np.diff(tpeaks_before_nutlin[1:]))
    # compute the initial and final phases
    cell_info["phase0"] = np.mod((tstart_nut-last_peak_before_nutlin)/meanT*2*np.pi,2*np.pi)
    cell_info["phase1"] = np.mod((tstart_nut-(ith_peak_after_nutlin-meanT))/meanT*2*np.pi,2*np.pi)
    cell_info["peaks"] = peaks
    return cell_info

def compute_PRC_from_PTC(cell_info):
    phase1 = cell_info["phase1"]
    phase0 = cell_info["phase0"]
    deltaphi=phase1-phase0
    PRC = (deltaphi + np.pi) % (2*np.pi) - np.pi # normalized in [-pi,pi]
    cell_info["PRC"] = PRC
    return cell_info

def plot_PTC(ax,phase0vec,phase1vec,nut_conc,markershape,markercolor):
    ax.plot(phase0vec,phase1vec, markershape,c=markercolor, label = "model", zorder=10);
    ax.set_xlabel("Initial phase"); ax.set_ylabel("Final phase");
    ax.set_xlim([0,2*np.pi]); ax.set_ylim([0,2*np.pi])
    ax.set_xticks([0,np.pi, 2*np.pi]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0,np.pi, 2*np.pi]); ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.title.set_text('nutlin'+str(nut_conc))
    
def plot_PRC(ax,phase0vec,PRC,nut_conc,markershape,markercolor):
    ax.plot(phase0vec,PRC, markershape,c=markercolor, label = 'model', zorder=10);
    ax.set_xlabel("Initial phase"); ax.set_ylabel("Phase shift");
    ax.set_xticks([0, np.pi, 2*np.pi]); ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"]); ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.title.set_text('Phase Response Curve (PRC)')