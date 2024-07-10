import numpy as np                                     
import scipy.signal as scy
import statistics

# UNITS: ALL IN TIMESTEPS (every timestep is 10 min)

############## PARAMETERS FOR ALGORITHM
prom_perc = 0.2
prom = 0
threshold_entrained = 0.15
threshold_mode_hop = 0.15

#################################################################################################

def compute_peaks(cell_info, prominence_perc = prom_perc):
    trace = cell_info["trace_zone2_smooth"]
    t2 = cell_info["t2"]
    nutlin_period = cell_info["nutlin_period [hour]"]
    peaks, peaks_properties = scy.find_peaks(trace, height = (None,None), width = (nutlin_period*6/8,None))
    prominences = scy.peak_prominences(trace, peaks)[0]
    width = peaks_properties['widths']
    mean_prom = np.mean(prominences)
    #std_prom = np.std(prominences)
    #peaks = peaks[abs((peaks_prom[1]['prominences']-mean_prom))<1*std_prom]
    peaks_above_threshold = peaks[prominences>(prominence_perc*mean_prom)]
    cell_info["peaks"] = peaks_above_threshold # index of peaks in zone 2
    cell_info["time_p53_peaks"] = t2[peaks_above_threshold]
    cell_info["all_peaks"] = peaks
    cell_info["peaks_amplitude"] = peaks_properties['peak_heights']
    return cell_info

def compute_peak_to_peak_distance(cell_info):
    peaks = cell_info["peaks"]
    cell_info["peak_to_peak_distance"] = np.diff(cell_info["t2"][peaks])
    return cell_info

def compute_num_peaks_per_interval(time_nut_pulses, time_p53_peaks, nut_period, q): 
    count_list = []   
    for n in range(len(time_nut_pulses) - q, -1, -q):
        start_time = time_nut_pulses[n]
        end_time = time_nut_pulses[n] + q * nut_period * 6

        if end_time <= time_nut_pulses[-1] + nut_period * 6:
            count = sum(start_time <= tpeak < end_time for tpeak in time_p53_peaks)
            count_list.append(count)
        else:
            break

    return np.array(count_list[::-1])                                                                                                          
                                                                
def check_peak_correlation(peaks_amp, window_size, threshold = 2.5):
    peak_amp1, peak_amp2 = peaks_amp[0::window_size], peaks_amp[1::window_size]
    if len(peak_amp1)>= 2 and len(peak_amp2) >= 2:
        peak_correlation = np.abs(np.mean(peak_amp1) - np.mean(peak_amp2)) / np.sqrt(np.std(peak_amp1)**2 + np.std(peak_amp2)**2)
        if peak_correlation > threshold: # Check if peak correlation is above threshold
            return True   
    return False


def is_period_2(trace,peaks):
    return check_peak_correlation(trace[peaks], window_size=2)


def weighted_mean(array):
    n = len(array)
    weights = [i + 1 for i in range(n)]  # Weights proportional to index
    weighted_sum = sum(x * w for x, w in zip(array, weights))
    sum_of_weights = sum(weights)
    mean = weighted_sum / sum_of_weights
    return mean

def compute_mean_rotation_number(peak_to_peak_distance, nutlin_period, weighted = True):
    if weighted:
        mean_peak_to_peak_dist = weighted_mean(peak_to_peak_distance)
    else:
        mean_peak_to_peak_dist = np.mean(peak_to_peak_distance)
    return nutlin_period*6/mean_peak_to_peak_dist

def is_entrained(peak_to_peak_distances, time_p53_peaks, time_nut_pulses, nutlin_period, threshold = threshold_entrained, entrain_modes=np.array([0.5,1,1.5,2,3])):
    p_modes = [1, 1, 3, 2, 3]
    q_modes = [2, 1, 2, 1, 1]
    rotation_number = compute_mean_rotation_number(peak_to_peak_distances,nutlin_period)
    entrained_bool = False
    for i,mode in enumerate(entrain_modes):
        num_peaks_per_interval = compute_num_peaks_per_interval(time_nut_pulses, time_p53_peaks, nutlin_period, q_modes[i])
        if (np.abs(rotation_number-mode) < mode*threshold) and statistics.mode(num_peaks_per_interval[int(len(num_peaks_per_interval)/2):])==p_modes[i]: 
            return mode
    return None


def is_mode_hopping(peak_to_peak_distances, nutlin_period, entrain_modes=np.array([0.5,1,1.5,2,3]), 
                    threshold = threshold_mode_hop):
    rot_num_vec = nutlin_period*6/peak_to_peak_distances
    no_periods_for_modehop = len(rot_num_vec)/3
    count_each_mode = np.zeros(len(entrain_modes))
    for i,mode in enumerate(entrain_modes):
        count_each_mode[i] = np.sum((np.abs(rot_num_vec-mode)<mode*threshold))                  
    count_each_mode_sorted=np.sort(count_each_mode)
    max_count_ind = np.where(count_each_mode==np.max(count_each_mode))
    if np.sum(count_each_mode>=1) > 1 and (count_each_mode_sorted[-1]>no_periods_for_modehop) and count_each_mode_sorted[-2]>no_periods_for_modehop-1: # (no_periods_for_modehop-1)
        # 1) there must be at least 2 modes detected
        # 2) the one that occurs most frequently must hold for at least no_periods_for_modehop
        # 3) the second most frequent must hold at least for no_periods_for_modehop-1
        mode_hopping_states = count_each_mode    
        return True, mode_hopping_states
    else:
        return False, []    
    
    
def find_entrainment_type(cell_info): 
    trace = cell_info["trace_zone2_smooth"]
    allpeaks = cell_info["peaks"]
    peak_to_peak_distance = cell_info["peak_to_peak_distance"]
    nutlin_period = cell_info["nutlin_period [hour]"]
    time_nutlin_pulses = cell_info["time_nutlin_pulses"] # in timesteps
    time_p53_peaks = cell_info["time_p53_peaks"] # in timesteps
    
    mode = is_entrained(peak_to_peak_distance, time_p53_peaks, time_nutlin_pulses, nutlin_period)
    if is_period_2(trace, allpeaks):
        cell_info["entrain_label"] = "Period-doubling"
        cell_info["entrain_mode"] = "Non-entrained"

    else:
        if mode: 
            cell_info["entrain_label"] = "Entrained" 
            cell_info["entrain_mode"] = str(mode) 
        else:
            bool_mode_hop, mode_hop_states = is_mode_hopping(peak_to_peak_distance, nutlin_period)
            if bool_mode_hop:
                cell_info["entrain_label"] = "Mode-hopping"
                cell_info["entrain_mode"] = "Non-entrained"
                cell_info["mode_hopping modes"] = mode_hop_states
            else:
                cell_info["entrain_label"] = "Unclassified"
                cell_info["entrain_mode"] = "Non-entrained"
    return cell_info



def find_smallest_indices(matrix, k):
    flattened = matrix.flatten()
    indices = np.argpartition(flattened, k)[:k]
    rows, cols = np.unravel_index(indices, matrix.shape)
    return rows, cols

def find_unique_pairs(rows,cols):
    a = []
    for i in range(len(rows)):
        a.append(np.sort([rows[i],cols[i]]))
    a = np.array(a)
    a_unique = np.unique(a,axis=0)
    return a_unique
    
    
    
def find_traces_that_start_together(dataset, nut_period, nut_conc, entrain_label, entrain_mode, num_traces_to_find = 20, initial_points_to_compare = 10):
    cond1 = dataset["nutlin_period [hour]"] == nut_period
    cond2 = dataset["nutlin_concentration [uM]"] == nut_conc
    cond3 = dataset["entrain_label"] == entrain_label
    if entrain_label == "Entrained":
        cond4 = dataset["entrain_mode"] == entrain_mode
    else:
        cond4 = True
    subset = dataset.loc[cond1 & cond2 & cond3 & cond4].reset_index(drop=True)
    traces_matrix = np.vstack(dataset.loc[cond1 & cond2 & cond3 & cond4,"trace_zone2"].reset_index(drop=True))    
    
    NL = traces_matrix.shape[0]
    similarity_matrix = np.ones((NL,NL))
    for i in range(NL):
        for j in range(NL):
            similarity_matrix[i,j] = (np.sum(np.abs(traces_matrix[i,0:initial_points_to_compare]-traces_matrix[j,0:initial_points_to_compare])))

    similarity_matrix += np.eye(NL)*1.0e+04 # Increases the diagonal elements of the IIC matrix by 10000 using a scaled identity matrix,
                                            # to add a large penalty to self-comparisons.
    rows, cols = find_smallest_indices(similarity_matrix, num_traces_to_find)
    cell_id_rows = np.array(subset["cell_id"][rows])
    cell_id_cols = np.array(subset["cell_id"][cols])
    rows_cols_unique = find_unique_pairs(cell_id_rows,cell_id_cols)
    return rows_cols_unique

               
        

def group_df(dataset, xcolumn):
    dataset_grouped_by_xcolumn = dataset.groupby("filename")[xcolumn].value_counts(normalize=True).unstack().fillna(0)
    # Reorder the rows
    columns_order = [8,6,7,5,3,4,2,1,0]
    dataset_grouped_by_xcolumn["columns_order"] = columns_order
    dataset_grouped_by_xcolumn = dataset_grouped_by_xcolumn.sort_values("columns_order")
    dataset_grouped_by_xcolumn = dataset_grouped_by_xcolumn.drop(columns = "columns_order")
    return dataset_grouped_by_xcolumn