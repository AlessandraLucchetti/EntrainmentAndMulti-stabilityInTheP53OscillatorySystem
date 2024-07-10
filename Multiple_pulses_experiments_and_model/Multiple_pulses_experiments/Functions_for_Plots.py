import numpy as np                                     
import matplotlib.pyplot as plt    
import seaborn as sns

from Functions_for_plot_style import set_plot_properties
from Functions_for_plot_style import background_color
from Functions_Load_smooth_data import extract_cell_info_from_dataset

def make_histo(dataset_grouped,color_list,filename_tosave):
    columns = [[0, 1, 2, 3, 4, 6],[5,6,7,8]]
    f, ax = plt.subplots(1,2, figsize = (10,2.5))
    for i in range(2):
        dataset_grouped_Xcolumns = dataset_grouped.iloc[columns[i], :]*100 #scaled between 0 and 100
        # Left plot: changed Text, const Aext=0.5uM

        # Right plot: changed Aext, const Text = 11h

        index = dataset_grouped_Xcolumns.index
        new_index = index.str.split("CFP_").str[1]
        dataset_grouped_Xcolumns.plot(ax = ax[i],kind='bar', stacked=True, width = 0.85,
                                              color = color_list, edgecolor="black", linewidth = 0.5,
                                             figsize=(3, 1.5))

        ax[i].set_frame_on(False)
        ax[i].tick_params(top=False,
                       bottom=True,
                       left=True,
                       right=False,
                       labelleft=True,
                       labelbottom=True)
        ax[i].set_yticks(range(0, 101, 20))
        ax[i].set_yticklabels([f"{i}" for i in range(0, 101, 20)])
        ax[i].set_xlabel(r"Nutlin period $T_{ext}$")
        ax[i].set_xticks(range(len(new_index)), new_index)
        ax[i].set_ylabel("Percentage of cells")
    ax[0].set_xlabel(r"Nutlin period $T_{ext}$")  
    ax[1].set_xlabel(r"Nutlin concentration")   
    
    ax[0].legend().set_visible(False)
    ax[1].legend(title='', loc='center right', bbox_to_anchor=(2.2, 0.5),frameon=False, fontsize = 7)
    ax[1].set_xticklabels(["0.25uM","0.5uM","1uM","2uM"])
    ax[1].set_title(r"$T_{ext}$ = 11h")
    set_plot_properties(ax[0])
    set_plot_properties(ax[1])
    ax[0].tick_params(axis='x', labelsize=7, width = 0.5)
    ax[1].tick_params(axis='x', labelsize=7, width = 0.5)

    f.set_size_inches(2.8,1.4, forward=True)
    plt.subplots_adjust(wspace=0.5)
    plt.savefig(filename_tosave+".svg",bbox_inches='tight')
    plt.show()


def plot_period_doubling_trace(dataset, cell_id,  nutlin_period, nutlin_concentration, offset_nut_pulses_in_plot, xlim_right, figure_path, bool_norm):
    cond1 = dataset["cell_id"]== cell_id
    cond2 = dataset["nutlin_period [hour]"]== nutlin_period
    cond3 = dataset["nutlin_concentration [uM]"]== nutlin_concentration
    subset = dataset.loc[cond1 & cond2 & cond3].reset_index(drop=True)

    if bool_norm:
        trace = subset["p53_normalized"][0]
        tracezone2 = subset["trace_zone2_normalized"][0]
    else:
        trace = subset["p53_trace"][0]
        tracezone2 = subset["trace_zone2"][0]
    time = subset["time"][0]/6
    t2 = subset["t2"][0]/6
    
    peaks_toplot = subset["peaks"][0]
    square_wave_toplot = np.array(subset["square_signal"][0])

    fig,ax=plt.subplots(figsize=(3.5,1))

    # Plot p53 trace
    plt.plot(time,trace,color="black")
    colors = ["#5BAA46" if i % 2 == 0 else 'tab:blue' for i in range(len(peaks_toplot))]  
    plt.scatter(t2[peaks_toplot], tracezone2[peaks_toplot], color = colors, s = 20)

    #Plot nutlin square wave
    (min_y,max_y) = plt.gca().get_ylim()
    width_nut_pulses = (max_y-min_y)/10
    plt.plot(square_wave_toplot[:,0]/6, width_nut_pulses*square_wave_toplot[:,1]+offset_nut_pulses_in_plot, color = "black")

    plt.xlabel("Time (h)")
    if bool_norm:
        plt.ylabel("p53 level (FC)")
    else:
        plt.ylabel("p53 level (a.u.)")
    plt.xlim(right=xlim_right)
    set_plot_properties(ax, style_left_bottom_axis_only = True)

    plt.savefig(figure_path+"Period2_"+str(nutlin_period)+".svg")
    plt.show()



def plot_mode_hopping_trace(dataset, cell_id, nutlin_period, nutlin_concentration, offset_nut_pulses_in_plot, figure_path, bool_norm):
    cond1 = dataset["cell_id"]==cell_id
    cond2 = dataset["nutlin_period [hour]"]==nutlin_period
    cond3 = dataset["nutlin_concentration [uM]"]==nutlin_concentration
    subset = dataset.loc[cond1 & cond2 & cond3].reset_index(drop=True)

    if bool_norm:
        trace = subset["p53_normalized"][0]
    else:
        trace = subset["p53_trace"][0]
    time = subset["time"][0]/6
    square_wave_toplot = np.array(subset["square_signal"][0])

    fig,ax=plt.subplots(figsize=(3.5,1))
    # Plot p53 trace
    plt.plot(time,trace,color="black")
    
    #Plot nutlin square wave
    (min_y,max_y) = plt.gca().get_ylim()
    width_nut_pulses = (max_y-min_y)/10
    plt.plot(square_wave_toplot[:,0]/6, width_nut_pulses*square_wave_toplot[:,1]+offset_nut_pulses_in_plot*min_y, color = "black")
    
    plt.xlabel("Time (h)")
    if bool_norm:
        plt.ylabel("p53 level (FC)")
    else:
        plt.ylabel("p53 level (a.u.)")
    set_plot_properties(ax, style_left_bottom_axis_only = True)

    plt.savefig(figure_path+"Modehopping_nut_period"+str(nutlin_period)+".svg")
    plt.show()



def make_histo_periods_entrained_vs_unclassified(dataset, nut_period, nut_conc,entrain_mode, figure_path):
    cond1 = dataset["nutlin_period [hour]"] == nut_period
    cond2 = dataset["nutlin_concentration [uM]"] == nut_conc
    cond3 = dataset["entrain_label"] == "Unclassified"
    cond4 = dataset["entrain_label"] == "Entrained"
    cond5 = dataset["entrain_mode"] == entrain_mode
    subset_uncl = dataset.loc[cond1 & cond2 & cond3].reset_index(drop=True)
    subset_entr = dataset.loc[cond1 & cond2 & cond4 & cond5].reset_index(drop=True)
    # Extract the column with arrays
    occurrences_column_uncl = subset_uncl["peak_to_peak_distance"]
    occurrences_column_entr = subset_entr["peak_to_peak_distance"]

    # Flatten the arrays
    all_occurrences_uncl = [item/6 for sublist in occurrences_column_uncl for item in sublist]
    all_occurrences_entr = [item/6 for sublist in occurrences_column_entr for item in sublist]

    fig, axs = plt.subplots(1,2,figsize=(3.2,1.8))

    # Create a histogram
    axs[0].hist(all_occurrences_uncl,  density=True, alpha=1, color='#701864', edgecolor='black', linewidth = 0.5, label = "Unclassified")
    axs[1].hist(all_occurrences_entr, density=True, alpha=1, color="#DBDBDB", edgecolor='black', linewidth = 0.5,label="Rot number: "+entrain_mode)
    axs[0].set_xlabel("Peak-to-peak distance (h)")
    axs[0].set_ylabel("Density")
    axs[1].set_xlabel("Peak-to-peak distance (h)")
    axs[1].set_ylabel("Density")
    axs[0].set_title('Unclassified\n $T_{ext}=$11h')
    axs[1].set_title('1:1 entrained\n $T_{ext}=$11h') #("Nut period: "+str(nut_period)+"h, nut conc: "+str(nut_conc)+"uM")
    set_plot_properties(axs[0])
    set_plot_properties(axs[1])
    #plt.legend()
    fig.tight_layout()
    fig.savefig(figure_path+"Histo_unclassified_vs_entrained.svg")
    plt.show()

def plot_chaos_traces(dataset, cell_ids, nutlin_period, nutlin_concentration, offset_nut_pulses_in_plot, figure_path):
    
    cond1 = dataset["nutlin_period [hour]"] == nutlin_period
    cond2 = dataset["nutlin_concentration [uM]"] == nutlin_concentration
    cond3 = dataset["cell_id"] == cell_ids[0]
    trace1 = dataset.loc[cond1 & cond2 & cond3,"p53_trace"].reset_index(drop=True)[0]
    cond4 = dataset["cell_id"] == cell_ids[1]
    trace2 = dataset.loc[cond1 & cond2 & cond4,"p53_trace"].reset_index(drop=True)[0]   
    square_wave_toplot = np.array(dataset.loc[cond1 & cond2,"square_signal"].reset_index(drop=True)[0])
    time = dataset.loc[cond1 & cond2,"time"].reset_index(drop=True)[0]

    fig,ax=plt.subplots(figsize=(3.5,1))

    # Plot the 2 p53 traces
    ax.plot(time/6,trace1, color = "black",label="Cell 1")
    ax.plot(time/6,trace2, color = "#ff8080ff", label = "Cell 2")

    # Plot nutlin pulses
    (min_y,max_y) = plt.gca().get_ylim()
    width_nut_pulses = (max_y-min_y)/10
    plt.plot(square_wave_toplot[:,0]/6, width_nut_pulses*square_wave_toplot[:,1]+offset_nut_pulses_in_plot*min_y, color = "black")

    # Plot labels and style
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("p53 level (a.u.)")
    set_plot_properties(ax, style_left_bottom_axis_only = True)
    legend = ax.legend(fontsize = 7)
    legend.get_frame().set_linewidth(0.5)
    plt.savefig(figure_path+"Single_cells_chaos_"+str(nutlin_period)+"_"+str(nutlin_concentration)+"_"+str(cell_ids)+".svg")
    plt.show()

def plot_single_cells_zone2(dataset, nut_period, nut_conc, smoothed_traces, entrain_label, entrain_mode, plotallpeaks = False):
    cond1 = dataset["nutlin_period [hour]"] == nut_period
    cond2 = dataset["nutlin_concentration [uM]"] == nut_conc
    cond3 = dataset["entrain_label"] == entrain_label
    cond4 = dataset["entrain_mode"] == entrain_mode
    cond = cond1 & cond2 & cond3 & cond4
    if smoothed_traces:
        traces_column = "trace_zone2_smooth"
    else:
        traces_column = "trace_zone2"
    data_toplot = dataset.loc[cond,traces_column].reset_index(drop=True)
    cell_ids = dataset.loc[cond,"cell_id"].reset_index(drop=True)
    time_toplot = dataset.loc[cond,"t2"].reset_index(drop=True)
    time_nutpeaks_toplot = dataset.loc[cond,"time_nutlin_pulses"].reset_index(drop=True)
    peaks_toplot = dataset.loc[cond,"peaks"].reset_index(drop=True)
    allpeaks_toplot = dataset.loc[cond,"all_peaks"].reset_index(drop=True)
    for i in range(len(data_toplot)):
        plt.figure(figsize=(3,2))
        peaks = peaks_toplot[i]
        allpeaks = allpeaks_toplot[i]
        time = time_toplot[i]
        data = data_toplot[i]
        time_nutpeaks = time_nutpeaks_toplot[i]
        plt.plot(time/6, data, color = "tab:grey")
        plt.vlines(time_nutpeaks/6,min(data),max(data),linestyle = "--",color="tab:grey")
        if plotallpeaks:
            plt.scatter(time[allpeaks]/6, data[allpeaks],color="red")
        plt.scatter(time[peaks]/6, data[peaks],color="tab:blue",zorder=1000)
        plt.title("cell id: "+str(cell_ids[i]))
        plt.show()

def plot_single_cells_allzones(fig,ax,dataset, nut_period, nut_conc, cell_id, bool_norm, 
                               entrain_label, entrain_mode, xlim_right,
                               plotallpeaks = False, offset_nut_pulses_in_plot = 0, 
                               background_color = background_color):
    cond1 = dataset["nutlin_period [hour]"] == nut_period
    cond2 = dataset["nutlin_concentration [uM]"] == nut_conc
    cond3 = dataset["entrain_label"] == entrain_label
    cond4 = dataset["entrain_mode"] == entrain_mode
    cond5 = dataset["cell_id"] == cell_id
    cond = cond1 & cond2 & cond3 & cond4 & cond5
    if bool_norm:
        traces_column = "p53_normalized"
    else:
        traces_column = "p53_trace"
    data_toplot = dataset.loc[cond,traces_column].reset_index(drop=True)
    time_toplot = dataset.loc[cond,"time"].reset_index(drop=True)

    for i in range(len(data_toplot)):
        square_wave_toplot = np.array(dataset.loc[cond1 & cond2,"square_signal"].reset_index(drop=True)[0])
        time = time_toplot[i]
        data = data_toplot[i]
        ax.plot(time/6, data, color = "black", linewidth = 0.7)
        (min_y,max_y) = ax.get_ylim()
        width_nut_pulses = (max_y-min_y)/10
        ax.plot(square_wave_toplot[:,0]/6, width_nut_pulses*square_wave_toplot[:,1]+offset_nut_pulses_in_plot*min_y, color = "black",linewidth = 0.7)
        ax.set_facecolor(background_color)
        ax.set_xlim(left=0, right = xlim_right)

def plot_single_cell_trace_exp(fig,ax,dataset_experimental, cell_id,  
                               nutlin_period, nutlin_concentration, bool_norm,
                               min_p = 0, max_p = -1,colorp = "black",
                               background_color = background_color):
    t2, x_xone2, time_nutpeaks = extract_cell_info_from_dataset(dataset_experimental, nutlin_period,
                                   nutlin_concentration, cell_id, "Zone2", bool_norm)
    plt.plot(t2,x_xone2,color=colorp)
    #plt.title("Cell_id: "+str(cell_id)+ "  nut_per: "+str(nutlin_period)+ "  nutlin_conc: "+str(nutlin_concentration))
    plt.vlines(time_nutpeaks,min(x_xone2),max(x_xone2),linestyle = "--",color="tab:grey")
    plt.xlim([time_nutpeaks[min_p]-1, time_nutpeaks[max_p]])
    ax.set_facecolor(background_color) 


def plot_mean_trace_with_std(fig,ax, dataset, nut_per, nut_conc, background_color, fig_path, square_pulses_bool=False):
    cond1 = dataset["nutlin_period [hour]"] == nut_per
    cond2 = dataset["nutlin_concentration [uM]"] == nut_conc
    cond = cond1 & cond2
    dataset_subset = dataset.loc[cond].reset_index(drop=True)
    num_cells = len(dataset_subset)
    print(num_cells)
    t0 = np.mean(dataset_subset["t2"])[0]/6
    time_nutpeaks = dataset_subset["time_nutlin_pulses"].reset_index(drop=True)[0]/6-t0
    toff=dataset_subset["toff"].reset_index(drop=True)[0]/6-t0
    time = np.mean(dataset_subset["t2"])/6-t0
    mean_p53 = np.mean(dataset_subset["trace_zone2_normalized"])
    std_p53 = 0.6745*np.std(np.array(dataset_subset["trace_zone2_normalized"]))#/np.sqrt(num_cells) # THIS IS INTERQUARTILE RANGE
    if square_pulses_bool:
        offset_nut_pulses_in_plot = 0
        square_wave_toplot = np.array(dataset_subset["square_signal"][0])
        #Plot nutlin square wave
        (min_y,max_y) = plt.gca().get_ylim()
        width_nut_pulses = (max_y-min_y)/3
        plt.plot(square_wave_toplot[:,0]/6-t0, width_nut_pulses*square_wave_toplot[:,1]+offset_nut_pulses_in_plot*min_y, color = "black")
        plt.xlim(min(time),toff)

    else:
        plt.vlines(time_nutpeaks,min(mean_p53-std_p53),max(mean_p53+std_p53),linestyle = "--",color="tab:grey")
    
    plt.plot(time,mean_p53, color = "#4d4d4dff")
    plt.fill_between(time,mean_p53-std_p53,mean_p53+std_p53, alpha = 0.5, facecolor = "gray", edgecolor = None)
    ax.set_facecolor(background_color)
    plt.xlabel("Time from start of external stimulus (h)")
    plt.ylabel("Mean p53 level (FC)")
    plt.ylim(bottom = -0.2)
    plt.savefig(fig_path+"mean_p53_nut_per_"+str(nut_per)+"_nut_conc_"+str(nut_conc)+".svg")

def plot_heatmap(subset, yticks_list,xticks_list,colorbar_ticks, colorbar_tick_labels, fig_path,title):
    trajectories_2d = np.array([traj for traj in subset["trace_zone2_normalized"].values])
    fig,ax=plt.subplots(figsize=(2,8))
    # Plot the heatmap
    sns.heatmap(trajectories_2d, cmap='viridis',vmin=colorbar_ticks[0],vmax=colorbar_ticks[-1], cbar_kws={"orientation":"horizontal","aspect":5,"shrink":0.5})
    # Customize the colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(colorbar_ticks)
    colorbar.set_label("p53 level (FC)")
    colorbar.set_ticklabels(colorbar_tick_labels)
    ax.set_yticks(yticks_list)
    ax.set_yticklabels(yticks_list)
    ax.set_xticks(xticks_list*6)
    ax.set_xticklabels(xticks_list)
    plt.ylabel("Cells")
    plt.xlabel("Time from start of external stimulus (h)")
    plt.savefig(fig_path+"Heatmap"+str(title)+".svg")
