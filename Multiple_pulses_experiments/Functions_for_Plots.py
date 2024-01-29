import numpy as np                                     
import matplotlib.pyplot as plt    
from Functions_for_plot_style import set_plot_properties

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
        ax[i].set_xlabel(r"Period external stimulus $T_{ext}$")
        ax[i].set_xticks(range(len(new_index)), new_index)
        ax[i].set_ylabel("Percentage of cells")
    ax[0].set_xlabel(r"Period external stimulus $T_{ext}$")  
    ax[1].set_xlabel(r"Coupling strength $A_{ext}$ [nutlin-3]")   
    
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


def plot_period_doubling_trace(dataset, cell_id,  nutlin_period, nutlin_concentration, offset_nut_pulses_in_plot, xlim_right, figure_path):
    cond1 = dataset["cell_id"]== cell_id
    cond2 = dataset["nutlin_period [hour]"]== nutlin_period
    cond3 = dataset["nutlin_concentration [uM]"]== nutlin_concentration
    subset = dataset.loc[cond1 & cond2 & cond3].reset_index(drop=True)

    trace = subset["p53_trace"][0]
    time = subset["time"][0]/6
    t2 = subset["t2"][0]/6
    tracezone2 = subset["trace_zone2"][0]
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
    plt.ylabel("p53 (a.u.)")
    plt.xlim(right=xlim_right)
    set_plot_properties(ax, style_left_bottom_axis_only = True)

    plt.savefig(figure_path+"Period2_"+str(nutlin_period)+".svg")
    plt.show()



def plot_mode_hopping_trace(dataset, cell_id, nutlin_period, nutlin_concentration, offset_nut_pulses_in_plot, figure_path):
    cond1 = dataset["cell_id"]==cell_id
    cond2 = dataset["nutlin_period [hour]"]==nutlin_period
    cond3 = dataset["nutlin_concentration [uM]"]==nutlin_concentration
    subset = dataset.loc[cond1 & cond2 & cond3].reset_index(drop=True)

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
    plt.ylabel("p53 (a.u.)")
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

    fig, axs = plt.subplots(1,2,figsize=(3.2,1.5))

    # Create a histogram
    axs[0].hist(all_occurrences_uncl,  density=True, alpha=1, color='#701864', edgecolor='black', linewidth = 0.5, label = "Unclassified")
    axs[1].hist(all_occurrences_entr, density=True, alpha=1, color="#DBDBDB", edgecolor='black', linewidth = 0.5,label="Rot number: "+entrain_mode)
    axs[0].set_xlabel("Peak-to-peak distance")
    axs[0].set_ylabel("Density")
    axs[1].set_xlabel("Peak-to-peak distance")
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
    ax.plot(time/6,trace1, color = "#5BAA46",label="Cell 1")
    ax.plot(time/6,trace2, color = "#2C661A", label = "Cell 2")

    # Plot nutlin pulses
    (min_y,max_y) = plt.gca().get_ylim()
    width_nut_pulses = (max_y-min_y)/10
    plt.plot(square_wave_toplot[:,0]/6, width_nut_pulses*square_wave_toplot[:,1]+offset_nut_pulses_in_plot*min_y, color = "black")

    # Plot labels and style
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("p53 (a.u.)")
    set_plot_properties(ax, style_left_bottom_axis_only = True)
    legend = ax.legend(fontsize = 7)
    legend.get_frame().set_linewidth(0.5)

    plt.savefig(figure_path+"Single_cells_chaos_"+str(nutlin_period)+"_"+str(nutlin_concentration)+"_"+str(cell_ids)+".svg",transparent=True)
    plt.show()